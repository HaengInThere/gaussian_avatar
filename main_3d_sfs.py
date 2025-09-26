from argparse import ArgumentParser, Namespace
import yaml
import os, sys
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import torch
from torchvision.utils import save_image, make_grid
# torch.cuda.empty_cache()
import torch.nn.functional as F
import uuid
from PIL import Image
# from lpipsPyTorch import lpips
from torch.utils.data import DataLoader
from libs.dataset.dataloader import MDIDataMultiview

# import torch.multiprocessing as mp

from libs.utils.graphics_utils import quat_rotate
from libs.models.mdi_head_avatar_shs import MDIHeadAvatar
from libs.utils.loss import PBRAvatarLoss, psnr, error_map, l1_loss, ssim
from libs.utils.general_utils import load_to_gpu
from libs.utils.camera_utils import get_camera_position
from libs.utils.ckpt_utils import save_full_checkpoint, force_load_model_from_ckpt, model_training_setup, build_lr_scheduler
# from libs.utils.optim import get_current_lrs
from libs.pbr.shade import rasterize_gs
from libs.pbr.shlight import SHLight
from libs.render.render_r3dg import render_base

# from libs.nets.model_avatar import UVPredictorUNet
from libs.nets.model_full_shs import UVPredictorUNet
from libs.nets.gaussian_uv_mapper import GaussianUVMapper


from libs.utils.log import prepare_output_and_logger

def training_report(cfg, tb_writer, iteration, losses, elapsed, testing_iterations, 
                    background, 
                    dataset, head_model, uv_net, mapper, train_stage='stage1', device='cuda'):
    # debug entry
    # print(f"[training_report] iter={iteration}")
    if tb_writer:
        if 'rgb_loss' in losses:
            tb_writer.add_scalar('train_loss_patches/rgb_loss', losses['rgb_loss'].item(), iteration)
        if 'vgg_loss' in losses:
            tb_writer.add_scalar('train_loss_patches/vgg_loss', losses['vgg_loss'].item(), iteration)
        if 'scale_loss' in losses:
            tb_writer.add_scalar('train_loss_patches/scale_loss', losses['scale_loss'].item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', losses['loss'].item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        print("[ITER {}] Evaluating".format(iteration))
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras' : dataset.getTestCameras()},
        )

        # switch to eval() during validation to freeze BN/Dropout
        was_train_uv = uv_net.training
        was_train_head = head_model.training
        uv_net.eval(); head_model.eval()

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                num_vis_img = 6
                image_cache = []
                gt_image_cache = []
                for idx, viewpoint in tqdm(enumerate(DataLoader(config['cameras'], shuffle=False, batch_size=None, num_workers=8)), total=len(config['cameras'])):
                    with torch.no_grad():
                        load_to_gpu(viewpoint, device)
                        geom = head_model.forward_geometry(viewpoint)
                        cam_pos = get_camera_position(viewpoint, device)

                        Vv = F.normalize(cam_pos - geom['verts'][0], dim=-1)   # (V, 3)
                        nv_cos = (F.normalize(geom['verts_normal'], dim=1) * Vv).sum(dim=1, keepdim=True)
                        # ---- UV unwrap from current vertex normals (no second FLAME call) ----
                        uv_inputs = mapper.unwrap_to_uv_rasterize(
                            vertex_values=torch.concat([geom['verts_normal'], Vv, nv_cos], dim=1),
                            texture_resolution=cfg['tex_size']
                        )                                   # (H, W, 3)
                        uv_inputs = uv_inputs.permute(2,0,1).unsqueeze(0) # (1,c,H,W)

                        feat_uv_dict = uv_net.forward_static(uv_inputs, flame_cond=geom['params'])
                        d3_output = head_model.build_gaussian(geom, uv_maps=feat_uv_dict)
                        render_output = rasterize_gs(viewpoint, d3_output["gaussian"], background, device)

                        image = torch.clamp(render_output['rgb_image'], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to(device), 0.0, 1.0)
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                        ssim_test += ssim(image, gt_image).mean().double()

                        if len(image_cache) < num_vis_img and idx % 200==0:
                            image_cache.append(image)
                            gt_image_cache.append(gt_image)

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                # lpips_test /= len(config['cameras'])          
                ssim_test /= len(config['cameras'])          
                # Log a batch of cached images once per evaluation
                if tb_writer and image_cache:
                    n_vis = min(num_vis_img, len(image_cache))
                    pred_batch = torch.stack(image_cache[:n_vis], dim=0)  # (N, C, H, W)
                    gt_batch   = torch.stack(gt_image_cache[:n_vis], dim=0)
                    tb_writer.add_images(f"{config['name']}/render", pred_batch, global_step=iteration)
                    tb_writer.add_images(f"{config['name']}/ground_truth", gt_batch, global_step=iteration)
                    tb_writer.add_images(f"{config['name']}/error", (pred_batch - gt_batch).abs(), global_step=iteration)
                print("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    # tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        # restore training modes
        uv_net.train(was_train_uv)
        head_model.train(was_train_head)
        torch.cuda.empty_cache()

     
def main(args):

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    torch.cuda.set_device(0)
    device="cuda"
    cfg_train = cfg["training"]

    # load dataset 
    dataset = MDIDataMultiview(args=args)
    # light_map_list = dataset.getLightList()
    shape, static_offset, id_name = dataset.get_id_params()
    output_path = os.path.join(args.output_base_path, id_name)
    os.makedirs(output_path, exist_ok=True)
    tb_writer = prepare_output_and_logger(cfg, save_path=output_path)

    # Consistent model construction style with train_3d_model_track.py
    head_model = MDIHeadAvatar(cfg=cfg, shape_params=shape, static_offset=static_offset, device='cuda').to(device)
    uv_net = UVPredictorUNet(cfg, uv_size=cfg["tex_size"], device=device).to(device)


    # Model training setup (optimizer/scheduler loading)
    model_training_setup(args, cfg_train, head_model, uv_net)
    sh_light = SHLight(init_coeff=0.0, device=device).to(device)
    sh_light.training_setup(cfg_train)

    # ---- LR scheduler setup (match train_3d_model_track style) ----
    sched_cfg = cfg_train.get('lr_schedule', None)
    total_steps = int(cfg_train.get('full_iter', args.full_iter))
    head_model.scheduler = build_lr_scheduler(head_model.optimizer, sched_cfg, total_steps)
    uv_net.scheduler     = build_lr_scheduler(uv_net.optimizer,     sched_cfg, total_steps)

    mapper = GaussianUVMapper(flame_model=None, device=device,
                            uvcoords=head_model.uvcoords,
                            uvfaces=head_model.uvfaces,
                            tri_faces=head_model.faces)


    start_iter = 1
    train_stage = args.train_stage
    if args.resume_checkpoint is not None and os.path.isfile(args.resume_checkpoint):
        ckpt = torch.load(args.resume_checkpoint, map_location='cpu')
        force_load_model_from_ckpt(head_model, ckpt['head_model'])
        force_load_model_from_ckpt(uv_net, ckpt['uv_net'])

        # start_iter = int(ckpt.get('iteration', 0)) + 1
        start_iter = 1

    full_iterations = cfg['training']['full_iter']
    loader_camera_train = DataLoader(dataset.getTrainCameras(), batch_size=None, num_workers=8, shuffle=True, pin_memory=True, persistent_workers=True)
    iter_camera_train = iter(loader_camera_train)
    
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # loss_func 
    criterions= PBRAvatarLoss(cfg=cfg).to(device)
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(start_iter, full_iterations + 1), desc="Training progress")

    for iteration in range(start_iter, full_iterations + 1):
        iter_start.record()

        try:
            viewpoint_cam = next(iter_camera_train)
        except StopIteration:
            iter_camera_train = iter(loader_camera_train)
            viewpoint_cam = next(iter_camera_train)

        load_to_gpu(viewpoint_cam, device)

        # ---- Single FLAME forward to get geometry-only outputs ----
        geom = head_model.forward_geometry(viewpoint_cam)

        # 1) Canonical UV → Gaussian 빌드
        feat_uv_dict = uv_net(geom)
        d3_output = head_model.build_gaussian(geom, uv_maps=feat_uv_dict)

        # 2) 전역 SH로 per-Gaussian 색 채우기
        g = d3_output["gaussian"]

        # 0) 상수 설정
        BASE_GRAY = 0.7    # 전 가우시안 공통 DC 회색
        FIXED_ALPHA = 1.0  # 전 가우시안 공통 불투명도

        # 1) 가우시안 개수와 쿼터니언 얻기
        R = g.get_rotation                        # (N,4)
        N = R.shape[0]

        # 2) 법선 계산. 로컬 z축을 월드로 회전


        # Debug/aux: check irradiance spread to avoid degenerate constant-light cases
        with torch.no_grad():
            L_dbg = sh_light.irradiance(g.get_normal)  # (N,)
            L_var = L_dbg.var().item()

        # 3) 전역 SH 조명 × 최적화된 알베도/컬러
        #    g._get_albedo: (N,3) 또는 (N,1) 가정. 필요시 3채널로 확장.
        albedo_rgb = g.get_albedo  # optimized per-Gaussian color/albedo in [0,1]
        shaded_dc = sh_light.shade_with_albedo(g.get_normal, albedo_rgb)  # (N,3)
        shaded_dc = shaded_dc.view(N, 1, 3).contiguous()  # (N,1,3)
        # Optional: zero-out higher-order color to enforce DC-only rendering

        # ----------- Gray render --------------
        # if isinstance(g, dict) and "features_rest" in g and g["features_rest"] is not None:
        #     g["features_rest"] = torch.zeros_like(g["features_rest"])
        # elif hasattr(g, "_features_rest") and g._features_rest is not None:
        #     g._features_rest = torch.zeros_like(g._features_rest)

        if isinstance(g, dict):
            g["features_dc"] = shaded_dc
            # g["opacity"]     = torch.full((N, 1), FIXED_ALPHA, device=device, dtype=torch.float32)
        else:
            g._features_dc = shaded_dc
            # g._opacity     = torch.full((N, 1), FIXED_ALPHA, device=device, dtype=torch.float32)
        # ---------- Gray render -------------------- 

        # 5) 베이직 스플랫 렌더
        render_output = render_base(viewpoint_cam, g, background)     # {'render': [1,3,H,W]}

        # 6) 손실은 흑백으로
        pred = torch.clamp(render_output['render'], 0.0, 1.0)
        gt   = torch.clamp(viewpoint_cam.original_image.to(device), 0.0, 1.0)
        # to_gray = lambda x: (0.2126*x[0:1] + 0.7152*x[1:2] + 0.0722*x[2:3])
        # pred_y = to_gray(pred)
        # gt_y   = to_gray(gt)

        rgb_l1 = F.l1_loss(pred, gt)
        ssim_loss = criterions.get_dssim_loss(pred, gt)
        # coeff_reg = 1e-4 * (sh_light.coeffs ** 2).mean()
        loss = ssim_loss
        loss_output = {'loss': loss, 'ssim_loss': ssim_loss, 'rgb_loss': rgb_l1}

        head_model.optimizer.zero_grad(set_to_none=True)
        uv_net.optimizer.zero_grad(set_to_none=True)
        sh_light.optimizer.zero_grad(set_to_none=True)


        loss.backward()

        head_model.optimizer.step()
        uv_net.optimizer.step()
        sh_light.optimizer.step()
        
        if iteration==1 or (iteration % 100 == 0):
            pred_render = torch.clamp(render_output['render'], 0.0, 1.0)
            gt_image    = torch.clamp(viewpoint_cam.original_image.to(device), 0.0, 1.0)
            grid = make_grid([pred_render, gt_image], nrow=2)
            # debug: track light coeff grad magnitude
            if sh_light.coeffs.grad is not None and tb_writer is not None:
                tb_writer.add_scalar('train/light_grad_mean', sh_light.coeffs.grad.abs().mean().item(), iteration)
            if tb_writer is not None:
                tb_writer.add_scalar('train/light_var', L_var, iteration)
            save_image(grid, f'{output_path}/img_log_{iteration:06d}.png')


        #----------------------- log print, tensorboard ------------------------#
        iter_end.record()
        with torch.no_grad():
            if tb_writer is not None:
                if head_model.optimizer is not None and len(head_model.optimizer.param_groups) > 0:
                    tb_writer.add_scalar('train/lr_head', head_model.optimizer.param_groups[0]['lr'], iteration)
                if uv_net.optimizer is not None and len(uv_net.optimizer.param_groups) > 0:
                    tb_writer.add_scalar('train/lr_uv', uv_net.optimizer.param_groups[0]['lr'], iteration)
            ema_loss_for_log = 0.4 * loss_output['loss'] + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                if 'rgb_loss' in loss_output:
                    postfix["rgb_loss"] = f"{loss_output['rgb_loss']:.{4}f}"
                if 'vgg_loss' in loss_output:
                    postfix["vgg_loss"] = f"{loss_output['vgg_loss']:.{4}f}"
                if 'ssim' in loss_output:
                    postfix["ssim"] = f"{loss_output['ssim']:.{4}f}"
                if 'scale_loss' in loss_output:
                    postfix["scale_loss"] = f"{loss_output['scale_loss']:.{4}f}"
                if 'rot_loss' in loss_output:
                    postfix["rot_loss"] = f"{loss_output['rot_loss']:.{4}f}"
                if 'lpips_loss' in loss_output:
                    postfix["lpips_loss"] = f"{loss_output['lpips_loss']:.{4}f}"
                if 'normal_loss' in loss_output:
                    postfix["normal_loss"] = f"{loss_output['normal_loss']:.{5}f}"
                if 'normal_ssim' in loss_output:
                    postfix["normal_ssim"] = f"{loss_output['normal_ssim']:.{5}f}"
                if 'mesh_normal_loss' in loss_output:
                    postfix["mesh_normal_loss"] = f"{loss_output['mesh_normal_loss']:.{5}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == full_iterations:
                progress_bar.close()
        
        if iteration in args.saving_iterations:
            ckp_dir = os.path.join(output_path, "train", "checkpoint")
            os.makedirs(ckp_dir, exist_ok=True)
            # Full checkpoint (모델/옵티마이저/스테이지/iter까지)
            save_full_checkpoint(
                save_dir=ckp_dir,
                iteration=iteration,
                stage=train_stage,
                head_model=head_model,
                uv_net=uv_net,
                cfg=cfg,
            )

            # 기존처럼 포인트 클라우드도 함께 저장
            ply_save_path = os.path.join(output_path, "train", f"pc_{iteration}.ply")
            head_model.save_gs(ply_save_path, d3_output["gaussian"])
        if iteration in args.testing_iterations:
            # Log
            training_report(cfg, tb_writer, iteration, loss_output,
                            iter_start.elapsed_time(iter_end), args.testing_iterations,
                            background,
                            dataset, head_model, uv_net, mapper, train_stage, device=device)



if __name__=="__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", required=False, default='./config/mdi.yaml', help="path to the yaml config file")
    parser.add_argument("--gpu_id", required=False, default=0, help="path to the yaml config file")
    parser.add_argument("--train_stage", required=False, default="full", help="path to the yaml config file")
    parser.add_argument("--testing_iterations", required=False, default=[1000000])
    parser.add_argument("--saving_iterations", required=False, default=[1000000])
    parser.add_argument("--full_iter", default=1000000)
    parser.add_argument("--white_background", required=False, default=True)
    parser.add_argument("--lgt_type", default=None) # cube or hdr
    # parser.add_argument("--base_path", default="/mnt/Database1/nersemble_export/196_EMO-1-shout+laugh") # /hdd1/csg/data/nersemble/196
    # parser.add_argument("--base_path", default="/hdd1/csg/src/GaussianAvatars_main/data/full_cvpr_bm/bala_maskBelowLine") # /hdd1/csg/data/nersemble/196
    # parser.add_argument("--base_path", default="/hdd1/DB/MDI/250814/KSM_O_PBR/")  # /hdd1/csg/data/nersemble/196
    parser.add_argument("--base_path", default="/hdd1/DB/nersemble_export/nersemble_export/UNION_226")  # /hdd1/csg/data/nersemble/196
    # parser.add_argument("--studio_hdr", default="/hdd1/csg/data/Environment_Maps/rgb_olat/RGB_06_intensity400_diameter40_CS_radianceSGNNLS.hdr")
    parser.add_argument("--output_base_path", default="./output/shs")
    parser.add_argument("--resume_checkpoint", 
                        default="/hdd1/leedoright/workspace/gaussian_avatar/output/track_shs/226/final_gt_normal/train/checkpoint/ckpt_050000.pt")
    # parser.add_argument("--resume_checkpoint", default=None)
    args, extras = parser.parse_known_args()
    
    main(args)


# /hdd1/DB/MDI/250814/KSM_O_TRACK