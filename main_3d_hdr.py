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

from libs.models.mdi_head_avatar import MDIHeadAvatar
from libs.utils.loss import PBRAvatarLoss, psnr, error_map, l1_loss, ssim
from libs.utils.general_utils import load_to_gpu
from libs.utils.camera_utils import get_camera_position
from libs.utils.ckpt_utils import save_full_checkpoint, force_load_model_from_ckpt, model_training_setup, build_lr_scheduler
# from libs.utils.optim import get_current_lrs
from libs.pbr.shade import rasterize_gs
from libs.pbr.light import DirectLightMap
from libs.render.pbr3d import render_stage
from libs.render.render_r3dg import render_base

# from libs.nets.model_avatar import UVPredictorUNet
from libs.nets.model_full import UVPredictorUNet
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

    torch.cuda.set_device(2)
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

    # Load light map 
    env_resolution = 128
    direct_env_light = DirectLightMap(env_resolution)
    direct_env_light.set_optimizable_light(H=env_resolution)
    direct_env_light.to(device)

    # Model training setup (optimizer/scheduler loading)
    model_training_setup(args, cfg_train, head_model, uv_net, direct_env_light)

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
        if 'env_light' in ckpt and hasattr(direct_env_light, 'load_state_dict'):
            try:
                direct_env_light.load_state_dict(ckpt['env_light'], strict=False)
            except Exception:
                pass
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

        # ---- Predict UV-space features and build Gaussian using the SAME geometry ----        
        direct_env_light.get_base_from_lights(None)
        # if viewpoint_cam.lgt_path == '/hdd1/DB/MDI/ENVMAP/rgb_olat/RGB_19_intensity400_diameter40_CS_radianceSGNNLS.hdr':
        #     continue
        
        feat_uv_dict = uv_net(geom, env_map=direct_env_light.base)   
        # {'features_dc': (1,3,H,W), 'opacity': (1,1,H,W)}
        d3_output = head_model.build_gaussian(geom, uv_maps=feat_uv_dict)

        render_output = render_base(viewpoint_cam, d3_output["gaussian"], background)
        render_pbr_output = render_stage(viewpoint_cam=viewpoint_cam,
                                            gs_output=d3_output,
                                            env=direct_env_light,
                                            bg_color=background)
        loss_output   = criterions.forward_hdr(
            render_pbr_output, d3_output, render_output, viewpoint_cam, background,
            light_map=direct_env_light,
            train_normal=False,
        )

        render_output
        loss = loss_output['loss']

        # ---- Optimizer/scheduler steps (match train_3d_model_track) ----
        head_model.optimizer.zero_grad(set_to_none=True)
        uv_net.optimizer.zero_grad(set_to_none=True)
        direct_env_light.optimizer.zero_grad(set_to_none=True)

        loss.backward()

        if head_model.optimizer is not None:
            head_model.optimizer.step()
        if uv_net.optimizer is not None:
            uv_net.optimizer.step()
        if direct_env_light.optimizer is not None:
            direct_env_light.optimizer.step()

        if getattr(head_model, 'scheduler', None) is not None:
            head_model.scheduler.step()
        if getattr(uv_net, 'scheduler', None) is not None:
            uv_net.scheduler.step()
        if getattr(direct_env_light, 'scheduler', None) is not None:
            uv_net.scheduler.step()

        # 1. GT Image / Pred Render / PBR Render
        # 2. GT Normal / Pred Normal / Diffuse
        # 3. Specular / Roughness / Metallic
        # 4. Base Color / Reflectance / Diffuse Light (HDR→톤매핑)
        if iteration == 1 or iteration % 100 == 0:
            with torch.no_grad():
                # --- small helpers ---

                def chw(x):
                    if x is None: return None
                    if x.dim() == 4 and x.shape[0] == 1 and x.shape[-1] == 3:
                        x = x.squeeze(0).permute(2, 0, 1)
                    elif x.dim() == 4 and x.shape[0] == 1 and x.shape[1] in (1, 3):
                        x = x.squeeze(0)
                    elif x.dim() == 3 and x.shape[-1] == 3:
                        x = x.permute(2, 0, 1)
                    elif x.dim() == 2:
                        x = x.unsqueeze(0)
                    return x.contiguous().float()

                to3     = lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1)
                viz01   = lambda x: x.clamp(0.0, 1.0)
                viznorm = lambda x: (x.clamp(-1, 1) * 0.5 + 0.5)  # [-1,1] → [0,1]
                vizhdr  = lambda x: (x / (1.0 + x)).clamp(0.0, 1.0)  # simple Reinhard

                # GT / fallbacks
                gt_image  = viz01(chw(viewpoint_cam.original_image))
                gt_normal = viz01(chw(getattr(viewpoint_cam, 'original_normal', None))) \
                            if getattr(viewpoint_cam, 'original_normal', None) is not None else gt_image
                blank3 = torch.zeros_like(gt_image)

                # convenient getter for dicts
                def get_from(d, key, viz=viz01, make3=False, fallback=None):
                    if isinstance(d, dict) and (key in d) and (d[key] is not None):
                        x = chw(d[key])
                        x = to3(x) if make3 else x
                        return viz(x)
                    return fallback

                # common preds
                pred_render = viz01(chw(render_output['render']))
                pred_normal = viznorm(chw(render_pbr_output['normal']))

                # PBR preds (optional)
                pbr_render   = get_from(render_pbr_output, 'pbr',          viz01,  False, pred_render)
                pbr_diffuse  = get_from(render_pbr_output, 'diffuse',         viz01,  False, pred_render)
                pbr_specular = get_from(render_pbr_output, 'specular',        viz01,  False, pred_render)
                base_color   = get_from(render_output,     'base_color',      viz01,  False, pred_render)
                shading_d  = get_from(render_pbr_output, 'shading_d',  viz01, True, to3(blank3[:1]))
                shading_s   = get_from(render_pbr_output, 'shading_s',   viz01, True, to3(blank3[:1]))

                roughness = get_from(render_output, 'roughness', viz01, True, to3(blank3[:1]))
                metallic  = get_from(render_output, 'metallic',  viz01, True, to3(blank3[:1]))

                # ---- assemble tiles (4 x 3) ----
                tiles = [
                    gt_image,    pred_render, pbr_render,    # row 1
                    gt_normal,   pred_normal, pbr_diffuse,   # row 2
                    pbr_specular, roughness,  metallic,      # row 3
                    base_color,  shading_d, shading_s     # row 4 (추가)
                ]

                # unify spatial size
                Hmin = min(t.shape[1] for t in tiles)
                Wmin = min(t.shape[2] for t in tiles)
                tiles = [F.interpolate(t.unsqueeze(0), size=(Hmin, Wmin),
                                    mode='bilinear', align_corners=False).squeeze(0)
                        for t in tiles]

                grid = make_grid(tiles, nrow=3)  # 4행 × 3열
                save_image(grid, f'{output_path}/img_log_{iteration:06d}.png')
        
        #------------------------ do gaussian maintain ------------------------#
        # head_model._add_densification_stats(viewspace_points, visibility_filter)

        #------------------------ optimize ------------------------#
        # Optimizer and scheduler steps are now handled by model_training_setup (as in train_3d_model_track.py)


        # ------------------------ densify ------------------------
        # if iteration % 1000 == 0:

        #     # do uv densification
        #     old_num = head_model.num_points
        #     if old_num < 200000:
                
        #         head_model._uv_densify(optimizers_group['gs'],
        #             increase_num = min(200000 - old_num, cfg_train["increase_num"]))
                
        #         print(f"Do UV densification, Guassian splats: {old_num} --> {head_model.num_points}.")
        #     else:
        #         print(f"Guassian splats: {old_num} has reached maximum number.")

        # ------------------------ prune ------------------------
        # if iteration % 1000 == 0:
        #     old_num = head_model.num_points
        #     head_model._prune_low_opacity_points(optimizers_group['gs'],
        #                                          min_opacity = 0.005)
            
        #     print(f"Prune low opacity points, Guassian splats: {old_num} --> {head_model.num_points}.")
        # # ------------------------ reset opacity ------------------------
        # if iteration % 60000 == 0 and iteration != 0:
        #     model._reset_opacity(optimizers_group['gs'])

        # ------------------------ save training snapshot ------------------------#
        # if (iteration % 100 == 0) or iteration == 1:
        #     save_path = os.path.join(output_path,"train", "rendering", f'train_step_{iteration:06d}.png')
        #     os.makedirs(os.path.join(output_path, "train","rendering"), exist_ok=True)
        #     model.visualization(render_output, d3_output, save_path, viewpoint_cam, device, background)

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
                env_light=direct_env_light,
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
    parser.add_argument("--gpu_id", required=False, default=2, help="path to the yaml config file")
    parser.add_argument("--train_stage", required=False, default="pbr", help="path to the yaml config file")
    parser.add_argument("--testing_iterations", required=False, default=[1000000])
    parser.add_argument("--saving_iterations", required=False, default=[1000000])
    parser.add_argument("--full_iter", default=1000000)
    parser.add_argument("--white_background", required=False, default=True)
    parser.add_argument("--lgt_type", default=None) # cube or hdr
    # parser.add_argument("--base_path", default="/mnt/Database1/nersemble_export/196_EMO-1-shout+laugh") # /hdd1/csg/data/nersemble/196
    # parser.add_argument("--base_path", default="/hdd1/csg/src/GaussianAvatars_main/data/full_cvpr_bm/bala_maskBelowLine") # /hdd1/csg/data/nersemble/196
    # parser.add_argument("--base_path", default="/hdd1/DB/MDI/250814/KSM_O_PBR/")  # /hdd1/csg/data/nersemble/196
    parser.add_argument("--base_path", default="/hdd1/DB/nersemble_export/nersemble_export/UNION_227")  # /hdd1/csg/data/nersemble/196
    # parser.add_argument("--studio_hdr", default="/hdd1/csg/data/Environment_Maps/rgb_olat/RGB_06_intensity400_diameter40_CS_radianceSGNNLS.hdr")
    parser.add_argument("--output_base_path", default="./output/nersemble")
    parser.add_argument("--resume_checkpoint", 
                        default="/hdd1/leedoright/workspace/gaussian_avatar/output/track/227/final_gt_normal/train/checkpoint/ckpt_040000.pt")
    # parser.add_argument("--resume_checkpoint", default=None)
    args, extras = parser.parse_known_args()
    
    main(args)


# /hdd1/DB/MDI/250814/KSM_O_TRACK