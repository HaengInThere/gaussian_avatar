from argparse import ArgumentParser, Namespace
import yaml
import os, sys
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
# torch.cuda.empty_cache()

import uuid
from PIL import Image
# from lpipsPyTorch import lpips
from torch.utils.data import DataLoader
from libs.dataset.dataloader import MDIDataMultiview


# import torch.multiprocessing as mp

from libs.models.mdi_head_avatar_shs import MDIHeadAvatar
from libs.utils.loss import PBRAvatarLoss, psnr, error_map, l1_loss, ssim
from libs.utils.general_utils import load_to_gpu
from libs.utils.ckpt_utils import save_full_checkpoint, model_training_setup, build_lr_scheduler
from libs.utils.log import prepare_output_and_logger
from libs.nets.model_full_shs import UVPredictorUNet
from libs.render.render_r3dg import render_base
import time


def training_report(cfg, tb_writer, iteration, losses, elapsed, testing_iterations, 
                    background, 
                    dataset, head_model, uv_net, train_stage='base', device='cuda'):

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
                normal_cache = []
                gt_normal_cache = []
                for idx, viewpoint in tqdm(enumerate(DataLoader(config['cameras'], shuffle=False, batch_size=None, num_workers=8)), total=len(config['cameras'])):
                    if len(image_cache) < num_vis_img and idx % 150==0:
                        with torch.no_grad():
                            load_to_gpu(viewpoint, device)
                            geom = head_model.forward_geometry(viewpoint)

                            feat_uv_dict = uv_net(geom)
                            d3_output = head_model.build_gaussian(geom, uv_maps=feat_uv_dict)
                            render_output = render_base(viewpoint, d3_output["gaussian"], background)
            
                            image = torch.clamp(render_output['render'], 0.0, 1.0)
                            gt_image = torch.clamp(viewpoint.original_image.to(device), 0.0, 1.0)
                            
                            # # Normal
                            # if viewpoint.original_normal is not None:

                            #     pred_n = render_output['normal'].to(device)   
                            #     gt_n01 = viewpoint.original_normal.to(device)  
                            #     mask = viewpoint.original_mask.to(device)       
                            #     if mask.ndim == 2:
                            #         mask = mask.unsqueeze(0)
                            #     bg = background[:, None, None] 

                            #     pred_vis = (pred_n.clamp(-1, 1) * 0.5 + 0.5)
                            #     gt_vis   = gt_n01.clamp(0, 1)

                            #     pred_normal = pred_vis * mask + bg * (1.0 - mask)
                            #     gt_normal   = gt_vis   * mask + bg * (1.0 - mask)
                            #     normal_cache.append(pred_normal)
                            #     gt_normal_cache.append(gt_normal)

                            l1_test += l1_loss(image, gt_image).mean().double()
                            psnr_test += psnr(image, gt_image).mean().double()
                            ssim_test += ssim(image, gt_image).mean().double()

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
                    # pred_normal = torch.stack(normal_cache[:n_vis], dim=0)  # (N, C, H, W)
                    # gt_normal   = torch.stack(gt_normal_cache[:n_vis], dim=0)
                    tb_writer.add_images(f"{config['name']}/render", pred_batch, global_step=iteration)
                    tb_writer.add_images(f"{config['name']}/ground_truth", gt_batch, global_step=iteration)
                    tb_writer.add_images(f"{config['name']}/error", (pred_batch - gt_batch).abs(), global_step=iteration)
                    # tb_writer.add_images(f"{config['name']}/normal", pred_normal, global_step=iteration)
                    # tb_writer.add_images(f"{config['name']}/normal_gt",   gt_normal,   global_step=iteration)
                print("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    # tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        head_model.train(was_train_head)
        torch.cuda.empty_cache()

     
def main(args):

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    torch.cuda.set_device(3)
    device="cuda"
    cfg_train = cfg["training"]

    # load dataset 
    dataset = MDIDataMultiview(args=args)
    shape, static_offset, id_name = dataset.get_id_params()
    output_path = os.path.join(args.output_base_path, id_name, f"{args.version}")
    os.makedirs(output_path, exist_ok=True)
    tb_writer = prepare_output_and_logger(cfg, save_path=output_path)

    # load model 
    head_model = MDIHeadAvatar(cfg=cfg, shape_params=shape, static_offset=static_offset, device='cuda').to(device)
    uv_net = UVPredictorUNet(cfg, uv_size=cfg["tex_size"], device=device).to(device)
    print("model loaded")

    model_training_setup(args, cfg_train, head_model, uv_net)

    # ---- LR scheduler setup ----
    sched_cfg = cfg_train.get('lr_schedule', None)
    total_steps = int(cfg_train.get('full_iter', args.full_iter))
    head_model.scheduler = build_lr_scheduler(head_model.optimizer, sched_cfg, total_steps)
    uv_net.scheduler     = build_lr_scheduler(uv_net.optimizer,     sched_cfg, total_steps)

    full_iterations = cfg['training']['full_iter']
    loader_camera_train = DataLoader(dataset.getTrainCameras(), batch_size=None, num_workers=16, shuffle=True, pin_memory=True, persistent_workers=True)
    iter_camera_train = iter(loader_camera_train)
    
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # loss_func 
    criterions = PBRAvatarLoss(cfg=cfg).to(device)
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    start_iter = 1
    progress_bar = tqdm(range(start_iter, full_iterations + 1), desc="Training progress")
    for iteration in range(start_iter, full_iterations + 1):
        iter_start.record()

        try:
            viewpoint_cam = next(iter_camera_train)
        except StopIteration:
            iter_camera_train = iter(loader_camera_train)
            viewpoint_cam = next(iter_camera_train)

        load_to_gpu(viewpoint_cam, device)
        geom = head_model.forward_geometry(viewpoint_cam)

        # Canonical training 
        feat_uv_dict = uv_net.forward_static_geo(geom)

        d3_output = head_model.build_static_gaussian(geom, uv_maps=feat_uv_dict)
        
        render_output = render_base(viewpoint_cam, d3_output["gaussian"], background)

        loss_output = criterions(render_output, d3_output, viewpoint_cam, background, device, train_normal=False)
        loss = loss_output['loss']

        viewspace_points    = render_output['viewspace_points']
        visibility_filter   = render_output['visibility_filter']

        head_model.optimizer.zero_grad(set_to_none=True)
        uv_net.optimizer.zero_grad(set_to_none = True)

        loss.backward()

        if iteration==1 or iteration % 1000 == 0:
            tmp_image = torch.cat([render_output['render'], viewpoint_cam.original_image], dim=-1)
            save_image(tmp_image, f'{output_path}/img_log_{iteration:06d}.png')
            save_image(head_model.canon_color, f'{output_path}/uv_color_{iteration:06d}.png')
            
        #------------------------ do gaussian maintain ------------------------#
        head_model._add_densification_stats(viewspace_points, visibility_filter)

        head_model.optimizer.step()
        uv_net.optimizer.step()

        # ---- LR scheduler step ----
        if head_model.scheduler is not None:
            head_model.scheduler.step()
        if uv_net.scheduler is not None:
            uv_net.scheduler.step()

        # ------------------------ densify ------------------------
        if iteration % cfg_train['densify_interval'] == 0 and iteration > 30000:
            old_num = head_model.num_points
            if old_num < 200000:
                head_model._uv_densify(increase_num = min(200000 - old_num, cfg_train["increase_num"]))
                
                print(f"Do UV densification, Guassian splats: {old_num} --> {head_model.num_points}.")
            else:
                print(f"Guassian splats: {old_num} has reached maximum number.")

        #----------------------- log print, tensorboard ------------------------#
        iter_end.record()
        with torch.no_grad():
            # --- Per-iteration TensorBoard logging for training losses ---
            if tb_writer is not None:
                tb_writer.add_scalar('train/total_loss', loss_output['loss'].item(), iteration)
                # Optional sub-losses if present
                for k in ['rgb_loss','vgg_loss','ssim','scale_loss','rot_loss','lpips_loss','normal_loss','normal_ssim','mesh_normal_loss']:
                    if k in loss_output:
                        val = loss_output[k]
                        tb_writer.add_scalar(f'train/{k}', val.item() if torch.is_tensor(val) else float(val), iteration)
                # Iteration time in milliseconds (CUDA event timing)
                tb_writer.add_scalar('train/iter_time_ms', iter_start.elapsed_time(iter_end), iteration)
                # ---- Log current learning rates ----
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
            save_full_checkpoint(
                save_dir=ckp_dir,
                iteration=iteration,
                stage=args.train_stage,
                head_model=head_model,
                uv_net=uv_net,
                env_light=None,
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
                            dataset, head_model, uv_net, args.train_stage, device=device)
            # if (iteration in args.saving_iterations):
            #     print("[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)
        

if __name__=="__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", required=False, default='./config/mdi.yaml', help="path to the yaml config file")
    parser.add_argument("--gpu_id", required=False, default=3, help="path to the yaml config file")
    parser.add_argument("--train_stage", required=False, default="full", help="path to the yaml config file")
    parser.add_argument("--testing_iterations", required=False, default=[10000, 30000, 50000, 100000, 200000,300000,500000,1000000])
    parser.add_argument("--saving_iterations", required=False, default=[1000, 5000, 10000, 20000, 30000, 40000, 50000, 100000,200000,300000,500000,1000000])
    parser.add_argument("--full_iter", default=300000)
    parser.add_argument("--white_background", required=False, default=True)
    # parser.add_argument("--base_path", default="/hdd1/DB/MDI/250814/KSM_B_TRACK")
    parser.add_argument("--base_path", default="/hdd1/DB/nersemble_export/nersemble_export/UNION_226") 
    parser.add_argument("--lgt_type", default=None) # cube or hdr
    parser.add_argument("--output_base_path", default="./output/track_shs")
    parser.add_argument("--version", default="final_gt_normal")
    parser.add_argument("--resume_checkpoint", default=None, help="Path to a training checkpoint (.pt) to resume from")
    args, extras = parser.parse_known_args()
    
    main(args)


