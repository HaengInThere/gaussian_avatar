from    __future__ import annotations

import  torch
from    torch import nn
import  torch.nn.functional as F
# import torchvision.transforms.functional as TF
import math
import torchvision.transforms.functional as FF
import  warnings
import  lpips
from    torch.autograd import Variable
from    math import exp
from    libs.utils.vgg_feature import VGGPerceptualLoss

from    pytorch3d.structures import     Meshes
from    pytorch3d.loss.mesh_laplacian_smoothing import   mesh_laplacian_smoothing
from    pytorch3d.loss.mesh_normal_consistency  import   mesh_normal_consistency

from    typing import Type, Union
from    dataclasses import dataclass, field

warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13")

# ------------------------------------------------------------------------------- #


from matplotlib import cm

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def error_map(img1, img2):
    error = (img1 - img2).mean(dim=0) / 2 + 0.5
    cmap = cm.get_cmap("seismic")
    error_map = cmap(error.cpu())
    return torch.from_numpy(error_map[..., :3]).permute(2, 0, 1)


# ------------------------------------------------------------------------------- #

# class BaseLoss(nn.Module):

#     @dataclass
#     class Params:
#         loss_weight:        float

#     def accumulate_gradients(self, model_output, ground_truth): # to be overridden by subclass
#         raise NotImplementedError
    
#     def forward(self, model_output, ground_truth):
#         return self.accumulate_gradients(model_output, ground_truth)
    
# ------------------------------------------------------------------------------- #

class UVDecoderLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg_loss = cfg["loss"]
        self.rgb_type=self.cfg_loss["rgb_type"]
        self.scale_threshold=self.cfg_loss["scale_threshold"]
        self.weight=self.cfg_loss["weight"]
        
        self.w_rgb_loss = self.weight["rgb_loss"]
        self.w_vgg_loss = self.weight["vgg_loss"]
        self.w_dssim_loss = self.weight["dssim_loss"]
        self.w_scale_loss = self.weight["scale_loss"]
        self.w_lpips_loss = self.weight["lpips_loss"]
        self.w_rot_loss = self.weight["rot_loss"]
        self.w_laplacian_loss = self.weight["laplacian_loss"]
        self.w_normal_loss = self.weight["normal_loss"]
        self.w_flame_loss = self.weight["flame_loss"]
        
        self.vgg_loss       = VGGPerceptualLoss()
        self.lpips_loss     = lpips.LPIPS(net='vgg').eval()
        self.l1_loss        = nn.L1Loss(reduction='mean')
        self.l2_loss        = nn.MSELoss(reduction='mean')

        self.laplacian_matrix   = None

    def get_dssim_loss(self, rgb_values, rgb_gt):
        return d_ssim(rgb_values, rgb_gt)

    def get_vgg_loss(self, rgb_values, rgb_gt):
        return self.vgg_loss(rgb_values, rgb_gt)

    def get_rgb_loss(self, rgb_values, rgb_gt):
        rgb_loss = self.l1_loss(rgb_values, rgb_gt)
        return rgb_loss
    
    def get_lpips_loss(self, rgb_values, rgb_gt, normalize=True):
        return self.lpips_loss(rgb_values, rgb_gt, normalize=normalize)
    
    def get_laplacian_smoothing_loss(self, verts_orig, verts):
        L = self.laplacian_matrix[None, ...].detach()

        basis_lap   = L.bmm(verts_orig).detach()
        offset_lap  = L.bmm(verts)

        diff = (offset_lap - basis_lap) ** 2
        diff = diff.sum(dim=-1, keepdim=True)

        return diff.mean()

    def forward(self, model_outputs, ground_truth, mask_image, gt_noraml, background):

        render_image = model_outputs['rgb_image'].unsqueeze(0)   # torch.Size([1, 3, 512, 512])
        normal_image = model_outputs["normal_map"].unsqueeze(0)
        gt_image     = ground_truth.unsqueeze(0)        # torch.Size([1, 3, 512, 512])
        alpha_mask   = mask_image.unsqueeze(0)
        gt_noraml = gt_noraml.unsqueeze(0)
        
        gt_image = (gt_image * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)
        gt_normal = (gt_noraml * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)
        

        # initialize the loss
        rgb_loss = self.get_rgb_loss(render_image, gt_image)
        loss = rgb_loss.clone()
        out = {'loss': loss, 'rgb_loss': rgb_loss}
        
        normal_loss = self.get_rgb_loss(normal_image, gt_normal) 
        out["normal_loss"] = normal_loss
        out["loss"] += normal_loss * 1.0

        dssim_loss = self.get_dssim_loss(render_image, gt_image)
        out['dssim_loss'] = dssim_loss
        out['loss'] += dssim_loss * 0.2

        normal_dssim_loss = self.get_dssim_loss(normal_image, gt_normal)
        out['normal_ssim'] = normal_dssim_loss
        out['loss'] += normal_dssim_loss * 0.2

        raw_rot  = model_outputs['raw_rot']
        rot_loss = torch.mean(raw_rot[..., 0] ** 2) + torch.mean(raw_rot[..., 2] ** 2)
        out['rot_loss'] = rot_loss
        out['loss'] += rot_loss * 0.0
            
        return out
    


class PBRAvatarLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg_loss = cfg["loss"]
        self.rgb_type=self.cfg_loss["rgb_type"]
        self.scale_threshold=self.cfg_loss["scale_threshold"]
        self.weight=self.cfg_loss["weight"]
        
        self.w_rgb_loss = self.weight["rgb_loss"]
        self.w_vgg_loss = self.weight["vgg_loss"]
        self.w_dssim_loss = self.weight["dssim_loss"]
        self.w_scale_loss = self.weight["scale_loss"]
        self.w_lpips_loss = self.weight["lpips_loss"]
        self.w_rot_loss = self.weight["rot_loss"]
        self.w_laplacian_loss = self.weight["laplacian_loss"]
        self.w_normal_loss = self.weight["normal_loss"]
        self.w_flame_loss = self.weight["flame_loss"]

        self.lpips_loss             = lpips.LPIPS(net='vgg').eval()
        self.l1_loss                = nn.L1Loss(reduction='mean')
        self.l2_loss                = nn.MSELoss(reduction='mean')

        self.laplacian_matrix       = None

    def get_dssim_loss(self, rgb_values, rgb_gt):
        return d_ssim(rgb_values, rgb_gt)

    def get_rgb_loss(self, rgb_values, rgb_gt):
        if self.rgb_type == 'l1':
            return self.l1_loss(rgb_values, rgb_gt)
        elif self.rgb_type== 'l2':
            return self.l2_loss(rgb_values, rgb_gt)
    
    def get_lpips_loss(self, rgb_values, rgb_gt, normalize=True):
        return self.lpips_loss(rgb_values, rgb_gt, normalize=normalize)

    def get_laplacian_smoothing_loss(self, verts_orig, verts):
        L = self.laplacian_matrix[None, ...].detach()

        basis_lap   = L.bmm(verts_orig).detach()
        offset_lap  = L.bmm(verts)

        diff = (offset_lap - basis_lap) ** 2
        diff = diff.sum(dim=-1, keepdim=True)

        return diff.mean()
    
    def lightmap_reg_loss(self, light_map, *, latlong=True,
                        w_tv=0.05, w_log_energy=1e-3, w_color=1e-2, w_log_var=1e-3, w_nonneg=1e-2):
        """
        light_map: (B, 3, H, W), linear HDR radiance [0, +inf)
        반환: {'loss': ..., 'tv': ..., 'log_energy': ..., 'color_balance': ..., 'log_var': ..., 'nonneg': ...}
        """
        eps = 1e-6
        L = light_map

        losses = {}

        # 1) 비음수 유도. 파라미터화에서 보장하지 않는 경우에만 의미가 있다.
        neg = torch.relu(-L)
        losses['nonneg'] = neg.mean()

        # 2) 총 에너지 제어. 로그 평균을 0 근처로 유도해 과노출을 억제한다.
        logL = torch.log(L.clamp_min(eps))
        losses['log_energy'] = logL.mean().abs()

        # 3) 구면 TV. lat–long이면 위도 가중치를 준다.
        if latlong:
            B, C, H, W = L.shape
            theta = torch.linspace(0.5/H, 1.0 - 0.5/H, H, device=L.device) * math.pi  # [0, pi]
            w_lat = torch.sin(theta).view(1, 1, H, 1)  # (1,1,H,1)
        else:
            w_lat = 1.0

        dx = L[..., :, 1:] - L[..., :, :-1]         # (B,3,H,W-1)
        dy = L[..., 1:, :] - L[..., :-1, :]         # (B,3,H-1,W)
        tv_h = dx.abs().mean()
        tv_v = (w_lat[..., 1:, :] * dy.abs()).mean()
        losses['tv'] = tv_h + tv_v

        # 4) 색 균형. 채널별 평균 크로마가 회색에 가깝도록 유도한다.
        mean_c = L.mean(dim=(-2, -1), keepdim=True)          # (B,3,1,1)
        gray   = mean_c.mean(dim=1, keepdim=True)            # (B,1,1,1)
        color_balance = (mean_c / (gray + eps) - 1.0).abs().mean()
        losses['color_balance'] = color_balance

        # 5) 하이라이트 완화. 로그 공간에서 분산을 억제한다.
        losses['log_var'] = logL.var(dim=(-2, -1), unbiased=False).mean()

        total = (
            w_tv * losses['tv'] +
            w_log_energy * losses['log_energy'] +
            w_color * losses['color_balance'] +
            w_log_var * losses['log_var'] +
            w_nonneg * losses['nonneg']
        )
        
        return total
    

    def get_normal_loss(self, pred_normal, gt_normal_01, mask=None, mode="unsigned", eps: float = 1e-8):
        """
        pred_normal: (B,3,H,W) in [-1,1]
        gt_normal_01: (B,3,H,W) in [0,1]
        mask: (B,H,W) or None
        """
        dim = 1  # channel dim for (B,3,H,W)

        gt = gt_normal_01
        pred_u = pred_normal
        gt_u   = gt

        dot = (pred_u * gt_u).sum(dim=dim)  # (B,H,W)

        if mode == "cos":
            loss_map = 1.0 - dot
        elif mode == "unsigned":
            loss_map = 1.0 - dot.abs()
        elif mode == "acos":
            loss_map = torch.arccos(torch.clamp(dot, -1.0 + 1e-4, 1.0 - 1e-4))
        else:  # fallback
            loss_map = (pred_u - gt_u).abs().mean(dim=dim)

        if mask is not None:
            loss = (loss_map * mask).sum() / (mask.sum() + eps)
        else:
            loss = loss_map.mean()
        return loss

    def light_mix_consistency_loss(
        self,
        light_module,
        w_mix: float = 1.0,
        w_white: float = 1e-2,
        w_nonneg: float = 1e-4,
        # 새로 추가: 스케일 하한/초기값 규제
        w_scale_low: float = 0.5,
        w_scale_ref: float = 0.1,
        # (선택) full 라이트 에너지 앵커
        w_energy: float = 0.0,
    ):
        """
        light_module: CubemapLight (light_maps: (K,6,H,W,3), global_scale: (K,3), global_scale_init: (K,3))
        """
        L = light_module.light_maps      # (K,6,H,W,3)
        S = light_module.global_scale    # (K,3)  -> 이미 softplus + s_min 로 양수/하한보장
        K = L.shape[0]
        scaled = L * S.view(K, 1, 1, 1, 3)

        # 1) full = 마지막 인덱스, pred = 나머지 합
        full = scaled[-1]                 # (6,H,W,3)
        pred = scaled[:-1].sum(dim=0)     # (6,H,W,3)
        mix_mse = F.mse_loss(pred, full)

        # 2) (약한) 화이트 prior: per-light 평균 크로마가 회색에 가깝도록
        m = scaled[:-1].mean(dim=(1,2,3))            # (K-1,3)
        gray = m.mean(dim=1, keepdim=True)           # (K-1,1)
        white = (m / (gray + 1e-6) - 1.0).abs().mean()

        # 3) (이전 안전장치) 음수 방지: softplus라면 거의 0이지만 남겨둠
        neg = torch.relu(-scaled[:-1]).mean()

        # 4) ★ 스케일 하한에서 멀어지게(바닥에 붙지 않게)
        s_min = float(light_module.global_scale_min.item())
        # s_min보다 살짝 더 큰 마진까지 붙는 것을 억제 (여유를 0.5*s_min로 줄 수도)
        s_floor = s_min + 0.5 * s_min
        scale_low = F.relu(s_floor - S).mean()   # S가 클수록 0, 작으면 큰 페널티

        # 5) ★ 초기 스케일 근처 유지 (albedo로 보상하기 어렵게)
        if hasattr(light_module, "global_scale_init"):
            S0 = light_module.global_scale_init  # (K,3)
            scale_ref = ((S - S0) ** 2).mean()
        else:
            scale_ref = S.new_tensor(0.0)

        # 6) (선택) full 라이트 에너지 앵커: 초기 full의 평균 에너지로 수렴 유도
        if w_energy > 0.0:
            if not hasattr(light_module, "full_energy_ref"):
                # 초기 full 에너지 저장(채널별)
                with torch.no_grad():
                    E_ref = (L[-1] * S[-1].view(1,1,1,3)).mean(dim=(0,1,2))
                light_module.register_buffer("full_energy_ref", E_ref)
            E_full = (full).mean(dim=(0,1,2))  # (3,)
            energy = F.mse_loss(E_full, light_module.full_energy_ref)
        else:
            energy = S.new_tensor(0.0)

        total = (
            w_mix * mix_mse +
            w_white * white +
            w_nonneg * neg +
            w_scale_low * scale_low +
            w_scale_ref * scale_ref +
            w_energy * energy
        )

        stats = {
            'light_mix_mse': mix_mse,
            'light_white': white,
            'light_neg': neg,
            'scale_low': scale_low,
            'scale_ref': scale_ref,
        }
        if w_energy > 0.0:
            stats['light_energy'] = energy

        return total, stats

    def get_dynamic_reg(self, d3_output):
            # --- softly penalize dynamic geometry magnitudes from the model ---
        loss = 0
        if isinstance(d3_output, dict) and ('reg_terms' in d3_output):
            regs = d3_output['reg_terms']
            w_dpos    = 0.1
            w_slog    = 0.001
            w_dnormal = 0.05
            w_rot     = 0.01
            w_canon_xyz = 0.0001
            w_canon_scale_log = 0.01
            w_delta   = 10.0

            w_opa_reg = 0.0001

            # if 'dpos' in regs:         loss = loss + w_dpos      * regs['dpos']
            # if 'scale_log' in regs:    loss = loss + w_slog      * regs['scale_log']
            # if 'dnormal' in regs:      loss = loss + w_dnormal   * regs['dnormal']
            # if 'rot' in regs:          loss = loss + w_rot       * regs['rot']
            # if 'opa_reg' in regs:      loss = loss + w_opa_reg   * regs['opa_reg']

            # if 'canon_xyz' in regs:    loss = loss + w_canon_xyz * regs['canon_xyz']
            # if 'canon_scale_log' in regs: loss = loss + w_canon_scale_log * regs['canon_scale_log']
            # if 'delta_vertex' in regs: loss = loss + w_delta     * regs['delta_vertex']
        
        return loss
    

    def forward(self, render_output, d3_output, viewpoint_cam, background, light_map=None, train_normal=True):
        ground_truth = viewpoint_cam.original_image
        mask_image = viewpoint_cam.original_mask
        gt_image     = ground_truth.unsqueeze(0)        # torch.Size([1, 3, 512, 512])
        alpha_mask   = mask_image.unsqueeze(0)
        gt_image = (gt_image * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)

        render_image = render_output['render'].unsqueeze(0)   # torch.Size([1, 3, 512, 512])
        normal_image = render_output["normal"].unsqueeze(0)
        # pnormal_image = render_output["rendered_pseudo_normal"].unsqueeze(0)

        # PBRAvatarLoss.forward 내부 예시
        def _down(x, size=512):
            H, W = x.shape[-2:]
            if min(H, W) <= size:
                return x
            return F.interpolate(x, size=size, mode='area')

        I_pred_small = _down(render_image)
        I_gt_small   = _down(gt_image)

        # Initialize the loss
        loss = self.get_rgb_loss(render_image, gt_image) * 0.8
        out = {'loss': loss, 'rgb_loss': loss}
        
        dssim_loss = self.get_dssim_loss(render_image, gt_image)
        out['ssim'] = dssim_loss
        out['loss'] += dssim_loss * 0.2

        lpips_loss = self.get_lpips_loss(render_image, gt_image)
        out['lpips'] = dssim_loss
        out['loss'] += lpips_loss.squeeze() * 0.2

        if viewpoint_cam.original_normal is not None and train_normal:
            gt_normal = viewpoint_cam.original_normal * 2 - 1
            gt_normal = gt_normal.unsqueeze(0)
            gt_normal = (gt_normal * alpha_mask + background[:, None, None] * (1.0 - alpha_mask))

            # Optional: gate supervision to visible regions using renderer alpha if available
            pred_vis = None
            if isinstance(render_output, dict):
                for k in ("alpha", "acc", "accum_alpha", "opacity"):
                    if k in render_output:
                        pred_vis = render_output[k]
                        break
            if pred_vis is not None:
                if pred_vis.dim() == 2:
                    pred_vis = pred_vis.unsqueeze(0)
                # threshold to a soft visibility mask
                vis_mask = (pred_vis > 1e-3).float()
                alpha_mask = alpha_mask * vis_mask

            # n_pred_small = _down(normal_image) * 0.5 + 0.5
            # n_gt_small   = _down(gt_normal)

            normal_loss = self.get_normal_loss(
                normal_image,      # (B,3,H,W), [-1,1]
                gt_normal,         # (B,3,H,W), [0,1]
                mask=alpha_mask,   # (B,H,W)x
                mode="cos"   # sign-invariant to avoid back-face flips
            )
            out["normal_loss"] = normal_loss
            out["loss"] += normal_loss * 0.2

            # lpips_normal_loss = self.get_lpips_loss(n_pred_small, n_gt_small) 
            # out['lpips_normal'] = lpips_normal_loss
            # out['loss'] += lpips_normal_loss.squeeze() * 0.2

            # --- dynamic geometry magnitude regularizers (keep small at start) ---
            w_reg_dnormal    = self.weight.get("dnormal_l2", 0.1)
            if isinstance(d3_output, dict) and ('reg_terms' in d3_output):
                regs = d3_output['reg_terms']

                if 'dnormal' in regs:
                    out['dnormal_l2'] = regs['dnormal_l2']
                    out['loss'] += w_reg_dnormal * regs['dnormal']

        # scale loss 1 : min max
        scale = d3_output['scale']
        scale_max, _ = torch.max(scale, dim=-1)
        scale_min, _ = torch.min(scale, dim=-1)
        scale_regu = F.relu(scale_max / scale_min - 5.0).mean()
        out["scale_loss"] = scale_regu
        out["loss"] += scale_regu * 0.01

        r = d3_output["reg_terms"]
        loss_reg = 0.0
        for key, w in r.items():
            if key in r and r[key] is not None:
                loss_reg = loss_reg + r[key]

        out["loss_reg"] = loss_reg
        out["loss"] += out["loss_reg"]

        return out


    def forward_hdr(self, render_pbr_output, d3_output, render_output, viewpoint_cam, background, light_map=None, train_normal=False):
        ground_truth = viewpoint_cam.original_image
        mask_image = viewpoint_cam.original_mask

        gt_image     = ground_truth.unsqueeze(0)        # torch.Size([1, 3, 512, 512])
        alpha_mask   = mask_image.unsqueeze(0)
        gt_image = (gt_image * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)

        gt_albedo = viewpoint_cam.original_albedo
        gt_roughness = viewpoint_cam.original_roughness
        gt_metallic = viewpoint_cam.original_metallic

        render_image_pbr = render_pbr_output['pbr'].unsqueeze(0)   # torch.Size([1, 3, 512, 512])
        normal_image = render_pbr_output["normal"].unsqueeze(0)

        def _down(x, size=512):
            H, W = x.shape[-2:]
            if max(H, W) <= size:
                return x
            return F.interpolate(x, size=size, mode='area')
        
        I_pred_small = _down(render_image_pbr)
        I_gt_small   = _down(gt_image)
        
        # Initialize the loss
        loss = self.get_rgb_loss(render_pbr_output['pbr'], gt_image) * 0.8
        out = {'loss': loss, 'rgb_loss': loss}
        
        dssim_loss = self.get_dssim_loss(I_pred_small, I_gt_small)
        out['ssim'] = dssim_loss
        out['loss'] += dssim_loss * 0.2


        base_gt = (render_output['render'] * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)

        albedo_rgb_loss = self.get_rgb_loss(render_pbr_output['base_color'], base_gt.squeeze()) * 0.1
        out['albedo_rgb_loss'] =  albedo_rgb_loss
        out['loss'] += albedo_rgb_loss

        albedo_dssim_loss = self.get_dssim_loss(render_pbr_output['base_color'], base_gt.squeeze())
        out['albedo_ssim'] = albedo_dssim_loss
        out['loss'] += albedo_dssim_loss * 0.1
        
        lpips_loss = self.get_lpips_loss(I_pred_small, I_gt_small)
        out['lpips'] = dssim_loss
        out['loss'] += lpips_loss.squeeze() * 0.2

        if viewpoint_cam.original_normal is not None and train_normal:
            gt_normal = viewpoint_cam.original_normal * 2 - 1
            gt_normal = gt_normal.unsqueeze(0)
            gt_normal = (gt_normal * alpha_mask + background[:, None, None] * (1.0 - alpha_mask))

            n_pred_small = _down(normal_image)
            n_gt_small   = _down(gt_normal)

            normal_loss = self.get_normal_loss(
                normal_image,      # (B,3,H,W), [-1,1]
                gt_normal,         # (B,3,H,W), [0,1]
                mask=alpha_mask,   # (B,H,W)x
                mode="cos"   # sign-invariant to avoid back-face flips
            )
            out["normal_loss"] = normal_loss
            out["loss"] += normal_loss * 1.0

            lpips_normal_loss = self.get_lpips_loss(n_pred_small, n_gt_small)
            out['lpips_normal'] = lpips_normal_loss
            out['loss'] += lpips_normal_loss.squeeze() * 0.2

        # --- dynamic geometry magnitude regularizers (keep small at start) ---
        # w_reg_dnormal    = self.weight.get("reg_dnormal", 0.000)
        # if isinstance(d3_output, dict) and ('reg_terms' in d3_output):
        #     regs = d3_output['reg_terms']

        #     out['dnormal_spec'] = regs['dnormal']
        #     out['loss'] += w_reg_dnormal * out['dnormal_spec']

        # # Light loss
        # if light_map is not None:
        #     if hasattr(light_map, 'light_maps') and hasattr(light_map, 'global_scale'):
        #         l_total, l_stats = self.light_mix_consistency_loss(light_map)
        #         out.update(l_stats)
        #         out['loss'] += l_total
        #     elif hasattr(light_map, 'env_base'):
        #         light_reg_loss = self.lightmap_reg_loss(light_map.env_base)
        #         out['light_reg'] = light_reg_loss
        #         out['loss'] += light_reg_loss

                
        # albedo_reg = ((d3_output['gaussian']._albedo - 0.1) ** 2).mean()
        # out['albedo_reg'] = albedo_reg
        # out['loss'] += out['albedo_reg'] * 0.01
        r = d3_output["reg_terms"]
        loss_reg = 0.0
        for key, w in r.items():
            if key in r and r[key] is not None:
                loss_reg = loss_reg + r[key]

        out["loss_reg"] = loss_reg
        out["loss"] += out["loss_reg"]

        return out

def forward_full(self, render_output, render_pbr_output, d3_output, viewpoint_cam, background, light_map=None, train_normal=False):
        ground_truth = viewpoint_cam.original_image
        mask_image = viewpoint_cam.original_mask

        gt_image     = ground_truth.unsqueeze(0)        # torch.Size([1, 3, 512, 512])
        alpha_mask   = mask_image.unsqueeze(0)
        gt_image = (gt_image * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)

        # gt_albedo = viewpoint_cam.original_albedo
        # gt_roughness = viewpoint_cam.original_roughness
        # gt_metallic = viewpoint_cam.original_metallic

        render_image_pbr = render_pbr_output['pbr'].unsqueeze(0)   # torch.Size([1, 3, 512, 512])
        normal_image = render_pbr_output["normal"].unsqueeze(0)

        def _down(x, size=256):
            H, W = x.shape[-2:]
            if max(H, W) <= size:
                return x
            return F.interpolate(x, size=size, mode='area')
        
        I_pred_small = _down(render_image_pbr)
        I_gt_small   = _down(gt_image)
        
        # Initialize the loss
        loss = self.get_rgb_loss(render_image_pbr, gt_image) * 0.8
        out = {'loss': loss, 'rgb_loss': loss}
        
        dssim_loss = self.get_dssim_loss(I_pred_small, I_gt_small)
        out['ssim'] = dssim_loss
        out['loss'] += dssim_loss * 0.2

        lpips_loss = self.get_lpips_loss(I_pred_small, I_gt_small)
        out['lpips'] = dssim_loss
        out['loss'] += lpips_loss.squeeze() * 0.2

        if viewpoint_cam.original_normal is not None and train_normal:
            gt_normal = viewpoint_cam.original_normal * 2 - 1
            gt_normal = gt_normal.unsqueeze(0)
            gt_normal = (gt_normal * alpha_mask + background[:, None, None] * (1.0 - alpha_mask))

            n_pred_small = _down(normal_image)
            n_gt_small   = _down(gt_normal)

            normal_loss = self.get_normal_loss(
                normal_image,      # (B,3,H,W), [-1,1]
                gt_normal,         # (B,3,H,W), [0,1]
                mask=alpha_mask,   # (B,H,W)x
                mode="cos"   # sign-invariant to avoid back-face flips
            )
            out["normal_loss"] = normal_loss
            out["loss"] += normal_loss * 1.0

            lpips_normal_loss = self.get_lpips_loss(n_pred_small, n_gt_small)
            out['lpips_normal'] = lpips_normal_loss
            out['loss'] += lpips_normal_loss.squeeze() * 0.2

        # --- dynamic geometry magnitude regularizers (keep small at start) ---
        w_reg_dnormal    = self.weight.get("reg_dnormal", 0.01)
        if isinstance(d3_output, dict) and ('reg_terms' in d3_output):
            regs = d3_output['reg_terms']

            out['dnormal_spec'] = regs['dnormal_spec']
            out['dnormal_spec_mag'] = regs['dnormal_spec_mag']
            out['loss'] += w_reg_dnormal * regs['dnormal_spec'] \
                        + w_reg_dnormal * regs['dnormal_spec_mag']

        # # Light loss
        # if light_map is not None:
        #     if hasattr(light_map, 'light_maps') and hasattr(light_map, 'global_scale'):
        #         l_total, l_stats = self.light_mix_consistency_loss(light_map)
        #         out.update(l_stats)
        #         out['loss'] += l_total
        #     elif hasattr(light_map, 'env_base'):
        #         light_reg_loss = self.lightmap_reg_loss(light_map.env_base)
        #         out['light_reg'] = light_reg_loss
        #         out['loss'] += light_reg_loss

                
        # albedo_reg = ((d3_output['gaussian']._albedo - 0.1) ** 2).mean()
        # out['albedo_reg'] = albedo_reg
        # out['loss'] += out['albedo_reg'] * 0.01

        return out



def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

    
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def d_ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return (1 - ssim_map.mean())
    else:
        return (1 - ssim_map.mean(1).mean(1).mean(1))
    

def get_tv_loss(
    gt_image: torch.Tensor,  # [3, H, W]
    prediction: torch.Tensor,  # [C, H, W]
    pad: int = 1,
    step: int = 1,
) -> torch.Tensor:
    if pad > 1:
        gt_image = F.avg_pool2d(gt_image, pad, pad)
        prediction = F.avg_pool2d(prediction, pad, pad)
    rgb_grad_h = torch.exp(
        -(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    rgb_grad_w = torch.exp(
        -(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    tv_h = torch.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2)  # [C, H-1, W]
    tv_w = torch.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2)  # [C, H, W-1]
    tv_loss = (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    if step > 1:
        for s in range(2, step + 1):
            rgb_grad_h = torch.exp(
                -(gt_image[:, s:, :] - gt_image[:, :-s, :]).abs().mean(dim=0, keepdim=True)
            )  # [1, H-1, W]
            rgb_grad_w = torch.exp(
                -(gt_image[:, :, s:] - gt_image[:, :, :-s]).abs().mean(dim=0, keepdim=True)
            )  # [1, H-1, W]
            tv_h = torch.pow(prediction[:, s:, :] - prediction[:, :-s, :], 2)  # [C, H-1, W]
            tv_w = torch.pow(prediction[:, :, s:] - prediction[:, :, :-s], 2)  # [C, H, W-1]
            tv_loss += (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    return tv_loss

def simple_tv_loss(x):
    tv_h = torch.pow(x[:, 1:, :] - x[:, :-1, :], 2).mean()
    tv_w = torch.pow(x[:, :, 1:] - x[:, :, :-1], 2).mean()
    return tv_h + tv_w

def rgb_to_hsv(img):
    # Convert tensor image to PIL for transformation
    img_pil = FF.to_pil_image(img.cpu().squeeze(0))
    img_hsv = img_pil.convert("HSV")
    
    # Convert back to tensor
    return FF.to_tensor(img_hsv).to(img.device)