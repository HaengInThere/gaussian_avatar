import math
import torch
import torch.nn.functional as F
from libs.utils.sh_utils import eval_sh_coef, eval_sh
from libs.utils.graphics_utils import fibonacci_sphere_sampling
import numpy as np
from libs.render.render_r3dg import rasterize_feature

def sample_incident_rays(normals, is_training=False, sample_num=24):
    if is_training:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=True)
    else:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=False)

    incident_dirs  = torch.nan_to_num(incident_dirs,  nan=0.0)
    incident_areas = torch.nan_to_num(incident_areas, nan=0.0)

    return incident_dirs, incident_areas  # [N, S, 3], [N, S, 1]


def rgb_to_srgb(img, clip=True):
    if isinstance(img, np.ndarray):
        assert len(img.shape) == 3 and img.shape[2] == 3, img.shape
        out = np.where(
            img > 0.0031308,
            np.power(np.clip(img, 0.0031308, None), 1.0 / 2.4) * 1.055 - 0.055,
            12.92 * img,
        )
        if clip:
            out = np.clip(out, 0.0, 1.0)
        return out
    elif isinstance(img, torch.Tensor):
        assert len(img.shape) == 3 and img.shape[0] == 3, img.shape  # CHW
        thr = 0.0031308
        out = torch.where(
            img > thr,
            torch.pow(torch.clamp(img, min=thr), 1.0 / 2.4) * 1.055 - 0.055,
            12.92 * img,
        )
        if clip:
            out = out.clamp(0.0, 1.0)
        return out
    else:
        raise TypeError("Unsupported input type. Supported types are numpy.ndarray and torch.Tensor.")


def _dot(a, b):
    return (a * b).sum(dim=-1, keepdim=True)  # [H, W, 1, 1]

def _f_diffuse(base_color):
    return base_color / np.pi  # [H, W, 1, 3]

# --- GGX Specular with Schlick Fresnel ---
# Shapes:
#   h_d_n, h_d_o, n_d_i, n_d_o: [N,S,1]
#   base_color: [N,1,3]
#   roughness, metallic: [N,1,1]

def _fresnel_schlick(hdotv, F0):
    # hdotv: [N,S,1], F0: [N,1,3] → [N,S,3]
    one_minus = (1.0 - hdotv.clamp(0.0, 1.0))
    return F0 + (1.0 - F0) * (one_minus ** 5)

def _ndf_ggx(alpha, cos_hn):
    # alpha: [N,1,1], cos_hn: [N,S,1] → [N,S,1]
    a2 = (alpha * alpha).clamp(min=1e-8)
    d = (cos_hn * cos_hn * (a2 - 1.0) + 1.0)
    return (a2 / (np.pi * (d * d + 1e-8))).clamp(min=0.0)

def _v_smith_ggx(alpha, cos):
    # Schlick-GGX "V" term that approximates G/(4*NdotL*NdotV)
    k = ((alpha + 1.0) ** 2) / 8.0
    return 0.5 / (cos * (1.0 - k) + k).clamp(min=1e-6)

def _f_specular(h_d_n, h_d_o, n_d_i, n_d_o, base_color, roughness, metallic):
    # Perceptual roughness → microfacet alpha
    roughness = roughness.clamp(0.04, 1.0)
    alpha = (roughness * roughness).clamp(1e-4, 1.0)

    # Dielectric F0 ≈ 0.04 (RGB). Metallic tints with base_color
    F0 = 0.04 * (1.0 - metallic) + base_color * metallic  # [N,1,3]
    # F0 = 0.04 * (1.0 - 0) + base_color * 0  # [N,1,3]

    D = _ndf_ggx(alpha, h_d_n)                             # [N,S,1]
    V = _v_smith_ggx(alpha, n_d_i) * _v_smith_ggx(alpha, n_d_o)  # [N,S,1]
    F = _fresnel_schlick(h_d_o, F0)                        # [N,S,3]

    return (D * V) * F                                     # [N,S,3]
    
def rendering_equation_python(base_color, roughness, metallic, normals, viewdirs,
                              f_shading_d=None, f_shading_s=None,  # f_visibility는 여기서 쓰지 않음
                              is_training=True, direct_light_env_light=None,
                              sample_num=64, shs_visibility=None):

    # R = torch.tensor([
    #     [0., 1., 0.],
    #     [0., 0., -1.],
    #     [-1., 0., 0.]
    # ], device=normals.device, dtype=normals.dtype)

    # # 1-2) 모든 관련 벡터들을 빛의 좌표계로 변환
    # # (N, 1, 3) 형태의 텐서에 (3, 3) 행렬을 곱하기 위해 transpose 후 다시 transpose 해줍니다.
    # normals_transformed = (R @ normals.transpose(-1, -2)).transpose(-1, -2)
    # viewdirs_transformed = (R @ viewdirs.transpose(-1, -2)).transpose(-1, -2)

    # 1-3) 샘플 방향, 샘플 가중치
    # 변환된 노멀을 기준으로 빛 방향을 샘플링합니다.
    incident_dirs, incident_areas = sample_incident_rays(normals, False, sample_num)

    # 2) 브로드캐스트 정리
    base_color = base_color.unsqueeze(-2)   # [N,1,3]
    roughness  = roughness.unsqueeze(-2).clamp(0.04, 1.0)    # [N,1,1]
    metallic   = metallic.unsqueeze(-2).clamp(0.0, 1.0)      # [N,1,1]
    normals    = torch.nan_to_num(normals.unsqueeze(-2), nan=0.0)  # [N,1,3]
    viewdirs   = viewdirs.unsqueeze(-2)     # [N,1,3]


    # 3) 환경광을 샘플 방향에서 평가  ← 여기서 방향성이 들어옵니다
    if direct_light_env_light is None:
        raise ValueError("direct_light_env_light must be provided.")
    incident_lights = direct_light_env_light.direct_light(incident_dirs)  # [N,S,3]

    # 4) 샘플별 하프벡터와 코사인들
    half_dirs = F.normalize(incident_dirs + viewdirs, dim=-1)  # [N,S,3]
    h_d_n = (half_dirs * normals).sum(dim=-1, keepdim=True).clamp(min=0.0)  # [N,S,1]
    h_d_o = (half_dirs * viewdirs).sum(dim=-1, keepdim=True).clamp(min=0.0) # [N,S,1]
    n_d_i = (normals * incident_dirs).sum(dim=-1, keepdim=True).clamp(min=0.0)  # [N,S,1]
    n_d_o = (normals * viewdirs).sum(dim=-1, keepdim=True).clamp(min=0.0)       # [N,1,1]

    # 5) BRDF
    f_d = base_color / np.pi                                        # [N,1,3]
    f_s = _f_specular(h_d_n, h_d_o, n_d_i, n_d_o, base_color, roughness, metallic)  # [N,S,3]


    # # 6) 광원 가시도/그림자. light-브랜치 출력은 [0,1]이어야 합니다
    # V_light_d = f_shading_d.reshape(base_color.shape[0], 1, 1)  # [N,1,1]
    # V_light_s = f_shading_s.reshape(base_color.shape[0], 1, 1)  # [N,1,1]

    # 7) 올바른 transport: Li · V · max(0,n·ω) · inv_pdf(or Δω)
    transport_d = incident_lights * n_d_i * incident_areas  # [N,S,3]
    # transport_d = V_light_d * incident_areas  # [N,S,3]
        # 7) 올바른 transport: Li · V · max(0,n·ω) · inv_pdf(or Δω)
    transport_s = incident_lights * n_d_i * incident_areas # V_light_s  # [N,S,3]

    # 8) 샘플 평균
    rgb_d = (f_d * transport_d).mean(dim=1, keepdim=True)  # [N,1,3]
    rgb_s = (f_s * transport_s).mean(dim=1, keepdim=True)                   # [N,1,3]
    rgb   = rgb_d + rgb_s

    # 진단용 셰이딩 스칼라 (transport 자체의 평균)
    shading = transport_d.mean(dim=1)  # [N,1,3] → later reduced as needed

    vis = None
    return rgb.squeeze(), rgb_d.squeeze(), rgb_s.squeeze(), vis, shading.squeeze()



def rendering_equation_python_comp(base_color, roughness, metallic, normals, viewdirs, lgt_vec, lgt_sh, lgt_in, is_training=True, direct_light_env_light=None, sample_num=64, shs_visibility = None):

    base_color = base_color.squeeze().contiguous() # N 1 3 
    roughness = roughness.squeeze().contiguous()
    metallic = metallic.squeeze().contiguous()
    normals = normals.squeeze().contiguous()
    viewdirs = viewdirs.squeeze().contiguous()

    normals = torch.nan_to_num(normals, nan=0.0)
    
    lgt_vec = lgt_vec.squeeze()

    half_dirs = lgt_vec + viewdirs # N 3
    half_dirs = F.normalize(half_dirs, dim=-1) # N 64 3 

    h_d_n = _dot(half_dirs, normals).clamp(min=0)
    h_d_o = _dot(half_dirs, viewdirs).clamp(min=0)
    n_d_i = _dot(normals, lgt_vec).clamp(min=0)
    n_d_o = _dot(normals, viewdirs).clamp(min=0)
    
    f_d = _f_diffuse(base_color) # N 1 3
    f_s = _f_specular(h_d_n, h_d_o, n_d_i, n_d_o, base_color, roughness.unsqueeze(-1), metallic.unsqueeze(-1)) # N 64 3

    falloff_shading = lgt_in * torch.exp(lgt_sh*(n_d_i-1.0))
    transport = n_d_i * falloff_shading  # N 1

    rgb_d = (f_d * transport_d) # 예측한거 
    shading = transport_d
    rgb_s = (f_s * transport)
    rgb = rgb_d + rgb_s

    return rgb, rgb_d.squeeze(), rgb_s, None, shading.squeeze()


def render_stage(viewpoint_cam, gs_output, env, bg_color, scaling_modifier=1.0):
    means3D = gs_output["gaussian"].get_xyz # N 3 
    opacity = gs_output["gaussian"].get_opacity
    scales = gs_output["gaussian"].get_scaling
    shs = gs_output["gaussian"].get_features
    rotations = gs_output["gaussian"].get_rotation
    active_sh_degree = gs_output["gaussian"].get_activate_sh_degree
    
    # metallic is not used in 3D pbr
    f_base_color,f_roughness,f_metallic,normal = gs_output["gaussian"].get_albedo, \
                                        gs_output["gaussian"].get_roughness, \
                                        gs_output["gaussian"].get_metallic, \
                                        gs_output["gaussian"].get_normal, \
                                        
    # f_shading_d = gs_output['sampled']['shading_d']
    # f_shading_s = gs_output['sampled']['shading_s']
    # f_visibility = gs_output['sampled']['visibility']
    f_base_color = gs_output["gaussian"].get_features.squeeze()


    viewdirs = F.normalize(viewpoint_cam.camera_center - means3D, dim=-1) # N 3 
    # rotated_3d = torch.matmul(means3D[:,None], data['world_view_transform'][:3, :3]).squeeze() + data['world_view_transform'][:3, 3][None]
    # cam_center_B = data['camera_center_B']
    # viewdirs_B = F.normalize(cam_center_B - rotated_3d, dim=-1)
    # viewdirs = F.normalize(cam_center - means3D, dim=-1)

    color_pbr, rgb_d, rgb_s, vis, shading = rendering_equation_python(f_base_color, f_roughness, f_metallic, normal, viewdirs, is_training=False, direct_light_env_light=env, shs_visibility=None)
    
    # Concatenate features (currently 23 channels), then pad to 24 for alignment
    pad1 = torch.ones_like(f_roughness)  # N x 1 dummy channel for alignment
    # f_shading = f_shading.view(-1, 1, 1)
    # f_visibility = f_visibility.view(-1,1, 1)
    gs_feat = torch.cat([
        color_pbr, f_roughness, f_metallic, f_base_color, normal, viewdirs, rgb_d, rgb_s, pad1
    ], -1)  # total 24 channels

    outs = rasterize_feature(means3D, shs=shs, color=None, features=gs_feat, opacity=opacity, scales=scales, rotations=rotations, viewpoint_cam=viewpoint_cam,
                                 scaling_modifier=scaling_modifier, bg_color=bg_color, active_sh_degree=active_sh_degree)
    rendered_feature = outs['feature']
    render_pbr, roughness, metallic, base_color, r_normal, r_view, r_rgb_d, r_rgb_s, _pad = rendered_feature.split([3, 1, 1, 3, 3, 3, 3, 3, 1], 0)
   
    r_dot = r_normal[0] * r_view[0] + r_normal[1] * r_view[1] + r_normal[2] * r_view[2]
    
    tmp_mask = torch.zeros_like(r_dot.clamp(0, 1))
    tmp_mask[-r_dot > 0.2] = 1
    tmp_mask = tmp_mask * outs['alpha'].squeeze()
    outs.update({'visibility': vis})
    outs.update({'pbr': render_pbr})
    # outs.update({'depth': depth})
    outs.update({'roughness': roughness})
    outs.update({'metallic': metallic})
    outs.update({'base_color': base_color})
    outs.update({'normal': r_normal})
    outs.update({'mask_inter':tmp_mask})
    outs.update({'viewdirs':r_view})
    outs.update({'diffuse':r_rgb_d})
    outs.update({'specular':r_rgb_s})
    # outs.update({'shading_d':shading_d})
    # outs.update({'shading_s':shading_s})
    return outs
