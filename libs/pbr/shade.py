import os, math
from typing import Dict, Optional, Union

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
import cv2 
from typing import Dict, List, Optional, Tuple, Union
from .light import CubemapLight, DirectLightMap
from torchvision.utils import save_image
from libs.render.render_step1 import render
# from libs.render.render_r3dg import rasterize_feature

def dir_to_uv(dirs: torch.Tensor) -> torch.Tensor:
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    theta = torch.atan2(x, z)  # [-π, π]
    phi = torch.asin(torch.clamp(y, -1.0, 1.0))  # [-π/2, π/2]
    
    u = 0.5 + theta / (2 * math.pi)  # [0, 1]
    v = 0.5 - phi / math.pi          # [0, 1]
    return torch.stack((u, v), dim=-1)  # [..., 2]

# Lazarov 2013, "Getting More Physical in Call of Duty: Black Ops II"
# https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
def envBRDF_approx(roughness: torch.Tensor, NoV: torch.Tensor) -> torch.Tensor:
    c0 = torch.tensor([-1.0, -0.0275, -0.572, 0.022], device=roughness.device)
    c1 = torch.tensor([1.0, 0.0425, 1.04, -0.04], device=roughness.device)
    c2 = torch.tensor([-1.04, 1.04], device=roughness.device)
    r = roughness * c0 + c1
    a004 = (
        torch.minimum(torch.pow(r[..., (0,)], 2), torch.exp2(-9.28 * NoV)) * r[..., (0,)]
        + r[..., (1,)]
    )
    AB = (a004 * c2 + r[..., 2:]).clamp(min=0.0, max=1.0)
    return AB


def saturate_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1, keepdim=True).clamp(min=1e-4, max=1.0)


# Tone Mapping
def aces_film(rgb: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    EPS = 1e-6
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    rgb = (rgb * (a * rgb + b)) / (rgb * (c * rgb + d) + e)
    if isinstance(rgb, np.ndarray):
        return rgb.clip(min=0.0, max=1.0)
    elif isinstance(rgb, torch.Tensor):
        return rgb.clamp(min=0.0, max=1.0)


def linear_to_srgb(linear: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(linear, torch.Tensor):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * torch.clamp(linear, min=eps) ** (5 / 12) - 11) / 200
        return torch.where(linear <= 0.0031308, srgb0, srgb1)
    elif isinstance(linear, np.ndarray):
        eps = np.finfo(np.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return np.where(linear <= 0.0031308, srgb0, srgb1)
    else:
        raise NotImplementedError


def _rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(
        f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0 / 2.4) * 1.055 - 0.055
    )


def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = (
        torch.cat((_rgb_to_srgb(f[..., 0:3]), f[..., 3:4]), dim=-1)
        if f.shape[-1] == 4
        else _rgb_to_srgb(f)
    )
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out


def _srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(
        f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4)
    )


def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = (
        torch.cat((_srgb_to_rgb(f[..., 0:3]), f[..., 3:4]), dim=-1)
        if f.shape[-1] == 4
        else _srgb_to_rgb(f)
    )
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out


def get_brdf_lut() -> torch.Tensor:
    brdf_lut_path = os.path.join(os.path.dirname(__file__), "brdf_256_256.bin")
    brdf_lut = torch.from_numpy(
        np.fromfile(brdf_lut_path, dtype=np.float32).reshape(1, 256, 256, 2)
    )
    return brdf_lut

def get_envmap_dirs(res: List[int] = [512, 1024]) -> torch.Tensor:
    gy, gx = torch.meshgrid(
        torch.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
        torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
        indexing="ij",
    )

    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

    reflvec = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)  # [H, W, 3]
    return reflvec


def compute_ks(F0, O, NdotV):
    # F0: (...,)   - base reflectance (e.g., 0.04 or RGB)
    # O:  (...,)   - opacity (scalar or tensor)
    # NdotV: (...,) - dot product between normal and view vector (cosine)

    # Ensure tensor
    if not isinstance(F0, torch.Tensor):
        F0 = torch.tensor(F0, dtype=torch.float32)
    if not isinstance(O, torch.Tensor):
        O = torch.tensor(O, dtype=torch.float32)
    if not isinstance(NdotV, torch.Tensor):
        NdotV = torch.tensor(NdotV, dtype=torch.float32)

    base = torch.maximum(1.0 - O, F0) - F0
    exponent = (-5.55473 * NdotV - 6.698316) * NdotV
    ks = F0 + base * torch.pow(2.0, exponent)

    return ks

def save_cubemap_cross(cubemap_tensor, save_path):
    """
    cubemap_tensor: [6, H, W, 3]  → save as cross layout [H*3, W*4, 3]
    Order: [right, left, top, bottom, front, back]
    Layout:
           [    top     ]
    [left, front, right, back]
           [  bottom    ]
    """
    right, left, top, bottom, front, back = [cubemap_tensor[i] for i in range(6)]
    H, W = top.shape[:2]

    # Create empty tensor for [3H, 4W, 3]
    cross = torch.zeros((H*3, W*4, 3), dtype=cubemap_tensor.dtype)

    # Assign to positions
    cross[0:H, W:2*W]       = top
    cross[H:2*H, 0:W]       = left
    cross[H:2*H, W:2*W]     = front
    cross[H:2*H, 2*W:3*W]   = right
    cross[H:2*H, 3*W:4*W]   = back
    cross[2*H:3*H, W:2*W]   = bottom

    # Save image
    cross_img = cross.permute(2, 0, 1).clamp(0, 1)  # [3, H*3, W*4]
    save_image(cross_img, save_path)

def pbr_shading(
    light: Union[CubemapLight,DirectLightMap],
    normals: torch.Tensor,  # [H, W, 3]
    view_dirs: torch.Tensor,  # [H, W, 3]
    albedo: torch.Tensor,  # [H, W, 3]
    roughness: torch.Tensor,  # [H, W, 1]
    mask: torch.Tensor,  # [H, W, 1]
    tone: bool = False, # false
    gamma: bool = False, # true
    occlusion: Optional[torch.Tensor] = None,  # [H, W, 1]
    irradiance: Optional[torch.Tensor] = None,  # [H, W, 1]
    metallic: Optional[torch.Tensor] = None,
    brdf_lut: Optional[torch.Tensor] = None,
    background: Optional[torch.Tensor] = None,
) -> Dict:
    H, W, _ = normals.shape
    if background is None:
        background = torch.zeros_like(normals)  # [H, W, 3]

    # prepare
    normals = normals.reshape(1, H, W, 3)
    view_dirs = view_dirs.reshape(1, H, W, 3)
    albedo = albedo.reshape(1, H, W, 3)
    roughness = roughness.reshape(1, H, W, 1)

    results = {}
    # prepare
    ref_dirs = (
        2.0 * (normals * view_dirs).sum(-1, keepdims=True).clamp(min=0.0) * normals - view_dirs
    )  # [1, H, W, 3]

    # # Diffuse lookup
    # save_cubemap_cross(light.diffuse, f"/hdd1/csg/src/MultiGSFace_main/debug/{light_name}_cube_diffuse.png")
    # save_cubemap_cross(light, f"/hdd1/csg/src/MultiGSFace_main/debug/cube.png")

    diffuse_light = dr.texture(
        light.diffuse[None, ...],  # [1, 6, 16, 16, 3]
        # light.base[None, ...],  # [1, 6, 16, 16, 3]
        normals.contiguous(),  # [1, H, W, 3]
        filter_mode="linear",
        boundary_mode="cube",
    )  # [1, H, W, 3]

    if occlusion is not None:
        diffuse_light = diffuse_light * occlusion[None] + (1 - occlusion[None]) * irradiance[None]

    results["diffuse_light"] = diffuse_light[0]
    result = results["diffuse_light"]# * (mask/255.0) 
    # result = torch.clamp(result, 0.0, 1.0)
    # save_image(result.permute(2,0,1), f"/hdd1/csg/src/MultiGSFace_main/debug/{light_name}_shading.jpg")

    diffuse_rgb = diffuse_light[0] * albedo[0]  # [H, W, 3]
    # save_image(diffuse_rgb.permute(2,0,1), f"/hdd1/csg/src/MultiGSFace_main/debug/{light_name}_diffuse.jpg")
    
    # return None
 
    # specular
    NoV = saturate_dot(normals, view_dirs)  # [1, H, W, 1]
    fg_uv = torch.cat((NoV, roughness), dim=-1)  # [1, H, W, 2]
    fg_lookup = dr.texture(
        brdf_lut,  # [1, 256, 256, 2]
        fg_uv.contiguous(),  # [1, H, W, 2]
        filter_mode="linear",
        boundary_mode="clamp",
    )  # [1, H, W, 2]

    # Roughness adjusted specular env lookup
    miplevel = light.get_mip(roughness)  # [1, H, W, 1]
    spec = dr.texture(
        light.specular[0][None, ...],  # [1, 6, env_res, env_res, 3]
        ref_dirs.contiguous(),  # [1, H, W, 3]
        mip=list(m[None, ...] for m in light.specular[1:]),
        mip_level_bias=miplevel[..., 0],  # [1, H, W]
        filter_mode="linear-mipmap-linear",
        boundary_mode="cube",
    )  # [1, H, W, 3]
    
    # # Compute aggregate lighting
    if metallic is None:
        F0 = torch.ones_like(albedo) * 0.04  # [1, H, W, 3]
    else:
        F0 = (1.0 - metallic) * 0.04 + albedo * metallic
        
    F0 = compute_ks(metallic, roughness, NoV)
    # F0 = torch.ones_like(albedo)
    
    ##################################################################################    
    fg0 = fg_lookup[..., 0:1] # [1, H, W]
    fg1 = fg_lookup[..., 1:2] # [1, H, W]

    reflectance = F0 * fg0 + fg1  # [1, H, W, 3]
    
    
    specular_rgb = spec * reflectance  # [1, H, W, 3]

    render_rgb = diffuse_rgb + specular_rgb  # [1, H, W, 3]

    render_rgb = render_rgb.squeeze()  # [H, W, 3]

    if tone:
        # Tone Mapping
        render_rgb = aces_film(render_rgb)
    else:
        render_rgb = render_rgb.clamp(min=0.0, max=1.0)

    ### NOTE: close `gamma` will cause better resuls in novel view synthesis but wrose relighting results.
    ### NOTE: it is worth to figure out a better way to handle both novel view synthesis and relighting
    if gamma:
        render_rgb = linear_to_srgb(render_rgb.squeeze())

    # render_rgb = torch.where(mask, render_rgb, background)
    results.update(
        {
            "render": render_rgb*mask,
            "diffuse": diffuse_rgb.squeeze()*mask,
            "specular": specular_rgb.squeeze()*mask,
            "spec":spec.squeeze(),
            "reflectance":reflectance.squeeze(),
            "F0":F0[0],
            "fg0":fg0.squeeze(),
            "fg1":fg1.squeeze()
        }
    )

    return results

def rasterize_gs(viewpoint_cam, gaussian, background, device):
        
    render_out = render(
        viewpoint_cam,
        gaussian,
        background.to(device),
        device=device,
        override_color=gaussian._features_dc.squeeze()  # Here
    )
    gaussian._features_dc
    render_image_       = render_out['render']
    viewspace_points_   = render_out['viewspace_points']
    visibility_filter_  = render_out['visibility_filter']
    radii_              = render_out['radii']
    

    output = {
        'rgb_image': render_image_,
        'normal_from_depth':render_out["normal_map_from_depth"],
        'normal_map':render_out["normal_map"],
        'albedo_map':render_out["albedo_map"],
        'roughness_map':render_out["roughness_map"],
        'metallic_map':render_out["metallic_map"],
        # ----- gaussian maintainer ----- #
        'viewspace_points': viewspace_points_,   # List
        'visibility_filter': visibility_filter_, # List
        'radii': radii_, # List
        'gaussian':gaussian,
    }

    return output




