import torch
import torch.nn as nn
import sys
from datetime import datetime
import numpy as np
import random
from typing import NamedTuple
from errno import EEXIST
from os import makedirs, path
import os
import PIL

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   

def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert 4 >= deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    # if deg == 0 and result.shape[-1] == 1:
    #     result = result.expand_as(dirs)
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                  C1 * y * sh[..., 1] +
                  C1 * z * sh[..., 2] -
                  C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                      C2[0] * xy * sh[..., 4] +
                      C2[1] * yz * sh[..., 5] +
                      C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                      C2[3] * xz * sh[..., 7] +
                      C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                          C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                          C3[1] * xy * z * sh[..., 10] +
                          C3[2] * y * (4 * zz - xx - yy) * sh[..., 11] +
                          C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                          C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                          C3[5] * z * (xx - yy) * sh[..., 14] +
                          C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                              C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                              C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                              C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                              C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                              C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                              C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                              C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                              C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
    
def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
    
    
def inverse_sigmoid(x):
    return torch.log(x/(1-x))



def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def get_minimum_axis(scales, rotations):
    sorted_idx = torch.argsort(scales, descending=False, dim=-1) # N 3
    R = build_rotation(rotations) # quat to matrix N 3 3 
    R_sorted = torch.gather(R, dim=2, index=sorted_idx[:,None,:].repeat(1, 3, 1)).squeeze()
    x_axis = R_sorted[:,0,:] # normalized by defaut

    return x_axis

def get_bg_color(bg_color:str):
    """Range between 0, 1"""
    if bg_color == 'white':
        background = torch.ones((3,), dtype=torch.float32)
    elif bg_color == "black":
        background = torch.zeros((3,), dtype=torch.float32)
    elif bg_color == "random":
        background = torch.rand((3,), dtype=torch.float32)
    else:
        raise ValueError("Invalid Color!")
    return background

def load_to_gpu(model_input, device):
    model_input.original_image = model_input.original_image.to(device)
    model_input.original_mask = model_input.original_mask.to(device)
    model_input.camera_center = model_input.camera_center.to(device)
    if model_input.original_normal is not None:
        model_input.original_normal = model_input.original_normal.to(device)

    model_input.world_view_transform = model_input.world_view_transform.to(device)
    model_input.camera_center = model_input.camera_center.to(device)



    for k, v in model_input.flame_param.items():
        model_input.flame_param[k] = v.to(device)
        
def load_mask(path, img_res):
    """
    Load and preprocess a binary mask image from the specified path.

    Args:
        path (str): The path to the mask image file.
        img_res (tuple): The target resolution (width, height) to resize the mask image.

    Returns:
        numpy.ndarray: A 2D numpy array representing the mask with pixel values in the range [0, 1].
    """

    img = PIL.Image.open(path)
    img = img.convert('L')
    img = img.resize(img_res)
    img = np.array(img)

    object_mask = img / 255

    return object_mask


def _to_chw01(img: torch.Tensor) -> torch.Tensor:
    """이미지 텐서를 CHW, float32, [0,1]로 정규화."""
    x = img.detach()
    if x.dim() == 3:
        # CHW 또는 HWC
        if x.shape[0] in (1, 3):
            pass  # CHW
        elif x.shape[-1] in (1, 3):
            x = x.permute(2, 0, 1)
        else:
            raise ValueError(f"Unexpected image shape: {tuple(x.shape)}")
    elif x.dim() == 2:
        x = x.unsqueeze(0)  # 1HW
    else:
        raise ValueError(f"Unexpected image dim: {x.dim()}")
    return x.float().clamp_(0.0, 1.0)


class RodriguesModule(nn.Module):
    def forward(self, rvec):
        r''' Apply Rodriguez formula on a batch of rotation vectors.

            Args:
                rvec: Tensor (B, 3)
            
            Returns
                rmtx: Tensor (B, 3, 3)
        '''
        theta = torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=1))
        rvec = rvec / theta[:, None]
        costh = torch.cos(theta)
        sinth = torch.sin(theta)
        return torch.stack((
            rvec[:, 0] ** 2 + (1. - rvec[:, 0] ** 2) * costh,
            rvec[:, 0] * rvec[:, 1] * (1. - costh) - rvec[:, 2] * sinth,
            rvec[:, 0] * rvec[:, 2] * (1. - costh) + rvec[:, 1] * sinth,

            rvec[:, 0] * rvec[:, 1] * (1. - costh) + rvec[:, 2] * sinth,
            rvec[:, 1] ** 2 + (1. - rvec[:, 1] ** 2) * costh,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) - rvec[:, 0] * sinth,

            rvec[:, 0] * rvec[:, 2] * (1. - costh) - rvec[:, 1] * sinth,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) + rvec[:, 0] * sinth,
            rvec[:, 2] ** 2 + (1. - rvec[:, 2] ** 2) * costh), 
        dim=1).view(-1, 3, 3)
    