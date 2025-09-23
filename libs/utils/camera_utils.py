import torch
from torch import nn
import numpy as np
import math, os
from typing import NamedTuple
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    BlendParams,
    SoftPhongShader,
    SoftSilhouetteShader,
    MeshRenderer,
    look_at_view_transform,
    MeshRendererWithFragments
)
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import interpolate_face_attributes
from torchvision.utils import save_image

class Camera(nn.Module):
    def __init__(self, frame_i, cam_i, R, T, FoVx, FoVy, cx,cy,bg, image,image_width, image_height, base_path, image_path, mask_path, normal_path, 
                 albedo_path, roughness_path, metallic_path, lgt_path, flame_path,
                 image_name, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, timestep=None):
        super(Camera, self).__init__()

        self.frame_i = frame_i
        self.cam_i = cam_i
        self.R = R
        self.T = T
        self.image_width = image_width
        self.image_height = image_height
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.bg = bg
        self.image = image
        
        self.base_path = base_path
        self.image_path = image_path
        self.image_name = image_name
        self.mask_path = mask_path
        self.normal_path = normal_path
        self.albedo_path = albedo_path
        self.roughness_path = roughness_path
        self.metallic_path = metallic_path
        self.lgt_path = lgt_path
        self.flame_path = flame_path
        self.timestep = timestep

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        # self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)  #.cuda()
        # self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # self.camera_center = self.world_view_transform.inverse()[3, :3]

        K = np.eye(4)
        fx = fov2focal(FoVx, image_width)
        fy = fov2focal(FoVy, image_height)
        K[0, 0] = fx  # Focal length in x
        K[1, 1] = fy  # Focal length in y
        K[0, 2] = cx  # Principal point x
        K[1, 2] = cy  # Principal point y
        self.intrinsics = torch.tensor(K)
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix_vhap(torch.Tensor(K), self.image_height, self.image_width, znear=self.znear, zfar=self.zfar)
            
        self.world_view_transform = self.world_view_transform.transpose(0,1)
        self.world_view_transform[1:3] *= -1

        self.full_proj_transform = torch.bmm(self.projection_matrix.unsqueeze(0), self.world_view_transform.unsqueeze(0)).squeeze(0)
        
        # return to original 
        self.world_view_transform[1:3] *= -1 
        self.world_view_transform = self.world_view_transform.transpose(0,1)
        self.full_proj_transform = self.full_proj_transform.transpose(0,1)
        self.full_proj_transform[:,1] *= -1
        
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def get_camera_position(viewpoint_cam, device):
    if hasattr(viewpoint_cam, "camera_center") and viewpoint_cam.camera_center is not None:
        return viewpoint_cam.camera_center.to(device).view(3)
    w2c = viewpoint_cam.world_view_transform.to(device)
    c2w = torch.inverse(w2c)
    return c2w[:3, 3]

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrix_vhap(K: torch.Tensor, H, W, znear=0.001, zfar=1000):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    s = K[0, 1]
    P = torch.zeros(4, 4, dtype=K.dtype, device=K.device)
    
    P[0, 0]  = fx * 2 / W 
    P[1, 1]  = fy * 2 / H
    P[0, 2]  = (W - 2 * cx) / W
    P[1, 2]  = (H - 2 * cy) / H
    P[2, 2]  = -(zfar+znear) / (zfar-znear)
    P[2, 3]  = -2*zfar*znear / (zfar-znear)
    P[3, 2]  = -1
    return P 


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def render_depth_map_mdi(vs, fvs, cam_info, output_dir='./', rasterizer=None, division=1, debug=False, device='cuda'):
    """
    Render a depth map from 3D vertices and faces using PyTorch3D.

    Args:
        vs (np.ndarray): Vertices (N, 3).
        fvs (np.ndarray): Faces (M, 3).
        image_size (int): Image size (H=W).
        R (np.ndarray): Rotation matrix (3x3).
        T (np.ndarray): Translation vector (3,).
        fov_degrees (float): Field of view in degrees.
        device (str): 'cuda' or 'cpu'.

    Returns:
        depth (H, W) torch.Tensor: Depth buffer in world units.
        mask (H, W) torch.Tensor: Foreground mask.
    """
    device = torch.device(device)
    vs = torch.tensor(vs, dtype=torch.float32, device=device).unsqueeze(0)
    fvs = torch.tensor(fvs, dtype=torch.int64, device=device)

    image_size = cam_info.image_height // division, cam_info.image_width // division
    world_view_transform = cam_info.world_view_transform.clone().to("cuda")
    world_view_transform[:,1] = -world_view_transform[:,1]
    world_view_transform[:,2] = -world_view_transform[:,2]
    RT = world_view_transform.T[None, ...]

    full_proj_transform = cam_info.full_proj_transform.clone()
    full_proj_transform[:,1] = -full_proj_transform[:,1]
    full_proj = full_proj_transform.T[None, ...].to("cuda")

    output = rasterizer.render_mesh(vs, fvs, RT, full_proj, image_size)
    
    normal_map = output["normal"] # 1 h w 3  
    depth_map = output["depth"] * (-1) # 1 h w
    mask = output["rast_out"][..., 2].flip(1) # 1 h w

    normal_map = normal_map + 1e-12

    # debug
    if debug:
        os.makedirs(output_dir, exist_ok=True)
        normal_map_vis = (normal_map+1)/2
        save_image(normal_map_vis[0].permute(2,0,1), os.path.join(output_dir, f'{cam_info.image_name}_normal.png'))
        depth_min = depth_map[depth_map > 0].min()
        depth_max = depth_map.max()
        depth_normalized = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
        depth_normalized = depth_normalized.clamp(0, 1)
        save_image(depth_normalized[0], os.path.join(output_dir, f'{cam_info.image_name}_depth.png'))


    return depth_map, normal_map, mask, image_size
