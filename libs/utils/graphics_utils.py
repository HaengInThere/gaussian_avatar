#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch3d
from pytorch3d.renderer import MeshRenderer, MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes

def rgb_to_srgb(img, clip=True):
    # hdr img
    if isinstance(img, np.ndarray):
        assert len(img.shape) == 3, img.shape
        assert img.shape[2] == 3, img.shape
        img = np.where(img > 0.0031308, np.power(np.maximum(img, 0.0031308), 1.0 / 2.4) * 1.055 - 0.055, 12.92 * img)
        if clip:
            img = np.clip(img, 0.0, 1.0)
        return img
    elif isinstance(img, torch.Tensor):
        assert len(img.shape) == 3, img.shape
        assert img.shape[0] == 3, img.shape
        img = torch.where(img > 0.0031308, torch.pow(torch.max(img, torch.tensor(0.0031308)), 1.0 / 2.4) * 1.055 - 0.055, 12.92 * img)
        if clip: 
            img = torch.clamp(img, 0.0, 1.0)
        return img
    else:
        raise TypeError("Unsupported input type. Supported types are numpy.ndarray and torch.Tensor.")

def srgb_to_rgb(img):
    # f is LDR
    if isinstance(img, np.ndarray):
        assert len(img.shape) == 3, img.shape
        assert img.shape[2] == 3, img.shape
        img = np.where(img <= 0.04045, img / 12.92, np.power((np.maximum(img, 0.04045) + 0.055) / 1.055, 2.4))
        return img
    elif isinstance(img, torch.Tensor):
        assert len(img.shape) == 3, img.shape
        assert img.shape[0] == 3, img.shape
        img = torch.where(img <= 0.04045, img / 12.92, torch.pow((torch.max(img, torch.tensor(0.04045)) + 0.055) / 1.055, 2.4))
        return img
    else:
        raise TypeError("Unsupported input type. Supported types are numpy.ndarray and torch.Tensor.")

def rotation_between_z(vec):
    v1 = -vec[..., 1]
    v2 = vec[..., 0]
    v3 = torch.zeros_like(v1)
    v11 = v1 * v1
    v22 = v2 * v2
    v33 = v3 * v3
    v12 = v1 * v2
    v13 = v1 * v3
    v23 = v2 * v3
    cos_p_1 = (vec[..., 2] + 1).clamp_min(1e-7)
    R = torch.zeros(vec.shape[:-1] + (3, 3,), dtype=torch.float32, device="cuda")
    R[..., 0, 0] = 1 + (-v33 - v22) / cos_p_1
    R[..., 0, 1] = -v3 + v12 / cos_p_1
    R[..., 0, 2] = v2 + v13 / cos_p_1
    R[..., 1, 0] = v3 + v12 / cos_p_1
    R[..., 1, 1] = 1 + (-v33 - v11) / cos_p_1
    R[..., 1, 2] = -v1 + v23 / cos_p_1
    R[..., 2, 0] = -v2 + v13 / cos_p_1
    R[..., 2, 1] = v1 + v23 / cos_p_1
    R[..., 2, 2] = 1 + (-v22 - v11) / cos_p_1
    R = torch.where((vec[..., 2] + 1 > 0)[..., None, None], R,
                    -torch.eye(3, dtype=torch.float32, device="cuda").expand_as(R))
    return R

def fibonacci_sphere_sampling(normals, sample_num, random_rotate=True):
    pre_shape = normals.shape[:-1]
    if len(pre_shape) > 1:
        normals = normals.reshape(-1, 3)
    delta = np.pi * (3.0 - np.sqrt(5.0))

    # fibonacci sphere sample around z axis
    idx = torch.arange(sample_num, dtype=torch.float, device='cuda')[None]
    z = 1 - 2 * idx / (2 * sample_num - 1)
    rad = torch.sqrt(1 - z ** 2)
    theta = delta * idx
    if random_rotate:
        theta = torch.rand(*pre_shape, 1, device='cuda') * 2 * np.pi + theta
    y = torch.cos(theta) * rad
    x = torch.sin(theta) * rad
    z_samples = torch.stack([x, y, z.expand_as(y)], dim=-2)

    # rotate to normal
    # z_vector = torch.zeros_like(normals)
    # z_vector[..., 2] = 1  # [H, W, 3]
    # rotation_matrix = rotation_between_vectors(z_vector, normals)
    rotation_matrix = rotation_between_z(normals)
    incident_dirs = rotation_matrix @ z_samples
    incident_dirs = F.normalize(incident_dirs, dim=-2).transpose(-1, -2)
    incident_areas = torch.ones_like(incident_dirs)[..., 0:1] * 2 * np.pi
    if len(pre_shape) > 1:
        incident_dirs = incident_dirs.reshape(*pre_shape, sample_num, 3)
        incident_areas = incident_areas.reshape(*pre_shape, sample_num, 1)
    return incident_dirs, incident_areas

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

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

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))



# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0,1), mode='constant', value=w)

def compute_face_normals(verts, faces):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    return face_normals

def compute_face_orientation(verts, faces, return_scale=False):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]

    a0 = safe_normalize(v1 - v0)
    a1 = safe_normalize(torch.cross(a0, v2 - v0, dim=-1))
    a2 = -safe_normalize(torch.cross(a1, a0, dim=-1))  # will have artifacts without negation

    orientation = torch.cat([a0[..., None], a1[..., None], a2[..., None]], dim=-1)

    if return_scale:
        s0 = length(v1 - v0)
        s1 = dot(a2, (v2 - v0)).abs()
        scale = (s0 + s1) / 2
    return orientation, scale

# def compute_vertex_normals(verts, faces):
#     i0 = faces[..., 0].long()
#     i1 = faces[..., 1].long()
#     i2 = faces[..., 2].long()

#     v0 = verts[..., i0, :]
#     v1 = verts[..., i1, :]
#     v2 = verts[..., i2, :]
#     face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
#     v_normals = torch.zeros_like(verts)
#     N = verts.shape[0]
#     v_normals.scatter_add_(1, i0[..., None].repeat(N, 1, 3), face_normals)
#     v_normals.scatter_add_(1, i1[..., None].repeat(N, 1, 3), face_normals)
#     v_normals.scatter_add_(1, i2[..., None].repeat(N, 1, 3), face_normals)

#     v_normals = torch.where(dot(v_normals, v_normals) > 1e-20, v_normals, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
#     v_normals = safe_normalize(v_normals)
#     if torch.is_anomaly_enabled():
#         assert torch.all(torch.isfinite(v_normals))
#     return v_normals


def compute_vertex_normals(verts, faces):
    # verts N,3
    # face F-N, 3
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    v_normals = torch.zeros_like(verts)
    N = verts.shape[0]
    # Face Normal을 Vertex Normal에 더하기
    v_normals.scatter_add_(0, i0[:, None].expand(-1, 3), face_normals)
    v_normals.scatter_add_(0, i1[:, None].expand(-1, 3), face_normals)
    v_normals.scatter_add_(0, i2[:, None].expand(-1, 3), face_normals)

    v_normals = torch.where(dot(v_normals, v_normals) > 1e-20, v_normals, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
    v_normals = safe_normalize(v_normals)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_normals))
    return v_normals


def project_gaussians_to_screenspace(means3D, viewpoint_camera):
    """
    Projects 3D Gaussian centers to 2D screenspace coordinates.

    Args:
        means3D (torch.Tensor): (N, 3) tensor of Gaussian centers in world coordinates.
        viewpoint_camera: Camera object with projection matrices.

    Returns:
        torch.Tensor: (N, 2) tensor of Gaussian centers in image coordinates.
    """
    device = means3D.device
    N = means3D.shape[0]
    
    # Convert means3D to homogeneous coordinates (N, 4)
    ones = torch.ones((N, 1), device=device)
    posw = torch.cat([means3D, ones], dim=1)  # (N, 4)
    
    # Clone and modify the transformation matrices to avoid altering the originals
    view_matrix = viewpoint_camera.world_view_transform.clone()
    view_matrix[:, 1] = -view_matrix[:, 1]  # Flip y-axis
    view_matrix[:, 2] = -view_matrix[:, 2]  # Flip z-axis
    RT = view_matrix.T[None, ...]
    
    proj_matrix = viewpoint_camera.full_proj_transform.clone()
    proj_matrix[:, 1] = -proj_matrix[:, 1]  # Flip y-axis
    full_proj = proj_matrix.T[None, ...].cuda()
    
    # Compute the Model-View-Projection (MVP) matrix
    # transform_matrix = proj_matrix @ view_matrix  # (4, 4)
    # transform_matrix = transform_matrix.cuda()
    
    # Apply the transformation: (N, 4) @ (4, 4)^T = (N, 4)
    projected = posw @ full_proj.T  # (N, 4)
    
    # Normalize by the w component to get normalized device coordinates (N, 3)
    projected = projected[:, :3] / (projected[:, 3].unsqueeze(1) + 1e-8)  # (N, 3)
    
    # Map normalized device coordinates to image coordinates
    x = (projected[:, 0] + 1) * 0.5 * (viewpoint_camera.image_width - 1)
    y = (1 - (projected[:, 1] + 1) * 0.5) * (viewpoint_camera.image_height - 1)  # Flip y-axis
    
    # Stack x and y to get (N, 2) tensor
    coords_2d = torch.stack([x, y], dim=1)  # (N, 2)
    
    return coords_2d

def project_3d_to_screenspace(means3D, viewpoint_camera):
    """
    Projects 3D Gaussian centers to 2D screenspace coordinates.

    Args:
        means3D (torch.Tensor): (N, 3) tensor of Gaussian centers in world coordinates.
        viewpoint_camera: Camera object with projection matrices.

    Returns:
        torch.Tensor: (N, 2) tensor of Gaussian centers in image coordinates.
    """
    device = means3D.device
    # Convert means3D to homogeneous coordinates
    ones = torch.ones((means3D.shape[0], 1), device=device)
    means3D_hom = torch.cat([means3D, ones], dim=1).T  # (4, N)

    # Compute the transformation matrix
    view_matrix = viewpoint_camera.world_view_transform.clone()
    view_matrix[:, 1] = -view_matrix[:, 1]  # Flip y-axis
    view_matrix[:, 2] = -view_matrix[:, 2]  # Flip z-axis

    proj_matrix = viewpoint_camera.full_proj_transform.clone()
    proj_matrix[:, 1] = -proj_matrix[:, 1]  # Flip y-axis

    transform_matrix = proj_matrix @ view_matrix  # (4, 4)

    # Project points
    projected = transform_matrix @ means3D_hom  # (4, N)

    # Normalize by w
    projected = projected[:3, :] / (projected[3, :] + 1e-8)  # (3, N)

    # Map to image coordinates
    x = (projected[0, :] + 1) * 0.5 * (viewpoint_camera.image_width - 1)
    y = (1 - (projected[1, :] + 1) * 0.5) * (viewpoint_camera.image_height - 1)  # Flip y-axis

    # Stack coordinates
    coords_2d = torch.stack([x, y], dim=1)  # (N, 2)
    return coords_2d


def sample_colors_at_gaussians(image, coords_2d):
    """
    Samples colors from the image at the given 2D coordinates using grid_sample.

    Args:
        image (torch.Tensor): Image tensor of shape (C, H, W).
        coords_2d (torch.Tensor): 2D coordinates of Gaussians (N, 2).
        image_size (tuple): Tuple of image size (H, W).

    Returns:
        torch.Tensor: Sampled colors of shape (N, C).
    """
    C, H, W = image.shape
    device = image.device

    # Normalize coordinates to [-1, 1] for grid_sample
    x = coords_2d[:, 0]
    y = coords_2d[:, 1]

    # Clamp coordinates to be within image bounds
    x = torch.clamp(x, 0, W - 1)
    y = torch.clamp(y, 0, H - 1)

    x_norm = (x / (W - 1)) * 2 - 1  # [0, W-1] -> [-1, 1]
    y_norm = (y / (H - 1)) * 2 - 1  # [0, H-1] -> [-1, 1]

    grid = torch.stack((x_norm, y_norm), dim=1)  # (N, 2)
    grid = grid.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)

    # Prepare image for grid_sample
    image = image.unsqueeze(0)  # (1, C, H, W)

    # Sample colors using grid_sample
    sampled = F.grid_sample(image, grid, align_corners=True)  # (1, C, N, 1)
    sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)  # (N, C)

    return sampled  # (N, C)


import torch
import torch.nn.functional as F

def crop_face(coords_2d, image, crop_size=224):
    """
    Crops the image around the projected 2D points to a square bounding box,
    then resizes it to the desired crop_size.

    Args:
        coords_2d (torch.Tensor): Projected 2D points of shape (1, num_points, 2).
        image (torch.Tensor): Original image of shape (C, H, W).
        crop_size (int): Desired size of the cropped image (crop_size x crop_size).

    Returns:
        cropped_image (torch.Tensor): Cropped and resized image of shape (C, crop_size, crop_size).
    """
    device = coords_2d.device
    coords_2d = coords_2d.squeeze(0)  # Shape: (num_points, 2)

    # Get min and max coordinates
    min_xy = torch.floor(coords_2d.min(dim=0)[0])  # (2,)
    max_xy = torch.ceil(coords_2d.max(dim=0)[0])   # (2,)

    # Compute center of the bounding box
    center_xy = (min_xy + max_xy) / 2  # (2,)

    # Compute size of the bounding box
    bbox_size = max_xy - min_xy  # (2,)
    bbox_side = torch.max(bbox_size)  # Scalar, the side length of the square bounding box

    # Adjust min and max coordinates to make the bounding box square
    half_side = bbox_side / 2
    min_x = center_xy[0] - half_side
    min_y = center_xy[1] - half_side
    max_x = center_xy[0] + half_side
    max_y = center_xy[1] + half_side

    # Convert coordinates to integers
    min_x = int(torch.floor(min_x).item())
    min_y = int(torch.floor(min_y).item())
    max_x = int(torch.ceil(max_x).item())
    max_y = int(torch.ceil(max_y).item())

    # Image dimensions
    C, H, W = image.shape

    # Clamp coordinates to image boundaries
    min_x = max(min_x, 0)
    min_y = max(min_y, 0)
    max_x = min(max_x, W)
    max_y = min(max_y, H)

    # Ensure the crop has non-zero size
    if min_x >= max_x or min_y >= max_y:
        print("Invalid bounding box. Returning None.")
        return None

    # Crop the image
    cropped_image = image[:, min_y:max_y, min_x:max_x]

    # Resize the cropped image to (crop_size, crop_size)
    cropped_image = F.interpolate(
        cropped_image.unsqueeze(0), size=(crop_size, crop_size), mode='bilinear', align_corners=False
    ).squeeze(0)

    return cropped_image


def save_coords_2d_image(coords_2d, W, H, save_path):
    import matplotlib.pyplot as plt
    """
    Visualizes the coords_2d tensor and saves it as an image.
    
    Args:
        coords_2d (torch.Tensor): Tensor of shape (N, 2), on GPU or CPU.
        W (int): Width of the image.
        H (int): Height of the image.
        save_path (str): Path to save the image.
    """
    coords_cpu = coords_2d.detach().cpu().numpy()
    plt.figure(figsize=(W/100, H/100), dpi=100)
    plt.scatter(coords_cpu[:, 0], coords_cpu[:, 1], s=1)
    plt.xlim(0, W)
    plt.ylim(H, 0)  # Flip Y-axis for image coordinates
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_coords_2d_image_numbering(coords_2d, W, H, save_path):
    import matplotlib.pyplot as plt
    """
    Visualizes the coords_2d tensor and saves it as an image with index labels.
    
    Args:
        coords_2d (torch.Tensor): Tensor of shape (N, 2), on GPU or CPU.
        W (int): Width of the image.
        H (int): Height of the image.
        save_path (str): Path to save the image.
    """
    coords_cpu = coords_2d.detach().cpu().numpy()
    plt.figure(figsize=(W/10, H/10), dpi=100)
    plt.xlim(0, W)
    plt.ylim(H, 0)  # Flip Y-axis for image coordinates
    plt.axis('off')
    
    for i, (x, y) in enumerate(coords_cpu):
        plt.scatter(x, y, s=10, color='blue')
        plt.text(x, y, str(i), fontsize=50, color='red',
                 bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    

def create_mask(coords_2d, image_size=512, radius=3):
    """
    Creates a mask for Gaussian splatting points.

    Args:
        coords_2d (torch.Tensor): (N, 2) tensor of 2D coordinates in image space.
        image_size (int): Size of the square image (assumed H=W=image_size).
        radius (int): Radius for Gaussian point coverage.

    Returns:
        torch.Tensor: Mask tensor (1, H, W) on the same device as coords_2d.
    """
    device = coords_2d.device  # 입력된 좌표의 장치 확인
    mask = torch.zeros((1, image_size, image_size), device=device)  # 마스크 텐서 초기화
    pixel_coords = coords_2d.round().long()  # 소수점을 반올림하여 정수 픽셀 좌표로 변환
    
    # 이미지 경계를 벗어난 좌표를 클램프
    pixel_coords[:, 0] = pixel_coords[:, 0].clamp(0, image_size-1)
    pixel_coords[:, 1] = pixel_coords[:, 1].clamp(0, image_size-1)
    
    # Gaussian 포인트에 해당하는 픽셀을 1로 설정
    mask[0, pixel_coords[:,1], pixel_coords[:,0]] = 1
    
    # 원형 커널 생성
    kernel_size = 2 * radius + 1
    y, x = torch.meshgrid(
        torch.arange(kernel_size, device=device),
        torch.arange(kernel_size, device=device)
    )
    center = radius
    dist_sq = (x - center)**2 + (y - center)**2
    kernel = (dist_sq <= radius**2).float().unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)
    
    # 컨볼루션을 통해 마스크 확대
    mask = mask.unsqueeze(0)  # (1, 1, H, W)
    mask = F.conv2d(mask, kernel, padding=radius)  # (1, 1, H, W)
    mask = (mask > 0).float().squeeze(0).squeeze(0)  # (H, W)
    mask = mask.unsqueeze(0)  # (1, H, W)
    
    return mask

def compute_vertex_visibility(vertices, faces, camera, image_size):
    """
    Compute per-vertex visibility mask from camera perspective.
    
    vertices: (B, V, 3) Tensor of FLAME model vertices
    faces: (F, 3) Face indices
    camera: Pytorch3D camera
    image_size: (H, W) of rendered image
    
    Returns:
    visibility_mask: (B, V) binary mask (1 if visible, 0 if occluded)
    """
    # Pytorch3D Mesh 구조 생성
    mesh = Meshes(verts=vertices, faces=faces.unsqueeze(0).expand(vertices.shape[0], -1, -1))

    # Rasterizer 설정 (Depth 기반)
    raster_settings = RasterizationSettings(
        image_size=image_size,  # Output image resolution
        blur_radius=0.0,  
        faces_per_pixel=1
    )

    # Rasterizer로 visibility map 얻기
    rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings)
    fragments = rasterizer(mesh)

    # Depth 값 추출 (H, W, 1) → Visible Pixel Depth
    depth_map = fragments.zbuf[..., 0]  # (B, H, W)

    # Vertex Depth와 비교하여 Visibility 결정
    verts_depth = vertices[:, :, 2]  # Z 값 기준
    visibility_mask = (verts_depth < depth_map.max(dim=1, keepdim=True)[0])  # True = Visible

    return visibility_mask.float()


def update_normal_expmap(
    base_n: torch.Tensor,      # (N,3), unit not required
    dnormal_raw: torch.Tensor, # (N,3), unconstrained net output
    angle_scale: float = 0.35, # 라디안 상한 (≈20°). 0.3~0.5 권장
    k: float = 1.0,            # 감도(스케일) 조절
    eps: float = 1e-8
) -> torch.Tensor:
    """
    n = cosθ * n0 + sinθ * u,  u는 접평면 방향 단위벡터
    θ = angle_scale * (2/π) * atan(k * ||t||)  ∈ [0, angle_scale]
    """
    n0 = F.normalize(base_n, dim=-1, eps=eps)

    # 접평면 성분으로 제한
    proj = (dnormal_raw * n0).sum(dim=-1, keepdim=True) * n0
    t = dnormal_raw - proj
    t_norm = t.norm(dim=-1, keepdim=True)  # (N,1)
    u = t / t_norm.clamp_min(eps)

    # 각도 매핑: 부드럽게 포화되는 atan
    theta = angle_scale * (2.0 / math.pi) * torch.atan(k * t_norm)

    # 작은 각도에서의 수치안정(테일러 근사)
    small = (t_norm < 1e-4)
    sin_t = torch.where(small, theta, torch.sin(theta))
    cos_t = torch.where(small, 1 - 0.5 * theta * theta, torch.cos(theta))

    n = cos_t * n0 + sin_t * u
    return F.normalize(n, dim=-1, eps=eps)


def quat_rotate(q, v):
    # q: (N,4) [w,x,y,z], v: (N,3)
    qw, qx, qy, qz = q.unbind(-1)
    # qvec × v
    t = torch.stack([
        qy*v[...,2] - qz*v[...,1],
        qz*v[...,0] - qx*v[...,2],
        qx*v[...,1] - qy*v[...,0]
    ], dim=-1)
    t = 2.0 * t
    # v' = v + qw*t + qvec × t
    v_rot = v + qw.unsqueeze(-1)*t + torch.stack([
        qy*t[...,2] - qz*t[...,1],
        qz*t[...,0] - qx*t[...,2],
        qx*t[...,1] - qy*t[...,0]
    ], dim=-1)
    return v_rot

def quat_from_two_vectors(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    최소 회전 쿼터니언: a -> b, q = [w, x, y, z]
    a,b: (...,3)
    """
    a = F.normalize(a, dim=-1, eps=eps)
    b = F.normalize(b, dim=-1, eps=eps)
    dot = (a * b).sum(dim=-1, keepdim=True)              # (...,1)
    cross = torch.cross(a, b, dim=-1)                    # (...,3)
    w = 1.0 + dot                                        # (...,1)
    q = torch.cat([w, cross], dim=-1)                    # (...,4)

    # 180도 근처 보정: w≈0이면 임의의 직교축을 사용
    near_pi = (w.abs() < 1e-6).squeeze(-1)
    if near_pi.any():
        a_np = a[near_pi]
        # a와 x축이 평행하면 y축 사용
        xaxis = torch.tensor([1.0, 0.0, 0.0], device=a.device).expand_as(a_np)
        axis = torch.cross(a_np, xaxis, dim=-1)
        small = (axis.norm(dim=-1, keepdim=True) < 1e-6).squeeze(-1)
        if small.any():
            yaxis = torch.tensor([0.0, 1.0, 0.0], device=a.device).expand_as(a_np[small])
            axis[small] = torch.cross(a_np[small], yaxis, dim=-1)
        axis = F.normalize(axis, dim=-1, eps=eps)
        q_pi = torch.cat([torch.zeros_like(axis[..., :1]), axis], dim=-1)   # w=0, xyz=axis
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
        q[near_pi] = q_pi
        return q
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)

def rodrigues_to_quat(rvec: torch.Tensor, angle_cap: float = 0.35, eps: float = 1e-8) -> torch.Tensor:
    """
    rvec = axis * angle  (축각 벡터)
    angle_cap로 각도를 클램프해서 수치 안정성 확보.
    반환: 쿼터니언 [w,x,y,z]
    """
    theta = rvec.norm(dim=-1, keepdim=True)                 # (...,1)
    scale = (angle_cap / theta.clamp_min(eps)).clamp(max=1.0)
    rvec = rvec * scale                                     # 각도 상한 적용

    theta = rvec.norm(dim=-1, keepdim=True).clamp_min(eps)
    axis  = rvec / theta
    half  = 0.5 * theta
    w = torch.cos(half)
    xyz = axis * torch.sin(half)
    q = torch.cat([w, xyz], dim=-1)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def vector_to_quat(v: torch.Tensor) -> torch.Tensor:
    """
    v: (..., 3) normalized direction vector
    return: (..., 4) quaternion (w, x, y, z)
    """
    z = torch.tensor([0, 0, 1], device=v.device, dtype=v.dtype).expand_as(v)
    v = F.normalize(v, dim=-1)

    # 회전축
    axis = torch.cross(z, v, dim=-1)
    axis_norm = torch.linalg.norm(axis, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = axis / axis_norm

    # 회전각
    cos_theta = (z * v).sum(dim=-1, keepdim=True).clamp(-1, 1)
    theta = torch.acos(cos_theta)

    # quaternion 생성
    half = theta * 0.5
    w = torch.cos(half)
    xyz = axis * torch.sin(half)
    return torch.cat([w, xyz], dim=-1)  # (w, x, y, z)


def rotate_canon_to_normal(base_normal: torch.Tensor, canon_xyz: torch.Tensor) -> torch.Tensor:
    """
    base_normal: (B,3,H,W)   – 회전 대상이 될 법선(목표 Z축)
    canon_xyz:   (B,3,H,W)   – 캐논 오프셋(대상 벡터)
    returns rotated_canon: (B,3,H,W)
    """
    # 0) 정규화된 목표 Z축
    z_axis = F.normalize(base_normal, dim=1, eps=1e-6)  # (B,3,H,W)

    # 1) y축 생성: x_old=[1,0,0]과 z축의 크로스. 단, 평행하면 x_old=[0,1,0] 사용
    old_x = torch.tensor([1.0, 0.0, 0.0], device=canon_xyz.device)
    alt_x = torch.tensor([0.0, 1.0, 0.0], device=canon_xyz.device)

    y_axis = torch.cross(old_x[None, :, None, None], z_axis, dim=1)
    alt_y  = torch.cross(alt_x[None, :, None, None], z_axis, dim=1)
    # 평행한 경우 대체축 사용
    y_axis = torch.where(y_axis.norm(dim=1, keepdim=True) < 1e-6, alt_y, y_axis)
    y_axis = F.normalize(y_axis, dim=1, eps=1e-6)

    # 2) x축 = z × y
    x_axis = torch.cross(z_axis, y_axis, dim=1)
    x_axis = F.normalize(x_axis, dim=1, eps=1e-6)

    # 3) 오프셋 좌표 분리
    v0 = canon_xyz[:, 0, :, :]  # (B,H,W)
    v1 = canon_xyz[:, 1, :, :]
    v2 = canon_xyz[:, 2, :, :]

    # 4) 새로운 좌표계로 변환: x*v0 + y*v1 + z*v2
    rotated = v0.unsqueeze(1) * x_axis + v1.unsqueeze(1) * y_axis + v2.unsqueeze(1) * z_axis
    return rotated





# ----------------------------
# UV-local ray-wise visibility prior
# ----------------------------
@torch.no_grad()
def _soft_visibility_prior(V_unit: torch.Tensor,
                           dist: torch.Tensor,
                           ang_thresh: float = 0.995,
                           beta: float = 40.0,
                           win: int = 3) -> torch.Tensor:
    """
    Compute a per-UV visibility prior from local ray-depth coherence.
    Idea: Gaussians seen under almost identical view directions likely 
    lie on the same camera ray. The nearest along that ray should be visible.
    We approximate per-UV transmittance by a weighted soft-min depth within a
    local KxK window, gated by angular alignment.

    Args:
        V_unit: [B,3,H,W] unit vectors from camera to Gaussian centers.
        dist:   [B,1,H,W] distances to camera.
        ang_thresh: cos(theta) threshold for "same ray" gating.
        beta:   softness for the soft-min and the final sigmoid scale.
        win:    neighborhood window size (odd, e.g., 3 or 5).
    Returns:
        vis_prior: [B,1,H,W] in (0,1), higher means more likely frontmost.
    """
    B, _, H, W = V_unit.shape
    K = win * win

    # Unfold local neighborhoods
    D_unf = F.unfold(dist, kernel_size=win, padding=win//2)           # [B, K, HW]
    V_unf = F.unfold(V_unit, kernel_size=win, padding=win//2)         # [B, 3*K, HW]

    # Center direction and depth
    Vc = V_unit.view(B, 3, -1)                                        # [B,3,HW]
    Dc = dist.view(B, 1, -1)                                          # [B,1,HW]

    # Reshape neighbors
    V_unf = V_unf.view(B, 3, K, H*W)                                  # [B,3,K,HW]
    D_unf = D_unf.view(B, K, H*W)                                     # [B,K,HW]

    # Cosine alignment with center direction
    cos = (V_unf * Vc.unsqueeze(2)).sum(dim=1)                        # [B,K,HW]
    # Gate to keep only almost-colinear neighbors
    gate = torch.clamp((cos - ang_thresh) / (1.0 - ang_thresh + 1e-6), min=0.0, max=1.0) + 1e-8  # [B,K,HW]

    # Weighted soft-min of depths
    # softmin(d) = -1/beta * log sum_i w_i * exp(-beta d_i)
    z = torch.logsumexp((-beta) * D_unf + gate.log(), dim=1)          # [B,HW]
    d_front = (-1.0 / beta) * z                                       # [B,HW]

    # Visibility prior: high if current depth is close to local front depth
    vis_prior = torch.sigmoid((d_front - Dc.squeeze(1)) * beta).view(B, 1, H, W)
    return vis_prior

# ----------------------------
# Global ray-wise visibility prior (directional binning)
# ----------------------------
@torch.no_grad()
def _global_ray_visibility_prior(
    V_unit: torch.Tensor,
    dist: torch.Tensor,
    n_theta: int = 32,
    n_phi: int = 64,
    beta: float = 40.0,
) -> torch.Tensor:
    """
    Compute a per-UV visibility prior by grouping *all* UV pixels that
    share nearly the same camera ray direction, regardless of UV proximity.
    Each direction bin accumulates a soft-min of depth using exp(-beta * d)
    weights, and each pixel compares its depth to its bin's front depth.

    Args:
        V_unit: [B,3,H,W] unit vectors camera→point.
        dist:   [B,1,H,W] distances.
        n_theta, n_phi: directional grid resolution over S^2.
        beta:   softness; larger → sharper front selection.
    Returns:
        vis_prior: [B,1,H,W] in (0,1).
    """
    B, _, H, W = V_unit.shape
    x = V_unit[:, 0].contiguous().view(B, -1)
    y = V_unit[:, 1].contiguous().view(B, -1)
    z = V_unit[:, 2].contiguous().view(B, -1).clamp(-1.0, 1.0)
    d = dist.view(B, -1)  # [B, HW]

    # Spherical coordinates → direction bins
    theta = torch.acos(z)  # [0, pi]
    phi = torch.atan2(y, x)
    two_pi = torch.tensor(2.0 * 3.141592653589793, device=V_unit.device, dtype=V_unit.dtype)
    phi = (phi % two_pi)  # [0, 2pi)

    it = torch.clamp((theta / 3.141592653589793 * n_theta).floor().long(), 0, n_theta - 1)
    ip = torch.clamp((phi / two_pi * n_phi).floor().long(), 0, n_phi - 1)
    bins = (it * n_phi + ip).view(B, -1)  # [B, HW]
    n_bins = n_theta * n_phi

    # 1) Per-bin depth minimum for numerical stability (shifted soft-min)
    #    d_min_bins[b, k] = min depth among pixels in bin k
    d_min_bins = d.new_full((B, n_bins), float('inf'))
    try:
        # PyTorch ≥ 1.13
        d_min_bins.scatter_reduce_(1, bins, d, reduce='amin', include_self=True)
    except Exception:
        # Fallback: approximate bin-min via two passes using scatter_add on masked values
        mask = torch.ones_like(d)
        # First pass: sum depths and counts per bin
        sum_bins = d.new_zeros(B, n_bins)
        cnt_bins = d.new_zeros(B, n_bins)
        for b in range(B):
            sum_bins[b].scatter_add_(0, bins[b], d[b])
            cnt_bins[b].scatter_add_(0, bins[b], mask[b])
        # Use average as a rough proxy if amin unsupported
        avg_bins = sum_bins / cnt_bins.clamp_min(1e-8)
        d_min_bins = torch.where(torch.isfinite(avg_bins), avg_bins, d.mean(dim=1, keepdim=True).expand_as(avg_bins))

    # Gather per-pixel bin-min and form shifted depths Δd ≥ 0
    d_min = torch.gather(d_min_bins, 1, bins)
    delta = (d - d_min).clamp_min(0.0)

    # 2) Stable soft-min front depth per bin using exp(-β Δd)
    #    This avoids underflow because Δd=0 at the nearest depth.
    w = torch.exp(-beta * delta)
    num = d.new_zeros(B, n_bins)
    den = d.new_zeros(B, n_bins)
    for b in range(B):
        num[b].scatter_add_(0, bins[b], w[b] * d[b])
        den[b].scatter_add_(0, bins[b], w[b])
    den = den.clamp_min(1e-8)
    d_front_bins = num / den  # [B, n_bins]

    # 3) Map back to per-pixel front depth and compute visibility prior
    d_front = torch.gather(d_front_bins, 1, bins).view(B, 1, H, W)
    vis_prior = torch.sigmoid((d_front - dist) * beta)
    return vis_prior
