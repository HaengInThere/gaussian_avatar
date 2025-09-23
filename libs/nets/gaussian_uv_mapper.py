import torch
from torch import nn
import torch.nn.functional as F
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings, OrthographicCameras

class GaussianUVMapper(nn.Module):
    def __init__(self, flame_model=None, device='cuda', uvcoords=None, uvfaces=None, tri_faces=None):
        super(GaussianUVMapper, self).__init__()
        self.device = device
        
        # 1) 직접 전달된 UV 토폴로지(권장)
        if (uvcoords is not None) and (uvfaces is not None) and (tri_faces is not None):
            # uvcoords = uvcoords * 2 - 1 
            # uvcoords[...,0] = - uvcoords[...,0]
            # uvcoords[...,1] = - uvcoords[...,1]
            self.register_buffer('uvcoords', uvcoords)
            self.register_buffer('uvfaces', uvfaces)
            self.register_buffer('tri_faces', tri_faces)

        # 2) 외부 모델 없이 기본 FLAME OBJ 로드(프로젝트 경로 다르면 실패 가능)
        elif flame_model is None:
            _, faces, aux = load_obj('flame_model/assets/flame/FlameMesh.obj', load_textures=False)
            uv_coords = aux.verts_uvs
            uv_coords = uv_coords * 2 - 1
            uv_coords[:, 1] = -uv_coords[:, 1]
            self.register_buffer('uvcoords', uv_coords)
            self.register_buffer('uvfaces', faces.textures_idx)
            self.register_buffer('tri_faces', faces.verts_idx)

        # 3) flame_model 유사 객체에서 가져오기
        else:
            uv_coords = flame_model.verts_uvs
            uv_coords = uv_coords * 2 - 1
            uv_coords[:, 1] = -uv_coords[:, 1]
            self.register_buffer('uvcoords', uv_coords)
            self.register_buffer('uvfaces', flame_model.textures_idx)
            self.register_buffer('tri_faces', flame_model.faces)
            
    def forward(self, uv_maps, gaussian_points, binding, vertices):
        gaussian_points = gaussian_points[None]
        binding = binding[None]
        batch_size = gaussian_points.shape[0]
        N = gaussian_points.shape[1]

        # faces와 uvfaces를 batch_size에 맞게 확장
        faces = self.tri_faces.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, F, 3)
        uvfaces = self.uvfaces.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, F, 3)
        uvcoords = self.uvcoords.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, V_uv, 2)

        # UV 좌표 계산
        uv_points = self.get_uv_mapping(
            gaussian_points, faces, vertices, uvcoords, uvfaces, binding
        )  # (batch_size, N, 2)

        # grid_sample을 위한 grid 준비 (grid는 [-1, 1] 범위여야 함)
        grid = uv_points.unsqueeze(2)  # (batch_size, N, 1, 2)

        # 샘플링된 피처를 저장할 딕셔너리 초기화
        sampled_features_dict = {}

        for key, uv_map in uv_maps.items():
            # uv_map의 크기 확인
            batch_size_uv_map, channels, H, W = uv_map.shape
            assert batch_size_uv_map == batch_size, f"uv_map '{key}'의 batch_size가 gaussian_points와 일치하지 않습니다."

            # grid_sample을 사용하여 UV 포인트에서 피처 샘플링
            sampled_features = F.grid_sample(
                uv_map, grid, mode='bilinear', align_corners=True
            )  # (batch_size, channels, N, 1)

            sampled_features = sampled_features.squeeze(3)  # (batch_size, channels, N)

            # 샘플링된 피처를 딕셔너리에 저장
            sampled_features_dict[key] = sampled_features

        return sampled_features_dict

    def get_uv_mapping(self, gaussian_points, faces, vertices, uvcoords, uvfaces, binding):
        """
        Gaussian 포인트들의 UV 좌표를 계산합니다.

        Args:
            gaussian_points (Tensor): (batch_size, N, 3)
            faces (Tensor): (batch_size, F, 3)
            vertices (Tensor): (batch_size, V, 3)
            uvcoords (Tensor): (batch_size, V_uv, 2)
            uvfaces (Tensor): (batch_size, F, 3)
            binding (Tensor): (batch_size, N)

        Returns:
            uv_points (Tensor): (batch_size, N, 2)
        """
        batch_size, N, _ = gaussian_points.shape

        # 각 포인트가 바인딩된 face의 버텍스 인덱스 가져오기
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1)  # (batch_size, 1)
        face_vertex_indices = faces[batch_indices, binding]  # (batch_size, N, 3)

        # 해당 face의 버텍스 좌표 가져오기
        point_face_vertices = vertices[
            batch_indices.unsqueeze(2), face_vertex_indices
        ]  # (batch_size, N, 3, 3)

        # barycentric 좌표 계산
        bary_coords = self.compute_barycentric_coordinates(
            gaussian_points, point_face_vertices
        )  # (batch_size, N, 3)

        # 해당 face의 UV 버텍스 인덱스 가져오기
        face_uv_indices = uvfaces[batch_indices, binding]  # (batch_size, N, 3)

        # 해당 face의 UV 버텍스 좌표 가져오기
        point_face_uvs = uvcoords[
            batch_indices.unsqueeze(2), face_uv_indices
        ]  # (batch_size, N, 3, 2)

        # barycentric 좌표를 사용하여 UV 좌표 계산
        uv_points = (bary_coords.unsqueeze(-1) * point_face_uvs).sum(dim=2)  # (batch_size, N, 2)

        return uv_points

    def compute_barycentric_coordinates(self, points, face_vertices):
        """
        포인트의 face 내 barycentric 좌표를 계산합니다.

        Args:
            points (Tensor): (batch_size, N, 3)
            face_vertices (Tensor): (batch_size, N, 3, 3)

        Returns:
            bary_coords (Tensor): (batch_size, N, 3)
        """
        v0 = face_vertices[:, :, 1] - face_vertices[:, :, 0]  # (batch_size, N, 3)
        v1 = face_vertices[:, :, 2] - face_vertices[:, :, 0]
        v2 = points - face_vertices[:, :, 0]

        # Dot product 계산
        d00 = torch.sum(v0 * v0, dim=2)  # (batch_size, N)
        d01 = torch.sum(v0 * v1, dim=2)
        d11 = torch.sum(v1 * v1, dim=2)
        d20 = torch.sum(v2 * v0, dim=2)
        d21 = torch.sum(v2 * v1, dim=2)

        denom = d00 * d11 - d01 * d01 + 1e-8  # 분모에 작은 값 추가하여 zero division 방지

        v = (d11 * d20 - d01 * d21) / denom  # (batch_size, N)
        w = (d00 * d21 - d01 * d20) / denom
        u = 1 - v - w

        bary_coords = torch.stack([u, v, w], dim=2)  # (batch_size, N, 3)

        return bary_coords
    
    
    def unwrap_to_uv_rasterize(self, vertex_values, texture_resolution=256):
        H, W = texture_resolution, texture_resolution
        device = vertex_values.device

        # 1) UV → NDC (원본 보존)
        verts_uv = self.uvcoords.clone()     # ★ clone
        verts_uv[:, 0] = 1.0 - verts_uv[:, 0]   # 필요 시 V만 플립
        verts_uv = verts_uv * 2 - 1
        verts_uv = verts_uv.clamp_(-1.0, 1.0)

        # 2) UV 평면 래스터
        verts_uv_3d = torch.cat(
            [verts_uv, torch.full((verts_uv.shape[0], 1), 0.1, device=device)],
            dim=1
        )
        meshes = Meshes(verts=[verts_uv_3d], faces=[self.uvfaces.to(device).long()])
        cameras = OrthographicCameras(device=device)
        raster_settings = RasterizationSettings(
            image_size=texture_resolution,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=False,
            cull_backfaces=False,            # ★ 안정화
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        fragments = rasterizer(meshes)
        face_idx    = fragments.pix_to_face[0, ..., 0]       # (H,W)
        bary_coords = fragments.bary_coords[0, ..., 0, :]    # (H,W,3)

        # 유효 픽셀 마스크(정확하게)
        valid_mask = (face_idx >= 0) & (bary_coords.min(dim=-1).values >= 0)

        C = vertex_values.shape[1]
        uv_map = torch.zeros((H, W, C), device=device)

        if valid_mask.any():
            face_idx_v = face_idx[valid_mask].to(self.uvfaces.device)
            tri_faces  = self.tri_faces.to(face_idx_v.device).long()
            faces_mesh = tri_faces[face_idx_v]               # (n_valid,3)

            V = vertex_values.shape[0]
            bad = ((faces_mesh < 0) | (faces_mesh >= V)).any(dim=1)
            if bad.any():
                keep = ~bad
                faces_mesh = faces_mesh[keep]
                bc = bary_coords[valid_mask][keep]
                ij = valid_mask.nonzero(as_tuple=False)[keep]
            else:
                bc = bary_coords[valid_mask]
                ij = valid_mask.nonzero(as_tuple=False)

            bc = (bc.clamp(0,1) / bc.sum(dim=-1, keepdim=True).clamp_min(1e-8)).to(vertex_values.dtype)
            face_values = vertex_values.to(faces_mesh.device)[faces_mesh]   # (n_keep,3,C)
            interpolated = (face_values * bc.unsqueeze(-1)).sum(dim=1)      # (n_keep,C)
            uv_map[ij[:, 0], ij[:, 1]] = interpolated.to(uv_map.device)

        return uv_map, valid_mask



    def compute_barycentric_uv_batch(self, pixel_coords, uv_coords):
        """
        Barycentric 좌표를 배치로 계산합니다.
        Args:
            pixel_coords (Tensor): (N, 2) 픽셀 좌표
            uv_coords (Tensor): (3, 2) UV 삼각형 좌표
        Returns:
            bary_coords (Tensor): (N, 3) Barycentric 좌표
        """
        u0, v0 = uv_coords[0]
        u1, v1 = uv_coords[1]
        u2, v2 = uv_coords[2]

        det = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
        det = det + 1e-8  # Zero division 방지

        l1 = ((v1 - v2) * (pixel_coords[:, 0] - u2) + (u2 - u1) * (pixel_coords[:, 1] - v2)) / det
        l2 = ((v2 - v0) * (pixel_coords[:, 0] - u2) + (u0 - u2) * (pixel_coords[:, 1] - v2)) / det
        l3 = 1 - l1 - l2

        return torch.stack([l1, l2, l3], dim=-1)            
    
 
# Rasterize 안쓰는 normal uv 
def unwrap_normals_to_uv(self, vertex_normals, texture_resolution=128):
    """
    Vertex Normal을 UV Texture Map으로 Unwrap
    Args:
        vertex_normals (Tensor): (V, 3) Vertex 법선
        texture_resolution (int): UV Texture 해상도
    Returns:
        uv_normal_map (Tensor): (H, W, 3) UV Normal Map
    """
    H, W = texture_resolution, texture_resolution
    device = vertex_normals.device

    # 1. UV 좌표 정규화 및 픽셀 좌표로 변환
    verts_uv = (self.uvcoords + 1) / 2  # [-1, 1] → [0, 1]
    verts_uv *= torch.tensor([W - 1, H - 1], device=device)  # [0, 1] → [0, W/H]
    verts_uv = verts_uv.clamp(min=0, max=min(W-1, H-1))

    # 2. UV Faces를 통해 Vertex Normals 재매핑
    uv_vertex_normals = vertex_normals[self.tri_faces.flatten()].reshape(self.tri_faces.shape[0], 3, 3)
    face_uvs = verts_uv[self.uvfaces]  # (F, 3, 2)
    
    # 3. 모든 픽셀 좌표 생성
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    pixel_coords = torch.stack([x, y], dim=-1).float().reshape(-1, 2)  # (H*W, 2)
    
    # 4. 각 픽셀과 Triangle의 관계 확인 (Barycentric 좌표)
    uv_normal_map = torch.zeros((H * W, 3), device=device)
    weight_map = torch.zeros((H * W, 1), device=device)

    for i in range(face_uvs.shape[0]):
        uvs = face_uvs[i]  # (3, 2)
        normals = uv_vertex_normals[i]  # (3, 3)
        
        bary_coords = self.compute_barycentric_uv_batch(pixel_coords, uvs)  # (H*W, 3)
        valid = (bary_coords >= 0).all(dim=1)  # 유효한 Barycentric 좌표
        
        if valid.any():
            valid_coords = bary_coords[valid]
            valid_normals = (valid_coords.unsqueeze(-1) * normals).sum(dim=1)
            uv_normal_map[valid] += valid_normals
            weight_map[valid] += 1

    # 5. Weight로 정규화 및 UV 맵으로 리쉐이프
    weight_map = weight_map.clamp(min=1e-8)
    uv_normal_map /= weight_map
    uv_normal_map = uv_normal_map.reshape(H, W, 3)

    return uv_normal_map


# def compute_barycentric_uv_batch(self, pixel_coords, uv_coords):
#     """
#     Barycentric 좌표를 배치로 계산합니다.
#     Args:
#         pixel_coords (Tensor): (N, 2) 픽셀 좌표
#         uv_coords (Tensor): (3, 2) UV 삼각형 좌표
#     Returns:
#         bary_coords (Tensor): (N, 3) Barycentric 좌표
#     """
#     u0, v0 = uv_coords[0]
#     u1, v1 = uv_coords[1]
#     u2, v2 = uv_coords[2]

#     det = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
#     det = det + 1e-8  # Zero division 방지

#     l1 = ((v1 - v2) * (pixel_coords[:, 0] - u2) + (u2 - u1) * (pixel_coords[:, 1] - v2)) / det
#     l2 = ((v2 - v0) * (pixel_coords[:, 0] - u2) + (u0 - u2) * (pixel_coords[:, 1] - v2)) / det
#     l3 = 1 - l1 - l2

#     return torch.stack([l1, l2, l3], dim=-1)            
