from typing import List, Optional

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import imageio
import imageio.v3 as iio  # or use OpenCV, PIL, etc.
import pyexr
import torchvision.transforms as T
from torchvision.transforms.functional import to_tensor
from .renderutils import diffuse_cubemap, specular_cubemap
from libs.utils.graphics_utils import srgb_to_rgb, rgb_to_srgb
# from libs.utils.image_utils import read_hdr

def cube_to_dir(s: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if s == 0:
        rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1:
        rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2:
        rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3:
        rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4:
        rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5:
        rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)


class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap: torch.Tensor) -> torch.Tensor:
        # avg_pool_nhwc
        y = cubemap.permute(0, 3, 1, 2)  # NHWC -> NCHW
        y = torch.nn.functional.avg_pool2d(y, (2, 2))
        return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC

    @staticmethod
    def backward(ctx, dout: torch.Tensor) -> torch.Tensor:
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
        for s in range(6):
            gy, gx = torch.meshgrid(
                torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                indexing="ij",
            )
            v = F.normalize(cube_to_dir(s, gx, gy), p=2, dim=-1)
            out[s, ...] = dr.texture(
                dout[None, ...] * 0.25,
                v[None, ...].contiguous(),
                filter_mode="linear",
                boundary_mode="cube",
            )
        return out


class CubemapLight(nn.Module):
    # for nvdiffrec
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(
        self,
        base_res: int = 512,
        scale: float = 0.5,
        bias: float = 0.25,
    ) -> None:
        super(CubemapLight, self).__init__()
        self.mtx = None
        # base = (torch.rand(6, base_res, base_res, 3, dtype=torch.float32, device="cuda") * scale + bias)
        base = torch.ones((6,base_res, base_res, 3), dtype=torch.float32, device="cuda")*0.65
        self.base = nn.Parameter(base)
        self.register_parameter("env_base", self.base)

    def xfm(self, mtx) -> None:
        self.mtx = mtx

    def clamp_(self, min: Optional[float]=None, max: Optional[float]=None) -> None:
        with torch.no_grad():
            self.base.clamp_(min, max)
    
    def regularizer(self):
        white = (self.env_base[..., 0:1] + self.env_base[..., 1:2] + self.env_base[..., 2:3]) / 3.0
        return torch.mean(torch.abs(self.env_base - white))

    def set_lgt_name(self, name):
        self.lgt_name = name
    
    def get_lgt_name(self):
        return self.lgt_name
    
    def build_cube_light_maps(self, light_map_list, hdr_size=(512, 1024), train_lights=True):
        """
        light_map_list: 각 라이트 폴더 경로 리스트. 폴더 안에 right,left,top,bottom,front,back.hdr가 있다고 가정.
        hdr_size: (H, W)
        """
        self.light_map_list = light_map_list
        self.num_light_map = len(light_map_list)
        H, W = 256,256 # hdr_size

        # Global scaling for each env map. shape 유지: (3, num_light_map)
        global_scale = torch.ones(self.num_light_map, 3, dtype=torch.float32) * 0.5

        # Cubemap: (num_light_map, 6, 3, H, W)
        light_maps = torch.zeros(self.num_light_map, 6, H, W, 3,  dtype=torch.float32)

        face_order = ["right", "left", "top", "bottom", "front", "back"]
        for idx, lgt_path in enumerate(light_map_list):
            for i, face in enumerate(face_order):
                path = f"{lgt_path}/{face}.hdr"
                hdr_np = read_hdr(path)                     # (H?, W?, 3) numpy
                hdr = torch.as_tensor(hdr_np, dtype=torch.float32)

                # 기본 형상 점검
                if hdr.ndim != 3 or hdr.shape[-1] != 3:
                    raise ValueError(f"Invalid HDR shape at {path}. Expected (H, W, 3), got {tuple(hdr.shape)}")

                # 크기 보정이 필요하면 bilinear 보간
                if (hdr.shape[0], hdr.shape[1]) != (H, W):
                    hdr = hdr.unsqueeze(0)                 # (1, 3, H0, W0)
                    hdr = F.interpolate(hdr, size=(H, W), mode="bilinear", align_corners=False)
                    hdr = hdr.squeeze(0)                                    # (3, H, W)
                else:
                    hdr = hdr                             # (3, H, W)

                # 초기 스케일
                light_maps[idx, i] = hdr # * 0.2 - Global 의 initial을 0.2로 잡아서 보정

        s_min = 0.05
        self.register_buffer("global_scale_min", torch.tensor(s_min, dtype=torch.float32))
        self.register_buffer("global_scale_init", global_scale.clone())  # 초기값 기록(규제용)
        if train_lights:
            self.light_maps = nn.Parameter(light_maps, requires_grad=True)
            self.global_scale = nn.Parameter(global_scale, requires_grad=True)
        else:
            self.register_buffer("light_maps", light_maps)
            self.register_buffer("global_scale", global_scale)

        s_min = 0.05
        def _softplus_inv(y: torch.Tensor, eps: float = 1e-6):
            # y는 s_min 이상이어야 함
            y = (y - s_min).clamp_min(eps)
            return torch.log(torch.expm1(y))  # softplus^-1
        self.global_scale_raw = nn.Parameter(_softplus_inv(global_scale))  # (K, 3)

    @property
    def global_scale(self):
        # 항상 s_min 이상 양수
        return F.softplus(self.global_scale_raw) + self.global_scale_min
        


    def get_mip(self, roughness: torch.Tensor) -> torch.Tensor:
        return torch.where(
            roughness < self.MAX_ROUGHNESS,
            (torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS)
            / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS)
            * (len(self.specular) - 2),
            (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS)
            / (1.0 - self.MAX_ROUGHNESS)
            + len(self.specular)
            - 2,
        )
        
    def build_mips_white(self, cutoff: float = 0.99) -> None:
        self.specular = [self.base]
        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        # 강제 diffuse를 흰색으로
        self.diffuse = torch.ones_like(self.specular[-1])

        for idx in range(len(self.specular)):
            self.specular[idx] = torch.ones_like(self.specular[idx])

    # def build_mips(self, cutoff: float = 0.99) -> None:
    #     self.specular = [self.base] # torch.Size([6, 256, 256, 3])
    #     while self.specular[-1].shape[1] > self.LIGHT_MIN_RES: # 16
    #         self.specular += [cubemap_mip.apply(self.specular[-1])] # 6, [256 128 64 32 16], 3 

    #     self.diffuse = diffuse_cubemap(self.specular[-1]) # 6 16 16 3

    #     for idx in range(len(self.specular) - 1):
    #         roughness = (idx / (len(self.specular) - 2)) * (
    #             self.MAX_ROUGHNESS - self.MIN_ROUGHNESS
    #         ) + self.MIN_ROUGHNESS
    #         self.specular[idx] = specular_cubemap(self.specular[idx], roughness, cutoff)
    #     self.specular[-1] = specular_cubemap(self.specular[-1], 1.0, cutoff)


    def build_mips(self, light_map_path=None, cutoff: float = 0.99) -> None:

        if light_map_path is not None:
            light_idx = self.light_map_list.index(light_map_path)
            self.specular = [self.light_maps[light_idx] * self.global_scale[light_idx]] # 6, 3, W,H
        else:
            self.specular = [self.base]

        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES: # 16
            self.specular += [cubemap_mip.apply(self.specular[-1])] # 6, [256 128 64 32 16], 3 

        self.diffuse = diffuse_cubemap(self.specular[-1]) # 6 16 16 3

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (
                self.MAX_ROUGHNESS - self.MIN_ROUGHNESS
            ) + self.MIN_ROUGHNESS
            self.specular[idx] = specular_cubemap(self.specular[idx], roughness, cutoff)
        self.specular[-1] = specular_cubemap(self.specular[-1], 1.0, cutoff)


    def export_envmap(
        self,
        filename: Optional[str] = None,
        res: List[int] = [512, 1024],
        return_img: bool = False,
    ) -> Optional[torch.Tensor]:
        # cubemap_to_latlong
        gy, gx = torch.meshgrid(
            torch.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
            indexing="ij",
        )

        sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
        sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

        reflvec = torch.stack(
            (sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1
        )  # [H, W, 3]
        color = dr.texture(
            self.base[None, ...],
            reflvec[None, ...].contiguous(),
            filter_mode="linear",
            boundary_mode="cube",
        )[
            0
        ]  # [H, W, 3]
        if return_img:
            return color
        else:
            cv2.imwrite(filename, color.clamp(min=0.0).cpu().numpy()[..., ::-1])

    def training_setup(self, training_args, lr_env: float = 1e-3):
        self.optimizer = torch.optim.Adam(
            [{'params': [self.light_maps, self.global_scale_raw], 
              'lr': training_args["light_lr"], "name": "light"}], eps=1e-3)
            

class DirectLightMap(nn.Module):
    def __init__(self, H=64):
        super().__init__()
        self.H, self.W = H, H * 2
        self.base = torch.ones(1, 3, self.H, self.W)   # 항상 텐서(버퍼 역할)
        self.base_param = None                          # HDR 없을 때만 사용(학습 대상)
        self.use_lights = False
        self.lidx = 0
        self.s_min = 0.05

    def set_lidx(self, i: int):
        self.lidx = int(i)

    def set_optimizable_light(self, H=64, train_lights=True):

        self.base = torch.ones(1, 1, H, H*2)

        # 학습 변수: light_maps, scale_raw (둘만)
        self.light_maps_ = nn.Parameter(self.base, requires_grad=train_lights)     # [K,3,H,W]
        init_scale = torch.full((1, 3), 0.0, dtype=torch.float32) # [K,3]
        def softplus_inv(y, eps=1e-6):
            y = (y - self.s_min).clamp_min(eps)
            return torch.log(torch.expm1(y))
        self.scale_raw = nn.Parameter(softplus_inv(init_scale), requires_grad=train_lights)  # [K,3]

        self.use_lights = True
        self.base_param = None  # HDR 모드에서는 사용 안 함

    def set_lights_from_hdr(self, light_map_list, train_lights=True):
        # HDR 리스트가 없으면 base-only 모드
        if not light_map_list[0]:
            self.use_lights = False
            return

        self.light_map_list = list(light_map_list)
        self.num_light_map  = len(self.light_map_list)

        # 첫 이미지 해상도에 맞춤
        first = iio.imread(self.light_map_list[0])      # (H0, W0, 3)
        H0, W0 = int(first.shape[0]), int(first.shape[1])
        if (H0, W0) != (self.H, self.W):
            self.base = torch.ones(1, 3, H0, W0)

        # HDR 적재 및 리사이즈 → [K,3,H,W]
        light_maps = torch.empty(self.num_light_map, 3, self.H, self.W, dtype=torch.float32)
        for k, p in enumerate(self.light_map_list):
            img = iio.imread(p)                  # (h,w,3)
            t = to_tensor(img).float()           # (3,h,w)
            if t.shape[1:] != (self.H, self.W):
                t = F.interpolate(t.unsqueeze(0), (self.H, self.W), mode="bilinear", align_corners=False).squeeze(0)
            light_maps[k] = t

        # 학습 변수: light_maps, scale_raw (둘만)
        self.light_maps_ = nn.Parameter(light_maps[:,0].view(-1,1,self.H, self.W)*0, requires_grad=train_lights)     # [K,3,H,W]
        init_scale = torch.full((self.num_light_map, 3), 0.0, dtype=torch.float32) # [K,3]
        def softplus_inv(y, eps=1e-6):
            y = (y - self.s_min).clamp_min(eps)
            return torch.log(torch.expm1(y))
        self.scale_raw = nn.Parameter(softplus_inv(init_scale), requires_grad=train_lights)  # [K,3]

        self.use_lights = True
        self.base_param = None  # HDR 모드에서는 사용 안 함

    def training_setup(self, cfg_training):
        lr_env = cfg_training['light_lr']
        if self.use_lights:
            params = [{'params': [self.light_maps_, self.scale_raw], 'lr': lr_env}]
        else:
            if self.base_param is None:
                self.base_param = nn.Parameter(self.base.clone(), requires_grad=True).to('cuda')  # [1,3,H,W]
            params = [{'params': [self.base_param], 'lr': lr_env}]
        self.optimizer = torch.optim.Adam(params, eps=1e-8)

    def _scale(self, idx: int):                 # [3]
        return self.s_min + F.softplus(self.scale_raw[idx])
    
    def _base(self, idx: int):                 # [3]
        light = F.softplus(self.light_maps_[idx].repeat(3,1,1))
        return light / (1.0 + light)

    def get_base_from_lights(self, light_map_path: str | None = None):
        """
        매 iteration 시작 전에 호출. 반환 없음. self.base만 갱신.
        그래프 유지. device는 메인에서 model.to(device)로 일괄 이동.
        """
        if light_map_path is None:
            env   = self._base(0)
            self.base = env.unsqueeze(0)
            return
        
        if not self.use_lights:
            if self.base_param is not None:
                self.base = self.base_param      # 그래프 유지
            return

        idx = self.lidx if light_map_path is None else self.light_map_list.index(light_map_path)
        # scale = self._scale(idx).view(3, 1, 1)   # [3,1,1]
        env   = self._base(idx) # * scale     # [3,H,W]  (둘 다 학습 변수)
        self.base = env.unsqueeze(0)             # [1,3,H,W]  (참조 대입. to() 안 함)


    @property
    def get_parms(self):
        parms_g = {'base': self.env_base,}
        outs = {}
        outs['gaussian'] = parms_g
        return outs
    
    def regularizer(self):
        white = torch.mean(self.env_base, 1, keepdim=True)
        return torch.mean(torch.abs(self.env_base - white))
    
    def clamp_(self, min: Optional[float]=None, max: Optional[float]=None) -> None:
        self.env_base.clamp_(min, max)

    def set_yaw(self, deg: float):
        """환경맵의 수평 방향 오프셋을 degree 단위로 지정."""
        self.yaw_rad = math.radians(deg)

    def debug_direct_light(envir_map):
        test_dirs = {
            "x+": torch.tensor([[1.0, 0.0, 0.0]]),
            "y+": torch.tensor([[0.0, 1.0, 0.0]]),
            "z+": torch.tensor([[0.0, 0.0, 1.0]])
        }

        H, W = envir_map.shape[-2], envir_map.shape[-1]

        for name, dirs in test_dirs.items():
            # 동일한 정의로 통일
            dirs = torch.nn.functional.normalize(dirs, dim=-1)
            theta = torch.atan2(dirs[:, 2], dirs[:, 0])                               # atan2(z, x)
            phi   = torch.atan2(torch.norm(dirs[:, (0, 2)], dim=-1), dirs[:, 1])      # atan2(sqrt(x^2+z^2), y)

            # [-1, 1]로 정규화. y는 부호 뒤집지 않음
            query_x = theta / np.pi
            query_y = (phi / np.pi) * 2.0 - 1.0

            grid = torch.stack((query_x, query_y), dim=-1).unsqueeze(0).unsqueeze(2)
            sampled = torch.nn.functional.grid_sample(
                envir_map, grid, mode="bilinear", align_corners=True, padding_mode="border"
            )  # [1, 3, N, 1]
            light_rgbs = sampled[0, :, :, 0].T  # [N, 3]

            # 픽셀 인덱스 확인
            col = ((query_x + 1) * 0.5) * (W - 1)
            row = ((query_y + 1) * 0.5) * (H - 1)

            print(f"\n--- {name} ---")
            print(f"dirs = {dirs.numpy()}")
            print(f"phi (deg)   = {(phi.item() * 180/np.pi):.2f}")
            print(f"theta (deg) = {(theta.item() * 180/np.pi):.2f}")
            print(f"grid coords = {grid.squeeze(0).squeeze(1).numpy()}  # (x,y in [-1,1])")
            print(f"sample col,row = [{int(round(col.item()))}] [{int(round(row.item()))}]")
            print(f"sampled RGB = {light_rgbs.detach().cpu().numpy()}")

    def direct_light(self, dirs, transform=None):

        envir_map =  self.base # [1, 3, H, W]

        shape = dirs.shape
        dirs = F.normalize(dirs.reshape(-1, 3), dim=-1)

        theta = torch.atan2(dirs[:, 0], dirs[:, 2])
        phi   = torch.acos(dirs[:, 1].clamp(-1.0 + 1e-6, 1.0 - 1e-6))

        query_x = - theta / np.pi  
        query_y = (phi / np.pi) * 2.0 - 1.0  

        grid = torch.stack((query_x, query_y), dim=-1).view(1, -1, 1, 2)   # [1, N, 1, 2]
        sampled = F.grid_sample(envir_map, grid, mode="bilinear",
                                align_corners=True, padding_mode="border")  # [1, 3, N, 1]
        light_rgbs = sampled[0, :, :, 0].T                                  # [N, 3]

        return light_rgbs.reshape(*shape)
    

    def direct_light_vanilla(self, dirs, transform=None):
        shape = dirs.shape
        dirs = dirs.reshape(-1, 3)
        envir_map = self.base # [1, 3, H, W]

        phi = torch.arccos(dirs[:, 2]).reshape(-1) - 1e-6
        theta = torch.atan2(dirs[:, 1], dirs[:, 0]).reshape(-1)
        query_y = (phi / np.pi) * 2 - 1
        query_x = - theta / np.pi
        grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0)
        light_rgbs = F.grid_sample(envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)
        return light_rgbs.reshape(*shape)


    @property
    def get_env(self):
        return self.env_base
    
    @property
    def diffuse(self):
        return self.get_env  # [1, 3, H, W]
    
    def build_mipmaps(self, num_levels=5):
        """
        Builds a list of downsampled versions of the environment map.
        Each level is blurred more (simulating rougher surfaces).
        """
        mips = [self.get_env]  # Level 0 (sharp)
        for i in range(1, num_levels):
            mips.append(F.avg_pool2d(mips[-1], kernel_size=2, stride=2))
        return mips
    
    @property
    def specular(self):
        return self.build_mipmaps()

class EnvLight(torch.nn.Module):
    def __init__(self, path=None, scale=1.0):
        super().__init__()
        self.device = "cuda"  # only supports cuda
        self.scale = scale  # scale of the hdr values
        self.to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device="cuda")

        self.envmap = self.load(path, scale=self.scale, device=self.device)
        self.transform = None

    @staticmethod
    def load(envmap_path, scale, device):
        if not envmap_path.endswith(".exr"):
            image = srgb_to_rgb(imageio.imread(envmap_path)[:, :, :3] / 255)
        else:
            # load latlong env map from file
            image = pyexr.open(envmap_path).get()[:, :, :3]

        image = image * scale

        env_map_torch = torch.tensor(image, dtype=torch.float32, device=device, requires_grad=False)

        return env_map_torch

    def direct_light(self, dirs, transform=None):
        shape = dirs.shape
        dirs = dirs.reshape(-1, 3)

        if transform is not None:
            dirs = dirs @ transform.T
        elif self.transform is not None:
            dirs = dirs @ self.transform.T

        envir_map =  self.envmap.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
        phi = torch.arccos(dirs[:, 2]).reshape(-1) - 1e-6
        theta = torch.atan2(dirs[:, 1], dirs[:, 0]).reshape(-1)
        # normalize to [-1, 1]
        query_y = (phi / np.pi) * 2 - 1
        query_x = - theta / np.pi
        grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0)
        light_rgbs = F.grid_sample(envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)
    
        return light_rgbs.reshape(*shape)




def _rotmat_yaw_pitch_roll(yaw_deg, pitch_deg, roll_deg, device, dtype):
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    R_y = torch.tensor([[ cy, 0.0, sy],
                        [ 0.0, 1.0, 0.0],
                        [-sy, 0.0, cy]], device=device, dtype=dtype)
    R_x = torch.tensor([[1.0, 0.0, 0.0],
                        [0.0, cp, -sp],
                        [0.0, sp,  cp]], device=device, dtype=dtype)
    R_z = torch.tensor([[ cr, -sr, 0.0],
                        [ sr,  cr, 0.0],
                        [0.0,  0.0, 1.0]], device=device, dtype=dtype)
    return R_y @ R_x @ R_z  # [3,3]

def rotate_env_for_direct_light(env, yaw_deg=0.0, pitch_deg=90.0, roll_deg=-90.0):
    """
    env: [1,3,H,W], linear RGB.
    direct_light 규약: theta=atan2(y,x), phi=acos(z),
                       u=-theta/pi, v=(phi/pi)*2-1.
    """
    assert env.dim() == 4 and env.shape[0] == 1
    device, dtype = env.device, env.dtype
    _, _, H, W = env.shape

    # 타깃 그리드 (u,v in [-1,1])
    u = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    v = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(v, u, indexing='ij')  # [H,W]

    # direct_light의 역매핑으로 타깃 방향 dir_t 생성
    theta_t = -uu * math.pi
    phi_t   = ((vv + 1.0) * 0.5) * math.pi
    sin_phi = torch.sin(phi_t)
    dir_x = sin_phi * torch.cos(theta_t)
    dir_y = sin_phi * torch.sin(theta_t)
    dir_z = torch.cos(phi_t)
    dir_t = torch.stack([dir_x, dir_y, dir_z], dim=-1).view(-1, 3)  # [HW,3]

    # 소스 방향: dir_s = R^{-1} dir_t. 회전행렬은 직교라 R^{-1}=R^T.
    R = _rotmat_yaw_pitch_roll(yaw_deg, pitch_deg, roll_deg, device, dtype)  # [3,3]
    dir_s = dir_t @ R  # row-vector 관례로 이렇게 쓰면 R^T가 적용된 효과

    # dir_s → (u_s, v_s) with direct_light 규약
    dx, dy, dz = dir_s[:, 0], dir_s[:, 1], dir_s[:, 2]
    theta_s = torch.atan2(dy, dx)
    phi_s   = torch.acos(dz.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    u_s = - theta_s / math.pi
    v_s = (phi_s / math.pi) * 2.0 - 1.0

    grid = torch.stack([u_s, v_s], dim=-1).view(1, H, W, 2)  # [1,H,W,2], (x,y)
    rotated = F.grid_sample(env, grid, mode='bilinear',
                            padding_mode='border', align_corners=True)
    return rotated  # [1,3,H,W]