# libs/pbr/shlight.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 2차 SH 기저. 순서: [1, y, z, x, xy, yz, 3z^2-1, xz, x^2-y^2]
def sh_basis_2nd(n: torch.Tensor) -> torch.Tensor:
    x, y, z = n[..., 0:1], n[..., 1:2], n[..., 2:3]
    B0 = 0.282095 * torch.ones_like(x)
    B1 = 0.488603 * y
    B2 = 0.488603 * z
    B3 = 0.488603 * x
    B4 = 1.092548 * x * y
    B5 = 1.092548 * y * z
    B6 = 0.315392 * (3.0 * z * z - 1.0)
    B7 = 1.092548 * x * z
    B8 = 0.546274 * (x * x - y * y)
    return torch.cat([B0,B1,B2,B3,B4,B5,B6,B7,B8], dim=-1)

# 3차 SH 기저. 순서: [1, y, z, x, xy, yz, 3z^2-1, xz, x^2-y^2,
#                     y(3x^2-y^2), xyz, y(5z^2-1), z(5z^2-3), x(5z^2-1), (x^2-y^2)z, x(x^2-3y^2)]
def sh_basis_3rd(n: torch.Tensor) -> torch.Tensor:
    x, y, z = n[..., 0:1], n[..., 1:2], n[..., 2:3]
    # l=0..2 (기존 9개)
    B0 = 0.282095 * torch.ones_like(x)
    B1 = 0.488603 * y
    B2 = 0.488603 * z
    B3 = 0.488603 * x
    B4 = 1.092548 * x * y
    B5 = 1.092548 * y * z
    B6 = 0.315392 * (3.0 * z * z - 1.0)
    B7 = 1.092548 * x * z
    B8 = 0.546274 * (x * x - y * y)
    # l=3 (추가 7개)
    B9  = 0.590044 * y * (3.0 * x * x - y * y)
    B10 = 2.890611 * x * y * z
    B11 = 0.457046 * y * (5.0 * z * z - 1.0)
    B12 = 0.373176 * z * (5.0 * z * z - 3.0)
    B13 = 0.457046 * x * (5.0 * z * z - 1.0)
    B14 = 1.445306 * (x * x - y * y) * z
    B15 = 0.590044 * x * (x * x - 3.0 * y * y)
    return torch.cat([B0,B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12,B13,B14,B15], dim=-1)

class SHLight(nn.Module):
    """
    Global, grayscale 3rd-order SH lighting.
    L(n) = sum_i c_i Y_i(n), i=0..15. 음수 방지를 위해 ReLU.
    """
    def __init__(self, init_coeff: float = 0.0, device: str = "cuda"):
        super().__init__()
        # Initialize coefficients with small random values around 0.1
        c = 0.1 + 0.01 * torch.randn(16, device=device)
        self.coeffs = nn.Parameter(c, requires_grad=True)

    @staticmethod
    def _normalize_normals(n: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return n / n.norm(dim=-1, keepdim=True).clamp_min(eps)

    def forward(self, normals: torch.Tensor) -> torch.Tensor:
        """
        normals: (N,3) 또는 (B,N,3). 반환: (...,1) irradiance ≥ 0
        """
        orig = normals.shape[:-1]
        n = self._normalize_normals(normals.reshape(-1, 3))
        B = sh_basis_3rd(n)                        # (M,16)
        L = (B * self.coeffs.view(1, -1)).sum(-1, keepdim=True)  # (M,1)
        return F.softplus(L, beta=5.0).reshape(*orig, 1)

    def shade_rgb(self, normals: torch.Tensor, base_gray) -> torch.Tensor:
        L = self.forward(normals)[..., 0]                 # (...,)
        rgb = (base_gray * L).unsqueeze(-1).repeat(1, 3)  # (N,3) contiguous
        return rgb.contiguous()

    def shade_with_albedo(self, normals: torch.Tensor, albedo_rgb: torch.Tensor) -> torch.Tensor:
        """
        Lambertian shading with per-Gaussian albedo.
        normals: (N,3) or (B,N,3)
        albedo_rgb: (N,3) or (B,N,3) in [0,1]. If (...,1) it will be broadcast to 3 channels.
        returns: same shape as albedo_rgb, contiguous
        """
        L = self.forward(normals)[..., 0]  # (...,)
        # Ensure albedo has 3 channels
        if albedo_rgb.shape[-1] == 1:
            albedo_rgb = albedo_rgb.repeat_interleave(3, dim=-1)
        shaded = albedo_rgb * L.unsqueeze(-1)
        return shaded.contiguous()
    
    def training_setup(self, cfg_train: dict):
        self.optimizer = torch.optim.Adam(
        [{'params': [self.coeffs], 'lr': 1e-2}],
        eps=1e-8
        )

    def irradiance(self, normals: torch.Tensor) -> torch.Tensor:
        return self.forward(normals)[..., 0]