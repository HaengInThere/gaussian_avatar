
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


from libs.utils.graphics_utils import quat_rotate, rotate_canon_to_normal
# ----------------------------
# GroupNorm helper
# ----------------------------
def _gn(num_channels: int, max_groups: int = 32):
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g //= 2
    return nn.GroupNorm(g, num_channels)

def _act(name: str):
    name = str(name).lower()
    return nn.ReLU(inplace=True) if name == "relu" else nn.GELU()

# ----------------------------
# Positional channel grid helper
# ----------------------------
def _make_pos_ch(H: int, W: int) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, H)
    xs = torch.linspace(-1.0, 1.0, W)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    ones = torch.ones_like(xx)
    pos = torch.stack([xx, yy, ones], dim=0).unsqueeze(0)  # (1,3,H,W)
    return pos

# ----------------------------
# Quaternion utilities (channel-first maps)
# ----------------------------
def _qnormalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # q: (B,4,H,W)
    n = torch.linalg.norm(q, dim=1, keepdim=True).clamp_min(eps)
    return q / n

def _qmul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product for per-pixel quaternions in (w,x,y,z) with shape (B,4,H,W).
    Returns (B,4,H,W).
    """
    w1, x1, y1, z1 = q1[:,0], q1[:,1], q1[:,2], q1[:,3]
    w2, x2, y2, z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=1)



# ----------------------------
# Lightweight attention & head stacks
# ----------------------------
class SEBlock(nn.Module):
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, ch // reduction)
        self.fc1 = nn.Conv2d(ch, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, ch, kernel_size=1, bias=True)
        self.act = _act("relu")
        self.gate = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool2d(x, 1)         # (B,C,1,1)
        w = self.fc2(self.act(self.fc1(w)))     # (B,C,1,1)
        w = self.gate(w)
        return x * w

class HeadStack(nn.Module):
    """
    Small per-head predictor:
      [Conv3x3 + GN + Act] x (depth-1)  +  Conv3x3(out)
    Optionally includes SE after each hidden conv.
    """
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None,
                 depth: int = 3, act: str = "gelu", use_se: bool = True):
        super().__init__()
        assert depth >= 1
        self.use_se = use_se
        mid_ch = in_ch if mid_ch is None else mid_ch

        blocks = []
        c_in = in_ch
        for i in range(max(0, depth - 1)):
            blocks += [nn.Conv2d(c_in, mid_ch, 3, padding=1), _gn(mid_ch), _act(act)]
            if use_se:
                blocks += [SEBlock(mid_ch)]
            c_in = mid_ch
        self.trunk = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.final = nn.Conv2d(c_in, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        y = self.final(h)
        return y

# ----------------------------
# Building blocks
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, act: str = "relu"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            _gn(out_ch), _act(act),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            _gn(out_ch), _act(act),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class UNet2Down(nn.Module):
    """
    Minimal U-Net encoder-decoder with two downsamples. Returns two skip connections.
    """
    def __init__(self, in_ch: int, base: int = 64, act: str = "relu"):
        super().__init__()
        c1, c2, c3 = base, base*2, base*4
        self.enc1 = ConvBlock(in_ch, c1, act=act)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(c1, c2, act=act)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c2, c3, act=act)

        self.up1 = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(c2 + c2, c2, 3, padding=1), _gn(c2), _act(act),
            nn.Conv2d(c2, c2, 3, padding=1)
        )
        self.up2 = nn.ConvTranspose2d(c2, c1, 2, 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(c1 + c1, c1, 3, padding=1), _gn(c1), _act(act),
            nn.Conv2d(c1, c1, 3, padding=1)
        )

        self.c1, self.c2, self.c3 = c1, c2, c3

    def forward(self, x: torch.Tensor):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        bn = self.bottleneck(p2)
        u1 = self.up1(bn)
        d1 = self.dec1(torch.cat([u1, e2], dim=1))
        u2 = self.up2(d1)
        d2 = self.dec2(torch.cat([u2, e1], dim=1))
        return bn, e2, e1, d2  # bottleneck, skip2, skip1, last_feat (c1)

# ----------------------------
# Main model (always U-Net, no FiLM)
# ----------------------------
class UVPredictorUNet(nn.Module):
    """
    Geometry net (pos+FLAME-cond) → delta-geometry maps on UV.
    Appearance net (geo-maps+FLAME-cond) → color/opacity/spec-residual maps.
    Canonical UV-geometry from head_model can be attached via `attach_canonical(...)`
    and will be composed with the predicted deltas when forming appearance inputs.
    """

    def __init__(self, args: Dict, uv_size: int, device: str = "cuda"):
        super().__init__()
        self.args = args
        self.uv_size = uv_size
        self.device = device

        # buffers
        self.register_buffer("pos_ch", _make_pos_ch(uv_size, uv_size))

        # config (kept minimal)
        mcfg = self.args.setdefault("model", {})
        self.act = str(mcfg.get("act", "relu"))
        self.base_ch_geo = int(mcfg.get("mid_ch_geo", 64))
        self.base_ch_app = int(mcfg.get("mid_ch_app", 64))
        self.cond_dim    = int(mcfg.get("flame_cond_dim", 118))
        self.cond_embed_ch = int(mcfg.get("cond_embed_ch", 16))

        # learnable ranges
        self.dpos_scale        = nn.Parameter(torch.tensor(float(mcfg.get("dpos_scale_init", 0.001))))
        self.scale_log_range   = nn.Parameter(torch.tensor(float(mcfg.get("scale_log_range_init", 2.0))))
        self.dnormal_scale     = nn.Parameter(torch.tensor(float(mcfg.get("dnormal_scale_init", 1.0))))
        self.dnormal_spec_scale= nn.Parameter(torch.tensor(float(mcfg.get("dnormal_spec_scale_init", 0.5))))

        # conditioning: FLAME → embedding map (concat)
        self.cond_geo_mlp = nn.Sequential(
            nn.Linear(self.cond_dim, self.base_ch_geo), _act(self.act),
            nn.Linear(self.base_ch_geo, self.cond_embed_ch),
        )
        self.cond_app_mlp = nn.Sequential(
            nn.Linear(self.cond_dim, self.base_ch_app), _act(self.act),
            nn.Linear(self.base_ch_app, self.cond_embed_ch),
        )
        self.cond_normal_mlp = nn.Sequential(
            nn.Linear(self.cond_dim, self.base_ch_app), _act(self.act),
            nn.Linear(self.base_ch_app, self.cond_embed_ch),
        )

        self.cond_view_mlp = nn.Sequential(
            nn.Linear(self.cond_dim, self.base_ch_app), _act(self.act),
            nn.Linear(self.base_ch_app, self.cond_embed_ch),
        )

        self.cond_light_mlp = nn.Sequential(
            nn.Linear(self.cond_dim, self.base_ch_app), _act(self.act),
            nn.Linear(self.base_ch_app, self.cond_embed_ch),
        )

        # Environment light-map encoder (expects 3xH* x W* 360 image)
        self.env_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), _gn(32), _act(self.act),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), _gn(64), _act(self.act),
            nn.AdaptiveAvgPool2d(1)  # -> (B,64,1,1)
        )
        self.env_to_app = nn.Linear(64, self.cond_embed_ch)

        # --- Geometry U-Net (input: pos_ch(3)+cond_embed) ---
        geo_in = 3 + self.cond_embed_ch
        self.unet_geo = UNet2Down(geo_in, base=self.base_ch_geo, act=self.act)
        c1g = self.unet_geo.c1  # width of last_feat

        hd = int(mcfg.get("head_depth_geo", 3))
        self.head_dpos         = HeadStack(c1g, 3, mid_ch=c1g, depth=hd, act=self.act, use_se=bool(mcfg.get("head_use_se", True)))
        self.head_scale        = HeadStack(c1g, 3, mid_ch=c1g, depth=hd, act=self.act, use_se=bool(mcfg.get("head_use_se", True)))
        self.head_rot          = HeadStack(c1g, 4, mid_ch=c1g, depth=hd, act=self.act, use_se=bool(mcfg.get("head_use_se", True)))
        self.head_dnormal_diff = HeadStack(c1g, 3, mid_ch=c1g, depth=hd, act=self.act, use_se=bool(mcfg.get("head_use_se", True)))

        # --- Appearance U-Net (input: geo_feat(c1g) + cam_dir(3) + cond_embed) ---
        self.app_in_ch = 3 + 3 + 4 + 32 + 3 + 7 + self.cond_embed_ch  # + env embedding

        self.unet_app = UNet2Down(self.app_in_ch, base=self.base_ch_app, act=self.act)
        c1a = self.unet_app.c1
        self.unet_normal = UNet2Down(self.app_in_ch, base=self.base_ch_app, act=self.act)
        c1normal = self.unet_normal.c1

        hd_app = int(mcfg.get("head_depth_app", 3))
        hd_normal = int(mcfg.get("head_depth_normal", 3))
        use_se = bool(mcfg.get("head_use_se", True))
        self.head_color      = HeadStack(c1a, 3, mid_ch=c1a, depth=hd_app, act=self.act, use_se=use_se)
        self.head_opacity      = HeadStack(c1a, 1, mid_ch=c1a, depth=hd_app, act=self.act, use_se=use_se)
        
        self.head_dnormal_spec = HeadStack(c1normal, 3, mid_ch=c1normal, depth=hd_normal, act=self.act, use_se=use_se)
        self.head_spec_scale   = HeadStack(c1normal, 1, mid_ch=c1normal, depth=hd_normal, act=self.act, use_se=use_se)

        # --- View-only visibility branch (cam + geom + cond) ---
        # Inputs: pos(3) + slog(3) + rot(4) + geo_feat(32) + cam_dir(3) + cam_feats(7) + cond(emb)
        self.view_in_ch = 3 + 3 + 4 + 32 + 3 + 7 + self.cond_embed_ch
        self.unet_view = UNet2Down(self.view_in_ch, base=self.base_ch_app, act=self.act)
        c1view = self.unet_view.c1
        self.head_view_vis = HeadStack(c1view, 1, mid_ch=c1view, depth=4, act=self.act, use_se=True)

        # --- Light-only shadow branch (env + geom + cond), NO cam ---
        # Inputs: pos(3) + slog(3) + rot(4) + geo_feat(32) + env_emb(emb) + cond(emb)
        self.light_in_ch = 3 + 3 + 4 + 32 + self.cond_embed_ch + self.cond_embed_ch
        self.unet_light = UNet2Down(self.light_in_ch, base=self.base_ch_app, act=self.act)
        c1light = self.unet_light.c1
        self.head_light_shading_d = HeadStack(c1light, 1, mid_ch=c1light, depth=4, act=self.act, use_se=True)
        self.head_light_shading_s = HeadStack(c1light*2, 1, mid_ch=c1light, depth=4, act=self.act, use_se=True)

        # safe init for heads
        with torch.no_grad():
            # geometry heads start near no-change
            self.head_dpos.final.weight.zero_()
            if self.head_dpos.final.bias is not None: self.head_dpos.final.bias.zero_()
            self.head_scale.final.weight.zero_()
            if self.head_scale.final.bias is not None: self.head_scale.final.bias.zero_()
            # self.head_dnormal_diff.final.weight.zero_()
            # if self.head_dnormal_diff.final.bias is not None: self.head_dnormal_diff.final.bias.zero_()

            # rotation head → identity quaternion (w=1,x=y=z=0)
            self.head_rot.final.weight.zero_()
            if self.head_rot.final.bias is not None:
                self.head_rot.final.bias.zero_()
                self.head_rot.final.bias.data[0] = 1.0  # w channel

            # appearance heads start neutral (color/opac logits ≈ 0, spec residual 0)
            self.head_opacity.final.weight.zero_()
            if self.head_opacity.final.bias is not None: self.head_opacity.final.bias.zero_()
            self.head_dnormal_spec.final.weight.zero_()
            if self.head_dnormal_spec.final.bias is not None: self.head_dnormal_spec.final.bias.zero_()
            self.head_spec_scale.final.weight.zero_()
            if self.head_spec_scale.final.bias is not None: self.head_spec_scale.final.bias.zero_()
            # self.head_view_vis.final.weight.zero_()
            # if self.head_view_vis.final.bias is not None: self.head_view_vis.final.bias.zero_()

            # optional: start shading near zero
            # self.head_light_shading.final.weight.zero_()
            # if self.head_light_shading.final.bias is not None: self.head_light_shading.final.bias.zero_()

        # optimizer holder (set in training_setup)
        self.optimizer = None
        self.scheduler = None


    def forward_gaussian_dgeo(self, flame_cond: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Input:
          flame_cond: (B, cond_dim)
        Output UV maps (B,C,H,W):
          dpos, scale_log, rot, dnormal_diffuse
        """
        B = flame_cond.shape[0]
        H = W = self.uv_size

        pos = self.pos_ch.expand(B, -1, -1, -1)
        cond_geo = self.cond_geo_mlp(flame_cond).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        g_in = torch.cat([pos, cond_geo], dim=1)

        _, _, _, g_last = self.unet_geo(g_in)

        dpos  = self.head_dpos(g_last) * self.dpos_scale
        dslog  = self.head_scale(g_last) * self.scale_log_range
        drot   = _qnormalize(self.head_rot(g_last))
        dnormal= torch.tanh(self.head_dnormal_diff(g_last)) * self.dnormal_scale

        return {"dpos": dpos, "dslog": dslog, "drot": drot, "dnormal": dnormal, 'geo_feat':g_last}

        # return {"dpos": dpos, "dslog": dslog, "drot": drot, "dnormal": dnormal}

    def _compose_geo_for_app(self, geom, dgeom):

        B, _, H, W = dgeom["dpos"].shape
        canon_xyz  = geom["canon_xyz"].unsqueeze(0).expand(B, -1, -1, -1)            # (B,3,H,W)
        canon_slog = geom["canon_slog"].unsqueeze(0).expand(B, -1, -1, -1)          # (B,3,H,W)
        canon_rot  = _qnormalize(geom["canon_rot"]).unsqueeze(0).expand(B, -1, -1, -1)  # (B,4,H,W)

        dpos    = dgeom["dpos"]
        dslog   = dgeom["dslog"]
        drot    = _qnormalize(dgeom["drot"])
        dnormal = dgeom["dnormal"]

        # ===== build_gaussian 와 동일한 기준으로 맞추기 =====
        # base_pos: (1,N,3) → (1,3,H,W) → (B,3,H,W)
        uv_base_pos = geom["uv_base_pos"].expand(B, -1, -1, -1)
        uv_base_normal = geom["uv_base_normal"].expand(B, -1, -1, -1)

        # Position
        canon_pos_rotated = rotate_canon_to_normal(uv_base_normal, canon_xyz)
        dpos_base_rotated = rotate_canon_to_normal(uv_base_normal, dpos)
        pos  = uv_base_pos + canon_pos_rotated + dpos_base_rotated   # ★ 
        # Scale (log-domain)
        slog = canon_slog + dslog                                     # ★ 동일

        # Rotation
        rot = _qmul(drot, canon_rot)                                 # ★ 동일
        rot = _qnormalize(rot)
        # Normal: base_normal (+ dnormal)
        base_normal_uv = uv_base_normal.expand(B, -1, -1, -1)
        if dnormal is not None:
            normal = F.normalize(base_normal_uv + dnormal, dim=1, eps=1e-6)
        else:
            normal = F.normalize(base_normal_uv, dim=1, eps=1e-6)

        return {
            "pos": pos,
            "slog": slog,
            "rot":  rot,
            "normal": normal,
            "geo_feat": dgeom["geo_feat"],
        }

    def forward_gaussian_app(
        self,
        flame_cond: torch.Tensor,
        geometry: Dict[str, torch.Tensor],
        cam_pos: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        B = flame_cond.shape[0]
        H = W = self.uv_size

        # 1) 모양 정리 후 배치 확장
        if cam_pos.dim() == 2:         # (B,3)
            cam_pos = cam_pos.view(B, 3, 1, 1)
        elif cam_pos.dim() == 4:       # (1,3,1,1)
            cam_pos = cam_pos.expand(B, -1, -1, -1)
        else:
            raise ValueError("cam_pos shape should be (B,3) or (1,3,1,1).")

        # Make cam features
        cam_pos = cam_pos.float().detach()  
        cam_dir = F.normalize(cam_pos, dim=1, eps=1e-6)        # (B,3,1,1)
        cam_dir = cam_dir.expand(B, 3, H, W)                   # (B,3,H,W)

        Vv = cam_pos - geometry['pos']
        dist = Vv.norm(dim=1, keepdim=True).clamp_min(1e-6) # * 
        V_unit = Vv / dist # *

        q = _qnormalize(geometry['rot'].permute(0,2,3,1).reshape(-1,4))
        z_axis = torch.tensor([0, 0, 1.0], device=q.device, dtype=q.dtype).expand(q.shape[0],3)
        n = quat_rotate(q, z_axis).view(1,H,W,3).permute(0,3,1,2)

        nv_cos = (n * V_unit)  # *
        cam_feats = torch.cat([V_unit, dist, nv_cos], dim=1)

        feats = torch.cat([geometry['pos'],  # 3
                           geometry['slog'],  # 3
                           geometry['rot'],  # 4
                           geometry['geo_feat'], # 32
                           cam_dir,  # 3
                           cam_feats # 7
                           ], dim=1)  # 3 + 3 + 4 + 32 + 3 + 7
        
        cond_app = self.cond_app_mlp(flame_cond).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        a_in = torch.cat([feats, cond_app], dim=1)
        _, _, _, a_last = self.unet_app(a_in)

        features_dc_logit = self.head_color(a_last)
        opacity_logit     = self.head_opacity(a_last)


        cond_normal = self.cond_normal_mlp(flame_cond).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        normal_in = torch.cat([feats, cond_normal], dim=1)
        _, _, _, normal_last = self.unet_normal(normal_in)

        dnormal_spec      = torch.tanh(self.head_dnormal_spec(normal_last)) * self.dnormal_spec_scale
        spec_scale        = torch.sigmoid(self.head_spec_scale(normal_last))
        return {
            "features_dc_logit": features_dc_logit,
            "opacity_logit":     opacity_logit,
            "dnormal_spec":      dnormal_spec,
            "spec_scale":        spec_scale,
        }

    def forward_lightview(
        self,
        flame_cond: torch.Tensor,
        geometry: Dict[str, torch.Tensor],
        cam_pos: Optional[torch.Tensor] = None,
        env_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = flame_cond.shape[0]
        H = W = self.uv_size

        # cam features (same construction as appearance)
        if cam_pos.dim() == 2:
            cam_pos = cam_pos.view(B, 3, 1, 1)
        elif cam_pos.dim() == 4:
            cam_pos = cam_pos.expand(B, -1, -1, -1)
        else:
            raise ValueError("cam_pos shape should be (B,3) or (1,3,1,1).")

        cam_pos = cam_pos.float().detach()
        cam_dir = F.normalize(cam_pos, dim=1, eps=1e-6)
        cam_dir = cam_dir.expand(B, 3, H, W)

        Vv   = cam_pos - geometry['pos']
        dist = Vv.norm(dim=1, keepdim=True).clamp_min(1e-6)
        V_unit = Vv / dist

        q = _qnormalize(geometry['rot'].permute(0,2,3,1).reshape(-1,4))
        z_axis = torch.tensor([0, 0, 1.0], device=q.device, dtype=q.dtype).expand(q.shape[0],3)
        n = quat_rotate(q, z_axis).view(1,H,W,3).permute(0,3,1,2)
        nv_cos = (n * V_unit)
        cam_feats = torch.cat([V_unit, dist, nv_cos], dim=1)

        # ---------- VIEW-ONLY BRANCH (no env) ----------
        # add FLAME conditioning (as requested)
        cond_view = self.cond_view_mlp(flame_cond).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        feats_view = torch.cat([
            geometry['pos'], geometry['slog'], geometry['rot'], geometry['geo_feat'],
            cam_dir, cam_feats, cond_view
        ], dim=1)
        _, _, _, view_last = self.unet_view(feats_view)
        vis_logit = self.head_view_vis(view_last)
        visibility = torch.sigmoid(vis_logit)

        # ---------- LIGHT-ONLY BRANCH (no cam) ----------
        cond_light = self.cond_light_mlp(flame_cond).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        if env_map.dim() == 3:
            env_map = env_map.unsqueeze(0)
        # 돌려보기
        # env_map_orig : [H, W, 3] numpy or torch; convert to [1,3,H,W] torch on GPU
        # env_map = rotate_env_for_direct_light(env_map, yaw_deg=0, pitch_deg=90, roll_deg=-90)
        e = self.env_encoder(env_map.detach()).view(env_map.shape[0], 64)
        env_emb = self.env_to_app(e).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        feats_light = torch.cat([
            geometry['pos'], geometry['slog'], geometry['rot'], geometry['geo_feat'],
            env_emb, cond_light
        ], dim=1)
        _, _, _, light_last = self.unet_light(feats_light)
        shading_d_logit = self.head_light_shading_d(light_last)
        # Specular는 view, light 둘다 봐야함         
        shading_s_logit = self.head_light_shading_s(torch.cat([view_last, light_last], dim=-3))
        shading_d       = torch.sigmoid(shading_d_logit)  # shadow/transmittance in [0,1]
        shading_s       = torch.sigmoid(shading_s_logit)  # shadow/transmittance in [0,1]
        return {
            "vis_logit":  vis_logit,
            "visibility": visibility,
            "shading_d_logit": shading_d_logit,
            "shading_d":   torch.sigmoid(shading_d),
            "shading_s_logit": shading_s_logit,
            "shading_s":   torch.sigmoid(shading_s),
        }

    def forward(self, geom, env_map: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        flame_cond=geom['params']
        # Predict Gaussians' geometry (dpos, dslog, drot)
        dgeom = self.forward_gaussian_dgeo(flame_cond)
        geometry = self._compose_geo_for_app(geom, dgeom)
        app_pred = self.forward_gaussian_app(
            flame_cond,
            geometry,
            cam_pos=geom["cam_pos"],
        )

        if env_map is not None:
            lv_pred = self.forward_lightview(
                flame_cond,
                geometry,
                cam_pos=geom["cam_pos"],
                env_map=env_map,
            )
            out = {
                **dgeom,
                **geometry,
                **app_pred,
                **lv_pred,
                "opacity": torch.sigmoid(app_pred["opacity_logit"]),
                "shading_d_uv": lv_pred["shading_d"],
            }
        else:
            lv_pred = None

            out = {
                **dgeom,
                **geometry,
                **app_pred,
                "opacity": torch.sigmoid(app_pred["opacity_logit"]),
            }
        return out

    # ----------------------------
    # Optim setup (minimal)
    # ----------------------------
    def training_setup(self, training_args: Dict, stage: str = "full"):
        """
        stage: "full" or "base" → train geometry + appearance
               "normal"         → train specular residual only (dnormal_spec)
        """

        # pick LR by stage
        if stage in ("full", "base"):
            lr = float(training_args.get("static_lr", 1e-3))
        elif stage == "pbr":
            lr = float(training_args.get("pbr_lr", 1e-3))
        else:
            raise ValueError("Invalid stage. Choose 'full', 'base', or 'normal'.")

        eps = float(training_args.get("adam_eps", 1e-8))

        params: list = []

        if stage in ("full", "base"):
            # geometry
            params += list(self.cond_geo_mlp.parameters())
            params += list(self.unet_geo.parameters())
            params += list(self.head_dpos.parameters())
            params += list(self.head_scale.parameters())
            params += list(self.head_rot.parameters())
            params += list(self.head_dnormal_diff.parameters())
            params += [
                        self.dpos_scale, 
                       self.scale_log_range, 
                       self.dnormal_scale]

            # appearance
            params += list(self.cond_app_mlp.parameters())
            params += list(self.unet_app.parameters())
            params += list(self.head_color.parameters())
            params += list(self.head_opacity.parameters())
            
            params += list(self.cond_normal_mlp.parameters())
            params += list(self.unet_normal.parameters())
            # params += list(self.head_dnormal_spec.parameters())
            # params += list(self.head_spec_scale.parameters())

            # view-only and light-only branches
            params += list(self.cond_view_mlp.parameters())
            params += list(self.unet_view.parameters())
            params += list(self.head_view_vis.parameters())

            params += list(self.cond_light_mlp.parameters())
            params += list(self.unet_light.parameters())
            params += list(self.head_light_shading_d.parameters())
            params += list(self.head_light_shading_s.parameters())
            # params += [self.dnormal_spec_scale]

            params += list(self.env_to_app.parameters())
            params += list(self.env_encoder.parameters())

            # keep only trainable nn.Parameter (leafs)
            params = [p for p in params if isinstance(p, nn.Parameter) and p.requires_grad]

        elif stage == "pbr":
            # params_geo += list(self.cond_normal_mlp.parameters())

            params += list(self.cond_normal_mlp.parameters())
            params += list(self.unet_normal.parameters())
            params += list(self.head_dnormal_spec.parameters())
            params += list(self.head_spec_scale.parameters())
            # params += list(self.head_dnormal_diff.parameters())
            # params += [self.dnormal_scale]

            # view-only and light-only branches
            params += list(self.cond_view_mlp.parameters())
            params += list(self.unet_view.parameters())
            params += list(self.head_view_vis.parameters())

            params += list(self.cond_light_mlp.parameters())
            params += list(self.unet_light.parameters())
            params += list(self.head_light_shading_d.parameters())
            params += list(self.head_light_shading_s.parameters())
            params += [self.dnormal_spec_scale]

            params += list(self.env_to_app.parameters())
            params += list(self.env_encoder.parameters())
            
            # keep only trainable nn.Parameter (leafs)
            params = [p for p in params if isinstance(p, nn.Parameter) and p.requires_grad]
        else:
            params = []

        # if len(params) == 0:
        #     raise RuntimeError("[UVPredictorUNet.training_setup] No trainable parameters collected.")

        # single Adam (no custom 'name' keys)
        self.optimizer = torch.optim.Adam(params, lr=lr, eps=eps)

        # optional scheduler
        sched_fn = training_args.get("lr_scheduler", None)
        self.scheduler = sched_fn(self.optimizer) if callable(sched_fn) else None



