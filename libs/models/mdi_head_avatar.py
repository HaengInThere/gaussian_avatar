import copy
import torch, os
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from pytorch3d.io   import load_obj
from pytorch3d.ops  import knn_points
from pytorch3d.transforms import (
    quaternion_to_axis_angle,
    matrix_to_quaternion,
    quaternion_multiply,
    axis_angle_to_quaternion
)

from libs.models.flame            import FlameHead
from libs.models.gaussian_model   import GaussianModel
# from libs.render.render_3dgs      import 
# from libs.render.render_step1     import render
# from libs.render.mesh_renderer    import NVDiffRenderer
from libs.nets.gaussian_uv_mapper import GaussianUVMapper
from libs.utils.general_utils   import inverse_sigmoid, RGB2SH, get_bg_color
from libs.utils.graphics_utils  import compute_vertex_normals, quat_rotate, update_normal_expmap, quat_from_two_vectors, rodrigues_to_quat
from libs.utils.camera_utils import get_camera_position
from libs.utils.mesh_sampling     import (
    uniform_sampling_barycoords,
    reweight_uvcoords_by_barycoords,
    reweight_verts_by_barycoords,
    sample_uvmap_by_barycoords
    
)
from libs.utils.mesh_compute      import (
    compute_face_orientation,
    compute_face_normals
)

import warnings
warnings.filterwarnings("ignore", message="No mtl file provided")
warnings.filterwarnings("ignore", message="Mtl file does not exist")

#-------------------------------------------------------------------------------#
def _get_attr(obj, names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None

class MDIHeadAvatar(nn.Module):
    def __init__(self, cfg, shape_params, static_offset, device='cuda'):
        super().__init__()
        self.cfg = cfg
        self.uv_resolution = cfg["tex_size"]
        self.shell_len     = float(max(cfg["normal_offset"], 1e-3)) 
        self.rodriguez_rotation = True
        self.max_sh_degree = 0
        self.device = device
        # self.device = device
        self.optimizer = None
        self.scheduler = None
        self._uv_valid_mask = None

        self.shape_params           = torch.tensor(shape_params)
        self.static_offset          = torch.tensor(static_offset)
        self.canonical_expression   = torch.zeros((1,self.cfg["n_expr"]))
        self.canonical_pose         = torch.zeros((1,self.cfg["n_pose"]))
        self.canonical_trans        = torch.zeros((1,3))

        self._register_flame()
        self._register_template_mesh(template_path=self.cfg["template_mesh_path"])


        self.uv_mapper = GaussianUVMapper(flame_model=None, device=device,
                                uvcoords=self.uvcoords, # 원래 그대로 
                                uvfaces=self.uvfaces,
                                tri_faces=self.faces).to(device)

        mean_scaling, max_scaling, scale_init = self.get_init_scale_by_knn(self.verts_sampling) 

        self.register_buffer('mean_scaling', mean_scaling)
        self.register_buffer('max_scaling', max_scaling)
        self.register_buffer('scale_init', scale_init)

        # ---------------- UV-space geometric maps (learnable) ----------------
        H = self.uv_resolution
        W = self.uv_resolution

        # 로그-스케일(3,H,W), xyz-델타(3,H,W), 로우 쿼터니언(4,H,W)
        self.canon_slog = nn.Parameter(
            torch.full((3, H, W), float(self.scale_init.item())).requires_grad_(True)
        )
        self.canon_xyz = nn.Parameter(
            torch.zeros(3, H, W).requires_grad_(True)
        )
        self.canon_rot = nn.Parameter(
            torch.zeros(4, H, W).requires_grad_(True)
        )
        self.canon_normal = nn.Parameter(
            torch.zeros(3, H, W).requires_grad_(True)
        )
        with torch.no_grad():
            # identity quat (w,x,y,z) = (1,0,0,0)
            self.canon_rot[0].fill_(1.0)

        
        _, face_scaling_canonical    = compute_face_orientation(self.canonical_verts.squeeze(0), self.faces, return_scale=True)
        self.register_buffer('face_scaling_canonical', face_scaling_canonical)

        self.delta_shapedirs    = torch.zeros_like(self.flame.shapedirs)
        # self.delta_shapedirs    = nn.Parameter(self.delta_shapedirs.requires_grad_(True))

        self.delta_posedirs     = torch.zeros_like(self.flame.posedirs)
        # self.delta_posedirs     = nn.Parameter(self.delta_posedirs.requires_grad_(True))

        self.delta_vertex       = torch.zeros_like(self.flame.v_template)
        self.delta_vertex       = nn.Parameter(self.delta_vertex.requires_grad_(True))

        # self.spec_residual_gain = float(self.cfg.get("spec_residual_gain", 0.5))

        self.uv_offset           = torch.zeros(1, self.uv_resolution, self.uv_resolution) # 1, w, h
        # self.uv_offset           = nn.Parameter(self.uv_offset.requires_grad_(True)).float()

        self.uv_color           = inverse_sigmoid(0.5 * torch.ones(3, self.uv_resolution, self.uv_resolution)) # feature_dim, 3, w, h
        # self.uv_color           = nn.Parameter(self.uv_color.requires_grad_(True)).float()

        self.uv_opacity         = inverse_sigmoid(0.1 * torch.ones(1, self.uv_resolution, self.uv_resolution))
        # self.uv_opacity         = nn.Parameter(self.uv_opacity.requires_grad_(True)).float()

        self.uv_normal           = torch.zeros(3, self.uv_resolution, self.uv_resolution) # 3, w, h
        # self.uv_normal           = nn.Parameter(self.uv_normal.requires_grad_(True)).float()

        # NOTE: wrap as nn.Parameter **after** setting dtype; do NOT call .float() on Parameter (it returns a plain Tensor)
        self.uv_albedo    = nn.Parameter(
            torch.ones(3, self.uv_resolution, self.uv_resolution, dtype=torch.float32)*0.2, requires_grad=True
        )
        self.uv_roughness = nn.Parameter(
            torch.zeros(1, self.uv_resolution, self.uv_resolution, dtype=torch.float32), requires_grad=True
        )
        self.uv_metallic  = nn.Parameter(
            torch.zeros(1, self.uv_resolution, self.uv_resolution, dtype=torch.float32), requires_grad=True
        )
        # assert isinstance(self.uv_albedo, nn.Parameter)
        # assert isinstance(self.uv_roughness, nn.Parameter)
        # assert isinstance(self.uv_metallic, nn.Parameter)


        self._register_init_gaussian()

        
    def _register_flame(self):

        self.flame = FlameHead(
            shape_params         = self.cfg["n_shape"],
            expr_params          = self.cfg["n_expr"],
            add_teeth=True,
        )
        
        canonical_verts, canonical_pose_feature, canonical_transformations = self.flame(
            self.shape_params[None, ...],
            self.canonical_expression,
            self.canonical_pose[:,:3],
            self.canonical_pose[:,3:6],
            self.canonical_pose[:,6:9],
            self.canonical_pose[:,9:],
            self.canonical_trans,
            self.static_offset,
            # delta_shapedirs=self.delta_shapedirs,
            # delta_posedirs=self.delta_posedirs,
        )
        
        # make sure call of FLAME is successful
        self.canonical_verts                    = canonical_verts
        self.flame.canonical_verts              = canonical_verts.squeeze(0)
        self.flame.canonical_pose_feature       = canonical_pose_feature
        self.flame.canonical_transformations    = canonical_transformations
    
    def _register_template_mesh(self, template_path):

        #----------------   load head mesh & process UV ----------------#
        verts, faces, aux = load_obj(template_path)

        uvcoords = aux.verts_uvs
        # uvcoords = uvcoords * 2 - 1  # Normalize to [-1, 1]
        # uvcoords[:, 1] = -uvcoords[:, 1]

        uvfaces     = faces.textures_idx
        faces       = faces.verts_idx

        face_index, bary_coords = uniform_sampling_barycoords(
            num_points    = self.uv_resolution * self.uv_resolution,
            tex_coord     = uvcoords,
            uv_faces      = uvfaces
        )

        uvcoords_sample = reweight_uvcoords_by_barycoords(
            uvcoords    = uvcoords,
            uvfaces     = uvfaces,
            face_index  = face_index,
            bary_coords = bary_coords
        )
        
        uvcoords_sample = uvcoords_sample[...,:2]

        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('faces', faces)
        self.register_buffer('template_verts', verts)
        self.register_buffer('face_index', face_index)
        self.register_buffer('bary_coords', bary_coords)
        self.register_buffer('uvcoords_sample', uvcoords_sample)

        #----------------   sample points in template mesh  ----------------#
        verts_sampling = reweight_verts_by_barycoords(
            verts         = verts.unsqueeze(0),
            faces         = faces,
            face_index    = face_index,
            bary_coords   = bary_coords
        )
    
        self.register_buffer('verts_sampling', verts_sampling)

        # uvcoords_sample: (N, 2) in [0,1]
        # _register_template_mesh() 안, uv_grid 만들기 직전에 한 줄 추가
        uvcoords_sample_flip = self.uvcoords_sample.clone()
        uvcoords_sample_flip[..., 1] = 1.0 - uvcoords_sample_flip[..., 1]       
        uvcoords_sample_flip[..., 0] = 1.0 - uvcoords_sample_flip[..., 0]  

        uv_grid = uvcoords_sample_flip * 2.0 - 1.0            # to [-1, 1]
        uv_grid = uv_grid.view(1, -1, 1, 2).contiguous().float()
        self.register_buffer('uv_grid', uv_grid)
        self.num_points = self.uv_resolution * self.uv_resolution   

    def _register_init_gaussian(self):
        # self.num_points = self.verts_sampling.shape[1]

        # purely geometric 파라미터만 남기고,
        # appearance 파라미터는 매 forward() 때 UV에서 새로 샘플한다.
        # scales = self.scale_init[..., None].repeat(self.num_points, 3).to(self.device)
        scale_base = (self.scale_init * self.cfg.get("init_scale_mult", 0.5))
        scales = scale_base[..., None].repeat(self.num_points, 3)
        rots   = torch.zeros((self.num_points, 4))
        rots[:, 0] = 1

        # ────────────────────────────────────────────────
        # ① offset 은 ‘학습 가능한 상수’로만 둔다
        #    (UV 텍스처에서 가져오지 않음 → 그래프 clean)
        self._offset = nn.Parameter(
            torch.zeros(self.num_points, 1, requires_grad=True)
        )

        # ② SH DC 용 placeholder (값은 forward 에서 매번 덮어씀)
        self._features_dc   = nn.Parameter(
            torch.zeros(self.num_points, 1, 3),  requires_grad=False
        )
        self._features_rest = torch.empty(
            self.num_points, 0, 3, requires_grad=False
        )

        # ③ geometry-only 파라미터
        # self._scaling  = nn.Parameter(scales, requires_grad=True)
        # self._rotation = nn.Parameter(rots,   requires_grad=True)
        self._scaling  = scales
        self._rotation = rots

        # 나머지 PBR 파라미터도 ‘빈 텐서 + no-grad’로 둡니다.
        self._opacity   = torch.empty(self.num_points, 1)   # no grad
        self._normal    = torch.empty(self.num_points, 3)
        self._albedo    = torch.empty(self.num_points, 3)
        self._roughness = torch.empty(self.num_points, 1)
        self._metallic  = torch.empty(self.num_points, 1)

        self.register_buffer('xyz_gradient_accum', torch.zeros(self.num_points, 1))
        self.register_buffer('denom',              torch.zeros(self.num_points, 1))
        self.register_buffer('max_radii2D',        torch.zeros(self.num_points))
        # if you use this as a boolean mask during training, make it bool:
        self.register_buffer('sample_flag',        torch.zeros(self.num_points, dtype=torch.bool))


    def _sample_uv_pbr_features(self):

        # self.uv_albedo, self.uv_roughness, self.uv_metallic: (C,H,W)

        def _precompute_xy_idx(H: int, W: int):
            # self.uv_grid: (1,N,1,2) in [-1,1]
            grid = self.uv_grid.view(1, -1, 2)[0]
            gx, gy = grid[:, 0], grid[:, 1]
            # align_corners=True → idx = round((g+1)/2 * (size-1))
            x = torch.round((gx + 1.0) * 0.5 * (W - 1)).long().clamp_(0, W - 1)
            y = torch.round((gy + 1.0) * 0.5 * (H - 1)).long().clamp_(0, H - 1)
            return x, y

        # albedo
        C, H, W = self.uv_albedo.shape
        x_idx, y_idx = _precompute_xy_idx(H, W)
        albedo_lin = self.uv_albedo[ :, y_idx, x_idx ].transpose(0, 1).contiguous()   # (N,3)

        # roughness
        Cr, Hr, Wr = self.uv_roughness.shape
        if (Hr, Wr) != (H, W):
            xr, yr = _precompute_xy_idx(Hr, Wr)
        else:
            xr, yr = x_idx, y_idx
        roughness_lin = self.uv_roughness[ :, yr, xr ].transpose(0, 1).contiguous()    # (N,1)

        # metallic
        Cm, Hm, Wm = self.uv_metallic.shape
        if (Hm, Wm) != (H, W):
            xm, ym = _precompute_xy_idx(Hm, Wm)
        else:
            xm, ym = x_idx, y_idx
        metallic_lin = self.uv_metallic[ :, ym, xm ].transpose(0, 1).contiguous()      # (N,1)

        # 범위 리매핑
        rmax, rmin = 1.0, 0.04
        roughness_lin = roughness_lin * (rmax - rmin) + rmin

        return albedo_lin, roughness_lin, metallic_lin
    
    def _sample_uv_features_from(self, uv_maps):
        """
        uv_maps: dict of (B=1,C,H,W) 텐서들. 반환은 {key: (N,C)}
        """
        # ★ self.uv_grid: (1,N,1,2) in [-1,1], align_corners=True 가정
        #    한 번만 N→픽셀 인덱스로 변환해서 모든 텍스처에 공통 사용
        _computed_idx = {"x": None, "y": None, "HW": None}

        def _get_xy_idx_for(tex4):
            # tex4: (1,C,H,W)
            H, W = tex4.shape[-2:]
            if (_computed_idx["HW"] is None) or (_computed_idx["HW"] != (H, W)):
                # (1,N,1,2) → (N,2)
                grid = self.uv_grid.view(1, -1, 2)[0]     # NDC [-1,1]
                gx = grid[:, 0]
                gy = grid[:, 1]
                # align_corners=True: idx = round( (g+1)/2 * (size-1) )
                x = torch.round((gx + 1.0) * 0.5 * (W - 1)).long()
                y = torch.round((gy + 1.0) * 0.5 * (H - 1)).long()
                x = x.clamp_(0, W - 1)
                y = y.clamp_(0, H - 1)
                _computed_idx.update({"x": x, "y": y, "HW": (H, W)})
            return _computed_idx["x"], _computed_idx["y"]

        def samp4(tex4, mask4=None):
            # tex4: (1,C,H,W), mask4: (1,1,H,W) or None
            x_idx, y_idx = _get_xy_idx_for(tex4)

            if mask4 is None:
                # 최근접 픽셀 인덱싱으로 그대로 읽기 → (C,N) → (N,C)
                out = tex4[0, :, y_idx, x_idx].transpose(0, 1).contiguous()
            else:
                # 값과 마스크를 각각 최근접 인덱싱 후 정규화
                val = (tex4 * mask4)[0, :, y_idx, x_idx]             # (C,N)
                wei =  mask4[y_idx, x_idx]                     # (1,N)
                out = (val / (wei + 1e-8)).transpose(0, 1).contiguous()
            return out  # (N,C)

        sampled = {}
        keys = [
            "features_dc_logit", "opacity_logit", "features_dc", "opacity",
            "normal","dnormal","dnormal_spec",
            "albedo", "roughness", "metallic", "offset",
            "dpos", "dslog", "drot",
            "pos", "slog", "rot",
            "shading_d_logit", "shading_d", 
            "shading_s_logit", "shading_s",
            "vis_logit", "visibility"
        ]

        # 유효 영역 마스크 준비. 값 맵과 해상도 동일해야 함: (1,1,H,W)
        mask4 = self._uv_valid_mask if isinstance(self._uv_valid_mask, torch.Tensor) else None

        for k in keys:
            if (uv_maps is not None) and (k in uv_maps) and (uv_maps[k] is not None):
                # appearance 맵과 기하 절대값은 마스크 적용 추천
                if k in ["pos", "normal", "albedo", "roughness", "metallic", "offset", "features_dc", "opacity"]:
                    sampled[k] = samp4(uv_maps[k], mask4=mask4)
                else:
                    # 델타류는 네트워크 전역 예측이 많아 마스크 없이도 OK. 필요시 mask4로 바꿔도 됨
                    sampled[k] = samp4(uv_maps[k], mask4=None)
        return sampled

    def forward_geometry(self, viewpoint_cam):
        # ---- FLAME ----
        verts, _, _ = self.flame(
            viewpoint_cam.flame_param["shape"][None, ...],
            viewpoint_cam.flame_param["expr"],
            viewpoint_cam.flame_param["rotation"],
            viewpoint_cam.flame_param["neck_pose"],
            viewpoint_cam.flame_param["jaw_pose"],
            viewpoint_cam.flame_param["eyes_pose"],
            viewpoint_cam.flame_param["translation"],
            static_offset=viewpoint_cam.flame_param["static_offset"],
            # delta_shapedirs=self.delta_shapedirs,
            # delta_posedirs=self.delta_posedirs,
            delta_vertex=self.delta_vertex
        )

        # ---------- Build (expr + pose + translation) → flame_cond (B=1, 118) ----------
        device = verts.device
        # expression
        expr = None
        if isinstance(viewpoint_cam.flame_param, dict):
            expr = viewpoint_cam.flame_param.get("expr", viewpoint_cam.flame_param.get("expression", None))
        else:
            expr = getattr(viewpoint_cam.flame_param, "expr",
                getattr(viewpoint_cam.flame_param, "expression", None))
        # pose parts
        pose_parts = []
        if isinstance(viewpoint_cam.flame_param, dict):
            for k in ["rotation", "neck_pose", "jaw_pose", "eyes_pose", "translation"]:
                v = viewpoint_cam.flame_param.get(k, None)
                if v is not None:
                    pose_parts.append(v.view(1, -1))
        else:
            for k in ["rotation", "neck_pose", "jaw_pose", "eyes_pose", "translation"]:
                v = getattr(viewpoint_cam.flame_param, k, None)
                if v is not None:
                    pose_parts.append(v.view(1, -1))
        pose = torch.cat(pose_parts, dim=1) if len(pose_parts) > 0 else None

        expr = expr.view(1, -1)
        pose = pose.view(1, -1)

        flame_cond = torch.cat([expr, pose], dim=1)
        flame_cond = flame_cond.float().detach()
        
        face_orien_mat, face_scaling = compute_face_orientation(verts, self.faces, return_scale=True)
        face_normals = compute_face_normals(verts, self.faces)

        # per-vertex normal (V,3) -> UV 샘플 포인트(N,3)
        verts_normal = compute_vertex_normals(verts.squeeze(0), self.faces)          # (V,3)

        scaling_ratio       = face_scaling / self.face_scaling_canonical
        flame_scaling_ratio = scaling_ratio[:, self.face_index]
        flame_orien_mat     = face_orien_mat[:, self.face_index]
        flame_orien_quat    = matrix_to_quaternion(flame_orien_mat)
        flame_normals       = face_normals[:, self.face_index]
        verts_normal = compute_vertex_normals(verts.squeeze(0), self.faces) 

        # base_pos = reweight_verts_by_barycoords(
        #     verts=verts,
        #     faces=self.faces,
        #     face_index=self.face_index,
        #     bary_coords=self.bary_coords
        # )
        # base_normal = reweight_verts_by_barycoords(
        #     verts=verts_normal.unsqueeze(0),   # (1,V,3)
        #     faces=self.faces,
        #     face_index=self.face_index,
        #     bary_coords=self.bary_coords
        # )                  

        # UV 래스터 맵 생성
        uv_base, uv_valid_mask = self.uv_mapper.unwrap_to_uv_rasterize(
            vertex_values=torch.concat([verts[0], verts_normal], dim=1), texture_resolution=self.uv_resolution)
        uv_base = uv_base.permute(2,0,1).unsqueeze(0) # (1,C,H,W)
        uv_base_pos, uv_base_normal, uv_flame_orient = uv_base[:,:3], uv_base[:,3:6], uv_base[:,6:]

        # -------- Camera UV maps for per-pixel appearance prediction --------
        # cam_pos: (1,3,1,1)
        cam_pos = get_camera_position(viewpoint_cam, device=self.device)
        if cam_pos.dim() == 2:
            cam_pos = cam_pos.view(1, 3, 1, 1)
        elif cam_pos.dim() == 1:
            cam_pos = cam_pos.view(1, 3, 1, 1)
        # pos_val[0]: (N,3) → (1,3,H,W)
        H = W = self.uv_resolution

        # reg_terms = {}
        # if self.delta_vertex is not None: reg_terms['delta_vertex'] = (self.delta_vertex ** 2).mean()

        return {
            'verts': verts,
            'faces': self.faces,
            'flame_scaling_ratio': flame_scaling_ratio,
            'flame_orien_quat': flame_orien_quat,
            'flame_normals': flame_normals,
            'verts_normal': verts_normal,  # (V,3)
            'uv_valid_mask': uv_valid_mask,

            'params': flame_cond,
            # 'reg_terms' : reg_terms,
            'cam_pos': cam_pos,
            'uv_base_pos': uv_base_pos,
            'uv_base_normal': uv_base_normal,

            'canon_xyz': self.canon_xyz,
            'canon_slog':self.canon_slog,
            'canon_rot': self.canon_rot,
        }

    def _bary_reweight_attr(self, attr_verts: torch.Tensor) -> torch.Tensor:
        """
        Map a per-vertex attribute (V, D) onto our N UV samples via face_index & bary_coords.
        Returns (N, D).
        """
        faces = self.faces.long()                 # (F,3)
        f_attr = attr_verts[faces]                # (F,3,D)
        sel    = f_attr[self.face_index.long()]   # (N,3,D)
        bc     = self.bary_coords[..., None]      # (N,3,1)
        out    = (sel * bc).sum(dim=1)            # (N,D)
        return out

    def build_gaussian(self, geom, uv_maps=None):

        self._uv_valid_mask = geom.get('uv_valid_mask', None)   # ★ mask 보관
        # Always sample PBR features at the top (only once)
        albedo_lin, roughness_lin, metallic_lin = self._sample_uv_pbr_features()

        eps_floor = 0.01
        # ===== Appearance =====
        sampled = self._sample_uv_features_from(uv_maps)
        eps = 1e-4
        # color logits
        color_logit = sampled['features_dc_logit']

        # opacity logits
        opacity_logit_raw = sampled['opacity_logit']
        prob = torch.sigmoid(opacity_logit_raw)

        prob = prob * (1.0 - eps_floor) + eps_floor
        prob = prob.clamp(eps_floor, 1 - 0.00001)
        opacity_logit_eff = inverse_sigmoid(prob)
        

        # ===== Gaussian container =====
        gaussian = GaussianModel(sh_degree=0)
        gaussian._features_dc   = color_logit[:, None, :]  # (N,1,3)
        gaussian._features_rest = torch.empty(color_logit.shape[0], 0, 3, requires_grad=False)
        gaussian._opacity       = opacity_logit_eff
        gaussian._albedo        = albedo_lin
        gaussian._roughness     = roughness_lin
        gaussian._metallic      = metallic_lin

        # ===== Geometry =====
        gaussian._xyz = sampled['pos']

        # --- after ---
        LOG_S_MIN, LOG_S_MAX = -8.0, -0.1  # exp() 기준 ≈ [3.4e-4, 20.1]
        slog = sampled['slog'].clamp(LOG_S_MIN, LOG_S_MAX)
        gaussian._scaling = slog
        # gaussian._scaling = sampled['slog']

        gaussian._rotation = sampled['rot']

        # Normal
        if 'dnormal_spec' in sampled:
            n = sampled['normal']
            d = sampled['dnormal_spec']

            # # tangent plane으로 투영
            # d_proj = d - (d * n).sum(-1, keepdim=True) * n

            # # 크기 제한
            # d_proj = torch.clamp(d_proj, -0.2, 0.2)  # optional, 안정성↑
            # print(1)
            # n_hat = F.normalize(n + d_proj, dim=-1, eps=1e-6)
            # gaussian._normal = n_hat

            # reg_dnormal = 1 - (n * n_hat).sum(-1).mean()

            gaussian._normal = F.normalize(n + d, dim=1, eps=1e-6)
            
        else:
            gaussian._normal = F.normalize(sampled['normal'], dim=-1, eps=1e-6)
            reg_dnormal = 0

        # reg는 그대로
        if torch.isnan(gaussian._features_dc).sum() or torch.isnan(gaussian._opacity).sum() or torch.isnan(gaussian._xyz).sum() or torch.isnan(gaussian._rotation).sum() or torch.isnan(gaussian._scaling).sum():
            print(gaussian._features_dc)
            print(gaussian._opacity)


        # -------------------------
        # Regularization terms
        # -------------------------
        # 유효 UV 마스크가 있으면 그 위에서 평균. 없으면 전 픽셀
        mask = geom.get("uv_valid_mask", None)  # (1,1,H,W) or (B,1,H,W)
        if mask is not None:
            if mask.dim() == 4 and mask.shape[0] == 1 and B > 1:
                mask = mask.expand(B, -1, -1, -1)
            m = mask.float().clamp(min=0.0, max=1.0)
            m = m.expand(1, 1, self.uv_resolution, self.uv_resolution)
            def _mean_on_mask(x):
                return (x * m).sum() / (m.sum() + 1e-8)
        else:
            def _mean_on_mask(x):
                return x.mean()

        reg_canon_xyz = _mean_on_mask(self.canon_xyz.abs()).mean(dim=0, keepdim=True)
        # 1) 델타 크기 억제
        reg_dpos   = _mean_on_mask((uv_maps['dpos']   ** 2).sum(dim=1, keepdim=True))      # R^3
        # reg_dslog  = _mean_on_mask((uv_maps['dslog']  ** 2).sum(dim=1, keepdim=True))      # R^3

        # 2) 회전 정체성 가까이. drot ≈ identity
        # drot는 이미 정규화. geodesic 근사로 (1 - w)^2 또는 |log(q)|^2 사용 가능
        reg_drot_id = _mean_on_mask(((1.0 - uv_maps['drot'][:, :1]) ** 2))                 # w-성분 중심

        reg_terms = {
            "canon_l1":       reg_canon_xyz,
            "dpos_l2":        reg_dpos,
            # "dslog_l2":       reg_dslog,
            # "dnormal_l2":     reg_dnormal,
            "drot_identity":  reg_drot_id,
            # "tv_dpos":        reg_tv_dpos,
            # "tv_dslog":       reg_tv_dslog,
            # "tv_drot":        reg_tv_drot,
            # "slog_softbound": reg_slog_bound,
            # "dnormal" :         reg_dnormal
        }

        return {
            'sampled': sampled,
            'scale': torch.exp(gaussian._scaling),                 # (N,3)
            'raw_rot': quaternion_to_axis_angle(gaussian._rotation),  # (N,3)
            'gaussian': gaussian,
            'verts': geom['verts'],
            'faces': geom['faces'],
            'reg_terms': reg_terms,
        }

    
    def forward(self, viewpoint_cam):
        geom = self.forward_geometry(viewpoint_cam)
        return self.build_gaussian(geom, uv_maps=None)

    def save_gs(self, save_path, gaussian):
        gaussian.save_ply(save_path)
        

    @staticmethod
    def get_init_scale_by_knn(points: torch.Tensor, max_points: int = 30000):
        """Robust KNN-based scale init. Subsample large N to avoid OOM/segfault."""
        # points: (1, N, 3) or (N, 3)
        if points.dim() == 3:
            assert points.shape[0] == 1, f"expected B=1, got {points.shape}"
            pts = points[0]
        else:
            pts = points
        device = pts.device
        N = pts.shape[0]
        if N > max_points:
            # random subsample on the same device
            idx = torch.randperm(N, device=device)[:max_points]
            pts_sub = pts[idx].unsqueeze(0)  # (1, M, 3)
        else:
            pts_sub = pts.unsqueeze(0)  # (1, N, 3)
        try:
            knn = knn_points(pts_sub.float(), pts_sub.float(), K=6)
            dists = torch.sqrt(knn.dists[..., 1])  # (1, M)
            mean_scaling = dists.mean()
        except Exception as e:
            # Fallback: use a tiny positive scale to proceed
            print(f"[MDI][WARN] knn_points failed with N={N}: {e}. Fallback to default scale.")
            mean_scaling = torch.tensor(1e-3, device=device)
        max_scaling = 10 * mean_scaling
        scale_init = torch.log(mean_scaling).detach().cpu()
        return mean_scaling, max_scaling, scale_init
    
    def _uv_densify(self, increase_num=1000):
        # --- 기존 그라드 읽기 ---
        xyz_grad = self.xyz_gradient_accum.squeeze(1)  # (N,)

        # ========== [A] face 선택을 "전체 uvfaces"에서 하도록 변경 ==========
        F = self.uvfaces.shape[0]
        device = self.device

        # 1) UV face 면적 (2D 폴리곤 면적)
        uv_tri = self.uvcoords[self.uvfaces]        # (F, 3, 2)
        v0, v1, v2 = uv_tri[:,0], uv_tri[:,1], uv_tri[:,2]
        # cross2D로 삼각형 면적: 0.5 * |(v1-v0) x (v2-v0)|
        area2 = torch.abs((v1[:,0]-v0[:,0])*(v2[:,1]-v0[:,1]) - (v1[:,1]-v0[:,1])*(v2[:,0]-v0[:,0]))
        face_area = 0.5 * area2 + 1e-12             # (F,)

        # 2) face별 "필요도" (선택): 기존 포인트의 gradient를 face로 집계 → 평균
        face_need = torch.zeros(F, device=device)
        face_cnt  = torch.zeros(F, device=device)
        face_need.scatter_add_(0, self.face_index, xyz_grad)         # 합
        face_cnt  .scatter_add_(0, self.face_index, torch.ones_like(xyz_grad))
        face_cnt  = face_cnt.clamp_min(1.0)
        face_need = face_need / face_cnt                              # 평균
        # 없애고 싶으면 face_need를 1로 두면 됨
        # face_need = torch.ones_like(face_area)

        # 3) 샘플 확률 = 면적 × 필요도
        weights = face_area * (face_need + 1e-6)
        if torch.isfinite(weights).any():
            weights = weights.clamp_min(1e-12)
            weights = weights / weights.sum()
        else:
            # 모든 값이 0/NaN일 때 안전장치: 면적만으로
            weights = face_area / face_area.sum()

        replacement = (increase_num > F)
        new_face_index = torch.multinomial(weights, increase_num, replacement=replacement)  # (M,)

        # ========== [B] 균등 바리센터릭 (sqrt-trick) ==========
        u = torch.rand(increase_num, device=device)
        v = torch.rand(increase_num, device=device)
        su = torch.sqrt(u)
        new_bary_coords = torch.stack([1.0 - su, su * (1.0 - v), su * v], dim=-1)  # (M,3)

        # ----- 기존 코드와 동일한 합치기/재계산 -----
        self.face_index  = torch.cat([self.face_index, new_face_index], dim=0)
        self.bary_coords = torch.cat([self.bary_coords, new_bary_coords], dim=0)

        new_sample_flag  = torch.ones((increase_num), device=device)
        self.sample_flag = torch.cat([self.sample_flag, new_sample_flag], dim=0)

        self.num_points  = self.bary_coords.shape[0]

        self.xyz_gradient_accum = torch.zeros((self.num_points, 1), device=device)
        self.denom             = torch.zeros((self.num_points, 1), device=device)
        self.max_radii2D       = torch.zeros((self.num_points),     device=device)

        # UV 좌표 재생성
        self.uvcoords_sample = reweight_uvcoords_by_barycoords(
            uvcoords    = self.uvcoords,
            uvfaces     = self.uvfaces,
            face_index  = self.face_index,
            bary_coords = self.bary_coords
        )
        self.uvcoords_sample = self.uvcoords_sample[..., :2]

        uvcoords_sample_flip = self.uvcoords_sample.clone()
        uvcoords_sample_flip[..., 1] = 1.0 - uvcoords_sample_flip[..., 1]
        uvcoords_sample_flip[..., 0] = 1.0 - uvcoords_sample_flip[..., 0]
        uv_grid = uvcoords_sample_flip * 2.0 - 1.0
        self.uv_grid = uv_grid.view(1, -1, 1, 2).contiguous().float()



    def _add_densification_stats(self, viewspace_point_tensor, update_filter):

        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def add_default_points(self, gs_optimizer: torch.optim.Adam):
        """
        Add a set of default Gaussians, as points on the back of the head are pruned during monocular training.
        """

        default_number = self.verts_sampling.shape[1]

        init_rgb = inverse_sigmoid(torch.Tensor([0.5, 0.5, 0.5]))[None, ...].float()
        init_rgb = init_rgb.repeat_interleave(default_number, dim=0).to(self.device)
        features = torch.zeros((init_rgb.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(self.device)
        features[:, :3, 0 ] = init_rgb
        features[:, 3:, 1:] = 0.0
        scales      = self.scale_init[...,None].repeat(default_number, 3).to(self.device)
        rots        = torch.zeros((default_number, 4), device=self.device)
        rots[:, 0]  = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((default_number, 1), dtype=torch.float, device=self.device))
        offset = torch.zeros((default_number, 1), device=self.device)

        new_offset        = nn.Parameter(offset.requires_grad_(True))
        new_features_dc   = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        new_scaling       = nn.Parameter(scales.requires_grad_(True))
        new_rotation      = nn.Parameter(rots.requires_grad_(True))
        new_opacity       = nn.Parameter(opacities.requires_grad_(True))

        new_face_index, new_bary_coords = uniform_sampling_barycoords(
            num_points    = self.uv_resolution * self.uv_resolution,
            tex_coord     = self.uvcoords,
            uv_faces      = self.uvfaces
        )

        new_attribute = {
            "opacity": new_opacity,
            "offset": new_offset,
            "color": new_features_dc,
            "rotation": new_rotation,
            "scaling": new_scaling
        }

        optimizable_tensors = {}
        for group in gs_optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = new_attribute[group["name"]]
            stored_state = gs_optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del gs_optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                gs_optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        self._opacity       = optimizable_tensors["opacity"]
        self._offset        = optimizable_tensors["offset"]
        self._features_dc   = optimizable_tensors["color"]
        self._rotation      = optimizable_tensors["rotation"]
        self._scaling       = optimizable_tensors["scaling"]

        self.face_index     = torch.cat([self.face_index, new_face_index], dim=0)
        self.bary_coords    = torch.cat([self.bary_coords, new_bary_coords], dim=0)

        new_sample_flag     = torch.ones((default_number), device=self.device)
        self.sample_flag    = torch.cat([self.sample_flag, new_sample_flag], dim=0)

        self.num_points     = self.bary_coords.shape[0]

        self.xyz_gradient_accum = torch.zeros((self.num_points, 1), device=self.device)
        self.denom = torch.zeros((self.num_points, 1), device=self.device)
        self.max_radii2D = torch.zeros((self.num_points), device=self.device)
        

    def training_setup(self, cfg_train: dict, stage: str = "full"):
        """
        Returns: torch.optim.Optimizer (단일 Adam, param_groups로 구분)
        stage: 'base' | 'pbr' | 'full'  (별칭 'stage1' -> 'base')
        """
        import torch.nn as nn
        import torch


        def cfg(key, default): return cfg_train.get(key, default)
        def is_trainable(p):   return isinstance(p, nn.Parameter) and p.requires_grad
        def add_param(bucket, name, attr, lr):
            p = getattr(self, attr, None)
            if is_trainable(p):
                bucket.append({"params": [p], "lr": float(lr), "name": name})

        groups = []
        adam_eps = float(cfg("adam_eps", 1e-6))

        if stage == "base":
            add_param(groups, "delta_vertex", "delta_vertex", cfg("delta_vertex_lr", 0.00001))
            # canonical UV geo (옵션)
            add_param(groups, "canon_xyz",  "canon_xyz",  cfg("lr_uv_canon",     1e-5))
            add_param(groups, "canon_slog", "canon_slog", cfg("lr_uv_canon",     1e-5))
            add_param(groups, "canon_rot",  "canon_rot",  cfg("lr_uv_canon",     1e-5))

        elif stage == "pbr":
            pbr_lr = float(cfg("pbr_tex_lr", 1e-3))
            add_param(groups, "albedo",    "uv_albedo",    pbr_lr)
            add_param(groups, "roughness", "uv_roughness", pbr_lr)
            add_param(groups, "metallic",  "uv_metallic",  pbr_lr)
            if len(groups) == 0:
                raise RuntimeError("[training_setup] No PBR parameters found.")

        elif stage == "full":
            # base
            add_param(groups, "delta_vertex", "delta_vertex", cfg("delta_vertex_lr", 0.00001))
            add_param(groups, "canon_xyz",  "canon_xyz",  cfg("lr_uv_canon",     1e-5))
            add_param(groups, "canon_slog", "canon_slog", cfg("lr_uv_canon",     1e-5))
            add_param(groups, "canon_rot",  "canon_rot",  cfg("lr_uv_canon",     1e-5))
            # pbr
            pbr_lr = float(cfg("pbr_tex_lr", 1e-1))
            add_param(groups, "albedo",    "uv_albedo",    pbr_lr)
            add_param(groups, "roughness", "uv_roughness", pbr_lr)
            add_param(groups, "metallic",  "uv_metallic",  pbr_lr)
        else:
            raise ValueError("stage must be 'base', 'pbr', or 'full'")

        if len(groups) == 0:
            raise RuntimeError("[training_setup] No trainable parameters collected.")

        opt = torch.optim.Adam(groups, lr=0.0, eps=adam_eps)
        self.optimizer = opt

        sched_fn = cfg_train.get("lr_scheduler", None)
        self.scheduler = sched_fn(opt) if callable(sched_fn) else None