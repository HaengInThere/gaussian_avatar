import math
import torch
import torch.nn.functional as F
import numpy as np
from .r3dg_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def rasterize_feature(means3D, shs, color, features, opacity, scales, rotations, viewpoint_cam, scaling_modifier=1.0, bg_color=None, active_sh_degree=0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    means2D = screenspace_points

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)
    intrinsic = viewpoint_cam.intrinsics

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_cam.image_height),
        image_width=int(viewpoint_cam.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        # cx=float(viewpoint_cam['K'][0,2]),
        # cy=float(viewpoint_cam['K'][1,2]),
        cx=float(intrinsic[0, 2]),
        cy=float(intrinsic[1, 2]),

        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_cam.world_view_transform.cuda(),
        projmatrix=viewpoint_cam.full_proj_transform.cuda(),
        sh_degree=active_sh_degree,
        campos=viewpoint_cam.camera_center.cuda(),
        prefiltered=False,
        backward_geometry=True,
        computer_pseudo_normal=True,
        debug=False
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    (num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth,
     rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, weights, radii) = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=color,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
        features=features,
    )

    mask = num_contrib > 0
    rendered_feature = rendered_feature * mask
    return { "render": rendered_image,
            "depth": rendered_depth,
            "rendered_pseudo_normal": rendered_pseudo_normal,
            'feature':rendered_feature,
            "alpha": rendered_opacity,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,}


def render_base(viewpoint_cam, gaussian, background):
    means3D = gaussian.get_xyz # N 3 
    opacity = gaussian.get_opacity
    scales = gaussian.get_scaling
    color = gaussian._features_dc.squeeze()
    rotations = gaussian.get_rotation
    active_sh_degree = gaussian.get_activate_sh_degree

    viewdirs = F.normalize(viewpoint_cam.camera_center - means3D, dim=-1) # N 3 
    
    base_color, roughness, metallic, normal = gaussian.get_albedo, gaussian.get_roughness, gaussian.get_metallic, gaussian.get_normal 

    gs_feat = torch.cat([base_color, roughness, metallic, normal, viewdirs], -1) # 3, 1, 1, 3, 3
    outs = rasterize_feature(means3D, shs=None, color=color, features=gs_feat, opacity=opacity, scales=scales, rotations=rotations, viewpoint_cam=viewpoint_cam,
                                 scaling_modifier=1.0, bg_color=background, active_sh_degree=active_sh_degree)
    
    rendered_feature = outs['feature']
    base_color, roughness, metallic, normal_, r_view = rendered_feature.split([3, 1, 1, 3, 3], 0)

    render = outs["render"]
    depth = outs["depth"]
    rendered_pseudo_normal = outs["rendered_pseudo_normal"]
    alpha = outs["alpha"]
    viewspace_points = outs["viewspace_points"]
    visibility_filter = outs["visibility_filter"]
    radii = outs["radii"]

    output = {
        'render': render,
        'depth':depth,
        'rendered_pseudo_normal':rendered_pseudo_normal,
        'alpha':alpha,
        'viewspace_points':viewspace_points,
        'visibility_filter':visibility_filter,
        'radii': radii,   

        'base_color': base_color, # List
        'roughness': roughness, 
        'metallic':metallic,
        'normal': normal_,
        'r_view': r_view,

    }

    return output