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
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.cubemap_to_fisheye import cubemap_to_fisheye, create_mapping_cache


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, return_depth=False, return_opacity=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    opacity_map = None

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    result = {"render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii}

    if return_depth:
        result["depth"] = depth
    if return_opacity and opacity_map is not None:  # None 체크 추가
        result["opacity"] = opacity_map
        
    return result


def render_cubemap(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0, override_color=None):
    # 매번 새로 생성 (메모리 절약)
    cubemap_cameras = viewpoint_camera.create_cubemap_cameras()
    
    cubemap_faces = []
    for i, cam in enumerate(cubemap_cameras):
        result = render(cam, pc, pipe, bg_color, scaling_modifier, override_color)
        cubemap_faces.append(result["render"])
    
    # 메모리 해제
    del cubemap_cameras
    
    return cubemap_faces


def render_fisheye(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                   scaling_modifier=1.0, override_color=None, mapping_cache=None,
                   return_cubemap=False):
    """
    Fisheye 카메라로 렌더링 (Cubemap → Fisheye 변환)
    
    Args:
        viewpoint_camera: FisheyeCamera 인스턴스
        pc: GaussianModel
        pipe: Pipeline parameters
        bg_color: Background color tensor
        scaling_modifier: Scaling modifier
        override_color: Override color
        mapping_cache: Cubemap→Fisheye 매핑 캐시 (속도 향상)
        return_cubemap: If True, also return cubemap faces
        
    Returns:
        dict: {
            "render": Fisheye 렌더링 이미지 [C, H, W],
            "cubemap_faces": (optional) 6개 cubemap face,
            "viewspace_points": for backward pass,
            "visibility_filter": for densification,
            "radii": for densification
        }
    """
    from scene.cameras import FisheyeCamera
    
    if not isinstance(viewpoint_camera, FisheyeCamera):
        raise ValueError("render_fisheye requires FisheyeCamera instance")
    
    # 1. Cubemap 렌더링 (6개 방향)
    cubemap_faces = render_cubemap(viewpoint_camera, pc, pipe, bg_color, 
                                    scaling_modifier, override_color)
    
    # 2. Cubemap → Fisheye 변환
    fisheye_image = cubemap_to_fisheye(
        cubemap_faces,
        height=viewpoint_camera.image_height,
        width=viewpoint_camera.image_width,
        fov=viewpoint_camera.fov,
        mapping_cache=mapping_cache
    )
    
    # 3. Backward를 위한 정보 수집
    # 여러 카메라의 정보를 합침 (첫 번째 카메라 기준)
    # Note: 이 부분은 학습 시 최적화가 필요할 수 있음
    first_cam = viewpoint_camera.cubemap_cameras[0]
    
    # Dummy screenspace points (실제로는 각 cubemap의 것들을 합쳐야 함)
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, 
                                          requires_grad=True, device="cuda")
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    # Visibility filter와 radii는 모든 cubemap face를 고려
    # 간단히 하기 위해 첫 번째 face의 것을 사용 (추후 개선 가능)
    result_dict = {
        "render": fisheye_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": torch.ones(pc.get_xyz.shape[0], dtype=torch.bool, device="cuda"),
        "radii": torch.ones(pc.get_xyz.shape[0], dtype=torch.int32, device="cuda")
    }
    
    if return_cubemap:
        result_dict["cubemap_faces"] = cubemap_faces
    
    return result_dict


def render_fisheye_optimized(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                             scaling_modifier=1.0, override_color=None, mapping_cache=None):
    """
    최적화된 Fisheye 렌더링
    """
    from scene.cameras import FisheyeCamera
    
    if not isinstance(viewpoint_camera, FisheyeCamera):
        raise ValueError("render_fisheye_optimized requires FisheyeCamera instance")
    
    cubemap_faces = []
    all_visibility_filters = []
    all_radii = []
    all_viewspace_points = []
    
    # 6개 방향 렌더링
    for cam in viewpoint_camera.cubemap_cameras:
        result = render(cam, pc, pipe, bg_color, scaling_modifier, override_color)
        
        cubemap_faces.append(result["render"])
        all_visibility_filters.append(result["visibility_filter"])
        all_radii.append(result["radii"])
        all_viewspace_points.append(result["viewspace_points"])
    
    # Cubemap → Fisheye 변환
    fisheye_image = cubemap_to_fisheye(
        cubemap_faces,
        height=viewpoint_camera.image_height,
        width=viewpoint_camera.image_width,
        fov=viewpoint_camera.fov,
        mapping_cache=mapping_cache
    )
    
    # Visibility와 radii 통합 (OR 연산 / max)
    combined_visibility = all_visibility_filters[0].clone()
    combined_radii = all_radii[0].clone()
    
    for i in range(1, 6):
        combined_visibility = combined_visibility | all_visibility_filters[i]
        combined_radii = torch.maximum(combined_radii, all_radii[i])
    
    # Viewspace points는 front face (+Z) 것을 사용
    primary_viewspace_points = all_viewspace_points[4]  # +Z face
    
    return {
        "render": fisheye_image,
        "viewspace_points": primary_viewspace_points,
        "visibility_filter": combined_visibility,
        "radii": combined_radii,
    }