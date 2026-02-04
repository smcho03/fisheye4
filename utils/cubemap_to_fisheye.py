"""
Cubemap to Fisheye Image Conversion (Fixed Version)
"""

import torch
import numpy as np
import math


def create_fisheye_mapping(height, width, fov=180.0, device='cuda'):
    """
    Fisheye 이미지의 각 픽셀이 cubemap의 어느 위치를 참조해야 하는지 매핑 생성
    
    Coordinate system:
    - Fisheye center looks along +Z axis
    - X is right, Y is down (image coordinates), but we convert to Y up for 3D
    """
    cx = width / 2.0
    cy = height / 2.0
    fov_rad = math.radians(fov)
    radius = min(cx, cy)
    
    # 픽셀 좌표 그리드 생성
    y_img, x_img = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    # 중심으로부터의 거리 (이미지 좌표계)
    dx = x_img - cx  # 오른쪽이 +
    dy = y_img - cy  # 아래쪽이 + (이미지 좌표계)
    r = torch.sqrt(dx**2 + dy**2)
    
    # 유효한 픽셀 마스크
    valid_mask = r <= radius
    
    # r=0일 때 division by zero 방지
    r_safe = torch.clamp(r, min=1e-8)
    
    # Equidistant fisheye projection: r = f * theta
    theta = (r / radius) * (fov_rad / 2.0)  # 중심에서 가장자리로 0 ~ fov/2
    
    # Azimuth angle (이미지 평면에서의 각도)
    phi = torch.atan2(dy, dx)  # -pi to pi
    
    # 3D 방향 벡터 계산
    # Fisheye 카메라가 +Z 방향을 바라본다고 가정
    # theta=0 -> (0,0,1), theta=pi/2 -> 수평 방향
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    # 3D direction vector
    # X: right, Y: up (3D), Z: forward
    vec_x = sin_theta * torch.cos(phi)
    vec_y = -sin_theta * torch.sin(phi)  # 이미지 Y와 3D Y는 반대
    vec_z = cos_theta
    
    # Cubemap face 결정
    abs_x = torch.abs(vec_x)
    abs_y = torch.abs(vec_y)
    abs_z = torch.abs(vec_z)
    
    face_idx = torch.zeros_like(r, dtype=torch.long)
    u = torch.zeros_like(r)
    v = torch.zeros_like(r)
    
    # Face 0: +X (right)
    mask = (abs_x >= abs_y) & (abs_x >= abs_z) & (vec_x > 0)
    if mask.any():
        ma = vec_x[mask]
        face_idx[mask] = 0
        u[mask] = (-vec_z[mask] / ma + 1.0) / 2.0
        v[mask] = (-vec_y[mask] / ma + 1.0) / 2.0
    
    # Face 1: -X (left)
    mask = (abs_x >= abs_y) & (abs_x >= abs_z) & (vec_x <= 0)
    if mask.any():
        ma = -vec_x[mask]
        ma = torch.clamp(ma, min=1e-8)
        face_idx[mask] = 1
        u[mask] = (vec_z[mask] / ma + 1.0) / 2.0
        v[mask] = (-vec_y[mask] / ma + 1.0) / 2.0
    
    # Face 2: +Y (up)
    mask = (abs_y > abs_x) & (abs_y >= abs_z) & (vec_y > 0)
    if mask.any():
        ma = vec_y[mask]
        face_idx[mask] = 2
        u[mask] = (vec_x[mask] / ma + 1.0) / 2.0
        v[mask] = (vec_z[mask] / ma + 1.0) / 2.0
    
    # Face 3: -Y (down)
    mask = (abs_y > abs_x) & (abs_y >= abs_z) & (vec_y <= 0)
    if mask.any():
        ma = -vec_y[mask]
        ma = torch.clamp(ma, min=1e-8)
        face_idx[mask] = 3
        u[mask] = (vec_x[mask] / ma + 1.0) / 2.0
        v[mask] = (-vec_z[mask] / ma + 1.0) / 2.0
    
    # Face 4: +Z (front) - fisheye가 바라보는 방향
    mask = (abs_z > abs_x) & (abs_z > abs_y) & (vec_z > 0)
    if mask.any():
        ma = vec_z[mask]
        face_idx[mask] = 4
        u[mask] = (vec_x[mask] / ma + 1.0) / 2.0
        v[mask] = (-vec_y[mask] / ma + 1.0) / 2.0
    
    # Face 5: -Z (back)
    mask = (abs_z > abs_x) & (abs_z > abs_y) & (vec_z <= 0)
    if mask.any():
        ma = -vec_z[mask]
        ma = torch.clamp(ma, min=1e-8)
        face_idx[mask] = 5
        u[mask] = (-vec_x[mask] / ma + 1.0) / 2.0
        v[mask] = (-vec_y[mask] / ma + 1.0) / 2.0
    
    # UV 범위 클램핑
    u = torch.clamp(u, 0.0, 1.0)
    v = torch.clamp(v, 0.0, 1.0)
    
    return face_idx, u, v, valid_mask


def sample_cubemap(cubemap_faces, face_idx, u, v, valid_mask):
    """
    Cubemap에서 bilinear interpolation으로 샘플링
    """
    device = face_idx.device
    C = cubemap_faces[0].shape[0]
    H, W = face_idx.shape
    
    # 출력 이미지 초기화 (검은색)
    output = torch.zeros(C, H, W, device=device)
    
    for i in range(6):
        mask = (face_idx == i) & valid_mask
        
        if mask.sum() == 0:
            continue
        
        face = cubemap_faces[i]
        face_h, face_w = face.shape[1], face.shape[2]
        
        # UV를 픽셀 좌표로 변환
        u_pixel = u[mask] * (face_w - 1)
        v_pixel = v[mask] * (face_h - 1)
        
        # Bilinear interpolation 좌표
        u0 = torch.floor(u_pixel).long()
        u1 = (u0 + 1).clamp(max=face_w - 1)
        v0 = torch.floor(v_pixel).long()
        v1 = (v0 + 1).clamp(max=face_h - 1)
        
        u0 = u0.clamp(0, face_w - 1)
        v0 = v0.clamp(0, face_h - 1)
        
        # Interpolation weights
        wu = u_pixel - u0.float()
        wv = v_pixel - v0.float()
        
        # 4개 코너 픽셀 값
        p00 = face[:, v0, u0]
        p01 = face[:, v0, u1]
        p10 = face[:, v1, u0]
        p11 = face[:, v1, u1]
        
        # Bilinear interpolation
        wu = wu.unsqueeze(0)
        wv = wv.unsqueeze(0)
        
        sampled = (1 - wu) * (1 - wv) * p00 + \
                  wu * (1 - wv) * p01 + \
                  (1 - wu) * wv * p10 + \
                  wu * wv * p11
        
        output[:, mask] = sampled
    
    return output


def cubemap_to_fisheye(cubemap_faces, height, width, fov=180.0, mapping_cache=None):
    """
    Cubemap을 Fisheye 이미지로 변환
    
    Args:
        cubemap_faces: 6개의 cubemap face [C, H, W]
            Order: [+X, -X, +Y, -Y, +Z, -Z]
        height, width: 출력 fisheye 이미지 크기
        fov: Field of view (degrees)
        mapping_cache: 미리 계산된 매핑 (속도 향상용)
    """
    device = cubemap_faces[0].device
    
    if mapping_cache is None:
        face_idx, u, v, valid_mask = create_fisheye_mapping(height, width, fov, device)
    else:
        face_idx = mapping_cache['face_idx']
        u = mapping_cache['u']
        v = mapping_cache['v']
        valid_mask = mapping_cache['valid_mask']
    
    fisheye_image = sample_cubemap(cubemap_faces, face_idx, u, v, valid_mask)
    
    return fisheye_image


def create_mapping_cache(height, width, fov=180.0, device='cuda'):
    """매핑 캐시 생성"""
    face_idx, u, v, valid_mask = create_fisheye_mapping(height, width, fov, device)
    
    return {
        'face_idx': face_idx,
        'u': u,
        'v': v,
        'valid_mask': valid_mask
    }


def save_debug_images(cubemap_faces, fisheye_image, output_dir="debug_fisheye"):
    """디버깅용 이미지 저장"""
    import os
    from torchvision.utils import save_image
    
    os.makedirs(output_dir, exist_ok=True)
    
    face_names = ['pos_x', 'neg_x', 'pos_y', 'neg_y', 'pos_z', 'neg_z']
    for i, (face, name) in enumerate(zip(cubemap_faces, face_names)):
        save_image(face.clamp(0, 1), f"{output_dir}/cubemap_{i}_{name}.png")
    
    save_image(fisheye_image.clamp(0, 1), f"{output_dir}/fisheye_result.png")
    print(f"Debug images saved to {output_dir}/")