"""
Cubemap to Fisheye Image Conversion
====================================
6개의 cubemap face를 fisheye 이미지로 변환하는 유틸리티
"""

import torch
import numpy as np
import math


def create_fisheye_mapping(height, width, fov=180.0, device='cuda'):
    """
    Fisheye 이미지의 각 픽셀이 cubemap의 어느 위치를 참조해야 하는지 매핑 생성
    
    Args:
        height (int): Fisheye 이미지 높이
        width (int): Fisheye 이미지 너비
        fov (float): Field of view in degrees (default: 180)
        device (str): 'cuda' or 'cpu'
        
    Returns:
        face_idx (torch.Tensor): 각 픽셀이 참조할 cubemap face 인덱스 [H, W]
        u (torch.Tensor): cubemap face 내 u 좌표 [0, 1] [H, W]
        v (torch.Tensor): cubemap face 내 v 좌표 [0, 1] [H, W]
        valid_mask (torch.Tensor): 유효한 픽셀 마스크 [H, W]
    """
    # Fisheye 이미지의 중심 좌표
    cx = width / 2.0
    cy = height / 2.0
    
    # FOV를 라디안으로 변환
    fov_rad = math.radians(fov)
    
    # 반지름 (이미지 크기의 절반)
    radius = min(cx, cy)
    
    # 픽셀 좌표 그리드 생성
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    # 중심으로부터의 거리
    dx = x - cx
    dy = y - cy
    r = torch.sqrt(dx**2 + dy**2)
    
    # 유효한 픽셀 마스크 (반지름 내부)
    valid_mask = r <= radius
    
    # Fisheye 왜곡: r → theta (각도)
    # Equidistant projection: r = f * theta
    theta = (r / radius) * (fov_rad / 2.0)
    
    # Azimuth angle (phi)
    phi = torch.atan2(dy, dx)
    
    # 3D unit vector 계산
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    
    # 3D 방향 벡터 (fisheye → 3D ray)
    vec_x = sin_theta * cos_phi
    vec_y = sin_theta * sin_phi
    vec_z = cos_theta
    
    # Cubemap face 결정 (가장 큰 절댓값을 가진 축)
    abs_x = torch.abs(vec_x)
    abs_y = torch.abs(vec_y)
    abs_z = torch.abs(vec_z)
    
    max_axis = torch.maximum(torch.maximum(abs_x, abs_y), abs_z)
    
    # Face index 초기화
    face_idx = torch.zeros_like(r, dtype=torch.long)
    
    # UV 좌표 초기화
    u = torch.zeros_like(r)
    v = torch.zeros_like(r)
    
    # Cubemap face 매핑
    # 0: +X (right), 1: -X (left), 2: +Y (up), 3: -Y (down), 4: +Z (front), 5: -Z (back)
    
    # +X face (right)
    mask = (abs_x >= abs_y) & (abs_x >= abs_z) & (vec_x > 0)
    face_idx[mask] = 0
    u[mask] = (-vec_z[mask] / vec_x[mask] + 1.0) / 2.0
    v[mask] = (-vec_y[mask] / vec_x[mask] + 1.0) / 2.0
    
    # -X face (left)
    mask = (abs_x >= abs_y) & (abs_x >= abs_z) & (vec_x < 0)
    face_idx[mask] = 1
    u[mask] = (vec_z[mask] / -vec_x[mask] + 1.0) / 2.0
    v[mask] = (-vec_y[mask] / -vec_x[mask] + 1.0) / 2.0
    
    # +Y face (up)
    mask = (abs_y >= abs_x) & (abs_y >= abs_z) & (vec_y > 0)
    face_idx[mask] = 2
    u[mask] = (vec_x[mask] / vec_y[mask] + 1.0) / 2.0
    v[mask] = (vec_z[mask] / vec_y[mask] + 1.0) / 2.0
    
    # -Y face (down)
    mask = (abs_y >= abs_x) & (abs_y >= abs_z) & (vec_y < 0)
    face_idx[mask] = 3
    u[mask] = (vec_x[mask] / -vec_y[mask] + 1.0) / 2.0
    v[mask] = (-vec_z[mask] / -vec_y[mask] + 1.0) / 2.0
    
    # +Z face (front)
    mask = (abs_z >= abs_x) & (abs_z >= abs_y) & (vec_z > 0)
    face_idx[mask] = 4
    u[mask] = (vec_x[mask] / vec_z[mask] + 1.0) / 2.0
    v[mask] = (-vec_y[mask] / vec_z[mask] + 1.0) / 2.0
    
    # -Z face (back)
    mask = (abs_z >= abs_x) & (abs_z >= abs_y) & (vec_z < 0)
    face_idx[mask] = 5
    u[mask] = (-vec_x[mask] / -vec_z[mask] + 1.0) / 2.0
    v[mask] = (-vec_y[mask] / -vec_z[mask] + 1.0) / 2.0
    
    return face_idx, u, v, valid_mask


def sample_cubemap(cubemap_faces, face_idx, u, v, valid_mask):
    """
    Cubemap에서 bilinear interpolation으로 샘플링
    
    Args:
        cubemap_faces (list of torch.Tensor): 6개의 cubemap face [C, H, W]
        face_idx (torch.Tensor): Face 인덱스 [H, W]
        u (torch.Tensor): U 좌표 [H, W]
        v (torch.Tensor): V 좌표 [H, W]
        valid_mask (torch.Tensor): 유효 마스크 [H, W]
        
    Returns:
        torch.Tensor: Fisheye 이미지 [C, H, W]
    """
    device = face_idx.device
    C = cubemap_faces[0].shape[0]
    H, W = face_idx.shape
    face_h, face_w = cubemap_faces[0].shape[1], cubemap_faces[0].shape[2]
    
    # 출력 이미지 초기화
    output = torch.zeros(C, H, W, device=device)
    
    # 각 face별로 샘플링
    for i in range(6):
        mask = (face_idx == i) & valid_mask
        
        if mask.sum() == 0:
            continue
            
        # UV 좌표를 픽셀 좌표로 변환
        u_pixel = u[mask] * (face_w - 1)
        v_pixel = v[mask] * (face_h - 1)
        
        # Bilinear interpolation
        u0 = torch.floor(u_pixel).long()
        u1 = torch.ceil(u_pixel).long()
        v0 = torch.floor(v_pixel).long()
        v1 = torch.ceil(v_pixel).long()
        
        # Clamp to valid range
        u0 = torch.clamp(u0, 0, face_w - 1)
        u1 = torch.clamp(u1, 0, face_w - 1)
        v0 = torch.clamp(v0, 0, face_h - 1)
        v1 = torch.clamp(v1, 0, face_h - 1)
        
        # Interpolation weights
        wu = u_pixel - u0.float()
        wv = v_pixel - v0.float()
        
        # Get pixel values
        face = cubemap_faces[i]
        
        p00 = face[:, v0, u0]  # [C, N]
        p01 = face[:, v0, u1]
        p10 = face[:, v1, u0]
        p11 = face[:, v1, u1]
        
        # Bilinear interpolation
        wu = wu.unsqueeze(0)  # [1, N]
        wv = wv.unsqueeze(0)  # [1, N]
        
        sampled = (1 - wu) * (1 - wv) * p00 + \
                  wu * (1 - wv) * p01 + \
                  (1 - wu) * wv * p10 + \
                  wu * wv * p11
        
        # 출력에 할당
        output[:, mask] = sampled
    
    return output


def cubemap_to_fisheye(cubemap_faces, height, width, fov=180.0, mapping_cache=None):
    """
    Main function: Cubemap을 Fisheye 이미지로 변환
    
    Args:
        cubemap_faces (list of torch.Tensor): 6개의 cubemap face [C, H, W]
            Order: [+X, -X, +Y, -Y, +Z, -Z]
        height (int): 출력 fisheye 이미지 높이
        width (int): 출력 fisheye 이미지 너비
        fov (float): Field of view in degrees
        mapping_cache (dict, optional): 매핑 캐시 (속도 향상)
        
    Returns:
        torch.Tensor: Fisheye 이미지 [C, H, W]
    """
    device = cubemap_faces[0].device
    
    # 매핑이 캐시되어 있지 않으면 생성
    if mapping_cache is None:
        face_idx, u, v, valid_mask = create_fisheye_mapping(height, width, fov, device)
    else:
        face_idx = mapping_cache['face_idx']
        u = mapping_cache['u']
        v = mapping_cache['v']
        valid_mask = mapping_cache['valid_mask']
    
    # Cubemap에서 샘플링
    fisheye_image = sample_cubemap(cubemap_faces, face_idx, u, v, valid_mask)
    
    return fisheye_image


def create_mapping_cache(height, width, fov=180.0, device='cuda'):
    """
    매핑을 미리 계산해서 캐시로 저장 (학습 시 속도 향상)
    
    Args:
        height (int): Fisheye 이미지 높이
        width (int): Fisheye 이미지 너비
        fov (float): Field of view in degrees
        device (str): 'cuda' or 'cpu'
        
    Returns:
        dict: 매핑 캐시
    """
    face_idx, u, v, valid_mask = create_fisheye_mapping(height, width, fov, device)
    
    return {
        'face_idx': face_idx,
        'u': u,
        'v': v,
        'valid_mask': valid_mask
    }


# 테스트 코드
if __name__ == "__main__":
    print("Testing cubemap to fisheye conversion...")
    
    # 테스트용 cubemap 생성 (각 face를 다른 색으로)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    face_size = 512
    
    cubemap_faces = []
    colors = [
        [1.0, 0.0, 0.0],  # +X: Red
        [0.0, 1.0, 0.0],  # -X: Green
        [0.0, 0.0, 1.0],  # +Y: Blue
        [1.0, 1.0, 0.0],  # -Y: Yellow
        [1.0, 0.0, 1.0],  # +Z: Magenta
        [0.0, 1.0, 1.0],  # -Z: Cyan
    ]
    
    for color in colors:
        face = torch.ones(3, face_size, face_size, device=device)
        for c in range(3):
            face[c] *= color[c]
        cubemap_faces.append(face)
    
    # Fisheye 변환
    print("Converting cubemap to fisheye...")
    fisheye_height = 1024
    fisheye_width = 1024
    
    # 매핑 캐시 생성
    cache = create_mapping_cache(fisheye_height, fisheye_width, fov=180.0, device=device)
    
    # 변환
    fisheye = cubemap_to_fisheye(cubemap_faces, fisheye_height, fisheye_width, mapping_cache=cache)
    
    print(f"Output shape: {fisheye.shape}")
    print("Conversion successful!")
    
    # 이미지 저장 (optional)
    try:
        from torchvision.utils import save_image
        save_image(fisheye, 'test_fisheye.png')
        print("Saved test image to test_fisheye.png")
    except:
        print("Could not save image (torchvision not available)")