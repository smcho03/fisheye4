"""
Fisheye 변환 테스트 스크립트
"""
import torch
import numpy as np
from utils.cubemap_to_fisheye import (
    create_mapping_cache, 
    cubemap_to_fisheye, 
    save_debug_images
)
from torchvision.utils import save_image


def create_test_cubemap(face_size=512, device='cuda'):
    """테스트용 cubemap 생성 - 각 face를 다른 색으로"""
    colors = [
        [1.0, 0.0, 0.0],  # +X: Red
        [0.0, 1.0, 0.0],  # -X: Green
        [0.0, 0.0, 1.0],  # +Y: Blue
        [1.0, 1.0, 0.0],  # -Y: Yellow
        [1.0, 0.0, 1.0],  # +Z: Magenta (front)
        [0.0, 1.0, 1.0],  # -Z: Cyan (back)
    ]
    
    cubemap_faces = []
    for color in colors:
        face = torch.ones(3, face_size, face_size, device=device)
        for c in range(3):
            face[c] *= color[c]
        
        # 방향 표시를 위한 그라데이션 추가
        x = torch.linspace(0, 1, face_size, device=device)
        y = torch.linspace(0, 1, face_size, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # 밝기 변화 추가
        face = face * (0.5 + 0.5 * xx.unsqueeze(0))
        
        cubemap_faces.append(face)
    
    return cubemap_faces


def test_fisheye_conversion():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 테스트 cubemap 생성
    face_size = 512
    cubemap_faces = create_test_cubemap(face_size, device)
    
    # Fisheye 변환
    fisheye_h, fisheye_w = 1024, 1024
    fov = 180.0
    
    print("Creating mapping cache...")
    cache = create_mapping_cache(fisheye_h, fisheye_w, fov, device)
    
    print("Converting cubemap to fisheye...")
    fisheye = cubemap_to_fisheye(cubemap_faces, fisheye_h, fisheye_w, fov, cache)
    
    print(f"Fisheye shape: {fisheye.shape}")
    
    # 결과 저장
    save_debug_images(cubemap_faces, fisheye, "debug_fisheye_test")
    
    print("\n=== Expected colors in fisheye ===")
    print("Center: Magenta (+Z, front)")
    print("Right edge: Red (+X)")
    print("Left edge: Green (-X)")
    print("Top edge: Blue (+Y)")
    print("Bottom edge: Yellow (-Y)")
    print("Outer ring (if FOV > 180): Cyan (-Z, back)")


if __name__ == "__main__":
    test_fisheye_conversion()