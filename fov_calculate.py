# 간단 계산
import math
width = 3264
fx = 995.35
fov_x = 2 * math.atan(width / (2 * fx))
fov_degrees = math.degrees(fov_x)
print(f"Actual FOV: {fov_degrees:.1f} degrees")
# 결과: 약 110-120도 정도일 거예요