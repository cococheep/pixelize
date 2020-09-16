import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True, help='path to image file')
parser.add_argument('--dsize', required=True, help='dot size')
parser.add_argument('--color', required=True, help='number of color')
args = parser.parse_args()
image_path = args.image
dsize = int(args.dsize)
color = int(args.color)

# image 불러오기
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# -------------------픽셀 dot화-----------------------
# dot size 설정
dot_size = dsize

# 줄였다 늘려서 dot_size로 모자이크
image = cv2.resize(image, dsize=(0, 0), fx=1.0 / dot_size, fy=1.0 / dot_size, interpolation=cv2.INTER_AREA)
image = cv2.resize(image, dsize=(0, 0), fx=1.0 * dot_size, fy=1.0 * dot_size, interpolation=cv2.INTER_NEAREST)

# -----------------색상 줄이기--------------------------
# number of color 설정
num_color = color
flag = 256 / num_color

# 명도로 나누기 때문에 우선 명도값 추출
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# 초기값 생성
mask = np.ndarray((num_color,) + v.shape, dtype=v.dtype)                # mask는 명도에 따라 분리되는 각 영역
bias = np.ndarray((num_color, 3), dtype=image.dtype)                    # bias는 각 mask가 채워질 색
fill = np.ndarray((3,) + mask[0].shape, dtype=mask[0].dtype)            # fill은 bias으로 채워진 각 채널 (b,g,r)
img = np.ndarray((num_color,) + image.shape, dtype=image.dtype)         # img는 fill 채널들이 합쳐진 것
final_img = np.zeros(img[0].shape, dtype=img.dtype)                     # 최종 image

for i in range(num_color):
    bias[i] = np.random.randint(low=flag * i, high=(flag * i) + flag - 1, size=3)

for i in range(num_color):
    mask[i] = cv2.inRange(v, flag * i, flag * (i+1) - 1)                # mask를 flag 기준으로 나눔
    for j in range(3):                                                  # 채널 수 만큼 반복
        fill[j] = mask[i] // 255 * bias[i][j]                           # 각 bgr채널을 지정한 색으로 채움
    img[i] = np.stack([fill[0], fill[1], fill[2]], axis=2)              # bgr채널을 모두 합쳐 각 영역을 생성
    final_img += img[i]                                                 # 각 영역을 모두 합쳐 이미지 생성

# result
cv2.imshow("final", final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
