import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib  # 새로 추가
from camera_calibration_self.lib import pykitti

# KITTI 데이터셋 경로 설정
basedir = 'D:/Datasets/KITTI'
date = '2011_09_26'
drive = '0001'
num_files = 108

# 데이터 로드
dataset = pykitti.raw(basedir, date, drive, frames=range(0, num_files))

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# 스테레오 매칭 설정
stereo = cv2.StereoBM_create()

def update_fig(idx):
    # 데이터 로드
    rgb = np.array(dataset.get_rgb(idx)[0])

    first_rgb = dataset.get_rgb(idx)
    disp_rgb = stereo.compute(
        cv2.cvtColor(np.array(first_rgb[0]), cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(np.array(first_rgb[1]), cv2.COLOR_RGB2GRAY))

    # LiDAR 데이터 로드 및 투영
    velo = dataset.get_velo(idx)
    velo[:, 3] = 1  # 동차 좌표 추가
    velo_image = dataset.calib.T_cam2_velo.dot(velo.T)
    velo_image = dataset.calib.P_rect_20.dot(velo_image)
    velo_image /= velo_image[2, :]  # 동차 좌표 정규화

    # 깊이에 따른 색상 매핑
    cmap = matplotlib.colormaps['jet']  # 수정된 부분
    max_depth = np.amax(velo_image[2, :])
    min_depth = np.amin(velo_image[2, :])

    if max_depth == min_depth:  # 모든 깊이 값이 같은 경우 처리
        max_depth = min_depth + 1

    # 이미지에 LiDAR 데이터 표시
    for i in range(velo_image.shape[1]):
        x, y, z = int(velo_image[0, i]), int(velo_image[1, i]), velo_image[2, i]
        if x >= 0 and y >= 0 and x < rgb.shape[1] and y < rgb.shape[0]:
            # 깊이 값을 0과 1 사이로 정규화
            normalized_depth = (z - min_depth) / (max_depth - min_depth)
            color = cmap(normalized_depth)[:3]  # RGBA에서 RGB로 변환
            rgb[y, x] = [int(c * 255) for c in color]

    # 이미지 표시
    ax[0].imshow(rgb)
    ax[0].set_title('LiDAR on RGB Image')
    ax[1].imshow(disp_rgb, cmap='viridis')
    ax[1].set_title('Stereo Disparity Map')
    plt.draw()

def on_key(event):
    if event.key == ' ':
        global current_idx
        current_idx = (current_idx + 1) % len(dataset.frames)
        update_fig(current_idx)
        plt.show()

# Connect the event handler and show the first image
current_idx = 0
update_fig(current_idx)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
