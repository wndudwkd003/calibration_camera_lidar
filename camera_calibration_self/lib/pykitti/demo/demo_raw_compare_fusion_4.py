import cv2
import numpy as np
from camera_calibration_self.lib import pykitti
from cv2 import ximgproc

# KITTI 데이터셋 경로 설정
basedir = 'D:/Datasets/KITTI'
date = '2011_09_26'
drive = '0001'
num_files = 108

# 데이터 로드
dataset = pykitti.raw(basedir, date, drive, frames=range(0, num_files))

# 스테레오 매칭 설정
# StereoSGBM 설정
minDisparity = 0
numDisparities = 16 * 5  # 16의 배수로 설정
blockSize = 5  # 홀수로 설정
P1 = 8 * 3 * blockSize ** 2
P2 = 32 * 3 * blockSize ** 2
disp12MaxDiff = 1
uniquenessRatio = 15
speckleWindowSize = 0
speckleRange = 2
preFilterCap = 63
mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY

stereo = cv2.StereoSGBM_create(minDisparity=minDisparity,
                               numDisparities=numDisparities,
                               blockSize=blockSize,
                               P1=P1,
                               P2=P2,
                               disp12MaxDiff=disp12MaxDiff,
                               uniquenessRatio=uniquenessRatio,
                               speckleWindowSize=speckleWindowSize,
                               speckleRange=speckleRange,
                               preFilterCap=preFilterCap,
                               mode=mode)

current_idx = 0


def update_fig(idx):
    # 데이터 로드
    rgb = np.array(dataset.get_rgb(idx)[0])

    first_rgb = dataset.get_rgb(idx)
    gray_left = cv2.cvtColor(np.array(first_rgb[0]), cv2.COLOR_RGB2GRAY)
    gray_right = cv2.cvtColor(np.array(first_rgb[1]), cv2.COLOR_RGB2GRAY)

    # 시차맵 생성
    disp_rgb = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    disp_rgb_8 = np.uint8(cv2.normalize(disp_rgb, None, 0, 255, cv2.NORM_MINMAX))
    disp_rgb_color = cv2.applyColorMap(disp_rgb_8, cv2.COLORMAP_JET)

    # WLS 필터 적용
    wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
    wls_filter.setLambda(8000)
    wls_filter.setSigmaColor(1.5)

    # 필터링된 깊이 맵 생성
    disp_rgb_16 = np.int16(cv2.normalize(disp_rgb, None, 0, 255, cv2.NORM_MINMAX))
    filtered_disp = wls_filter.filter(disp_rgb_16, gray_left, None, None)
    filtered_disp_vis = cv2.normalize(filtered_disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    filtered_disp_color = cv2.applyColorMap(np.uint8(filtered_disp_vis), cv2.COLORMAP_JET)

    # LiDAR 데이터 로드 및 투영
    velo = dataset.get_velo(idx)
    velo[:, 3] = 1  # 동차 좌표 추가
    velo_image = dataset.calib.T_cam2_velo.dot(velo.T)
    velo_image = dataset.calib.P_rect_20.dot(velo_image)
    velo_image /= velo_image[2, :]  # 동차 좌표 정규화

    # 거리(깊이)에 따른 색상 매핑
    max_depth = np.max(velo[:, 2])  # Z 좌표가 거리 정보
    min_depth = np.min(velo[:, 2])
    depth = velo[:, 2]

    # 거리를 0에서 255 사이의 값으로 정규화
    normalized_depth = ((depth - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    color_map = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)

    # 이미지에 LiDAR 데이터 표시
    for i in range(velo_image.shape[1]):
        x, y = int(velo_image[0, i]), int(velo_image[1, i])
        if x >= 0 and y >= 0 and x < rgb.shape[1] and y < rgb.shape[0]:
            cv2.circle(rgb, (x, y), 2, (0, 0, 255), -1)  # 빨간색으로 표시



    # 이미지에 LiDAR 데이터 표시
    for i, point in enumerate(velo_image.T):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < rgb.shape[1] and 0 <= y < rgb.shape[0]:
            # 거리에 따른 색상 적용
            color = color_map[i]
            cv2.circle(rgb, (x, y), 2, (int(color[0]), int(color[1]), int(color[2])), -1)

    # 이미지 표시
    cv2.imshow('LiDAR on RGB Image', rgb)
    cv2.imshow('Stereo Disparity Map', disp_rgb_color)
    cv2.imshow('Filtered Stereo Disparity Map', filtered_disp_color)

while True:
    update_fig(current_idx)
    key = cv2.waitKey(0)
    current_idx = (current_idx + 1) % len(dataset.frames)
    if key in [ord('q'), ord('Q')]:  # 'q' 키 코드
        break

cv2.destroyAllWindows()
