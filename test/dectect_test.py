import cv2
import copy
import time
import argparse
import os

from pupil_apriltags import Detector

os.add_dll_directory("C:/Users/ymail/.conda/envs/pt2/lib/site-packages/pupil_apriltags.libs")

device = 1
cap_width = 960
cap_height = 540
families = 'tag36h11'
nthreads = 1
quad_decimate = 2.0
quad_sigma = 0.0
refine_edges = 1
decode_sharpening = 0.25
debug = 0


def main():
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # 가우시안 필터에 사용될 값
    block_size = 91  # Threshold 적용할 영역 사이즈, 홀수
    C = 7  # 계산된 경계 값에서 차감할 값

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] can't execute camera!")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 가우시안
        # threshold = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C)

        # OTSU 임계값
        _, threshold = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        cv2.imshow("threshold result", threshold)

        # CHAIN_APPROX_NONE: 모든 컨투어 포인트를 반환, 자세한 포인트
        # CHAIN_APPROX_SIMPLE: 꼭짓점 포인트만 반환, 메모리 절약
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if 100 < w * h < 10000:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("contours result", frame)

        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:
            break


if __name__ == "__main__":
    main()
