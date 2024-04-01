import time
import os

import cv2
import numpy as np
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

thickness = 2
radius = 5
length_magnification = 3

K = (718.0185818253857, 718.7782113904716, 323.4567247857622, 235.99239839273147)
dist = (0.09119377823619247, 0.0626265233941177, -0.007768343835168918, -0.004451209268144528, 0.48630799030494654)
mm = 22


def main():
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    detector = Detector(
        families=families,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=refine_edges,
        decode_sharpening=decode_sharpening,
        debug=debug,
    )

    elapsed_time = 0

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("[ERROR] can't execute camera!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tags = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=K,
            tag_size=mm,
        )

        result = draw_tag(frame, tags)

        cv2.putText(result,
                    "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                    cv2.LINE_AA)

        elapsed_time = time.time() - start_time

        cv2.imshow("result", result)

        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()


def draw_axes_for_tag(image, tag, point):
    # 회전 행렬을 회전 벡터로 변환
    camera_matrix = np.array([[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1]])
    dist_coeffs = np.array(dist).reshape(-1, 1)
    point_3d = np.array([point[0], point[1], 0.0])

    # 회전 행렬을 회전 벡터로 변환
    rvec, _ = cv2.Rodrigues(np.array(tag.pose_R))
    tvec = np.array(tag.pose_t)

    # 3D 모델 포인트: 태그 중심에서 각 축을 따라 일정 거리만큼 뻗어나간 점들
    axis_length = mm * length_magnification
    axis_points = np.float32([
        [0, 0, 0],  # 원점
        [axis_length, 0, 0],  # x축
        [0, -axis_length, 0],  # y축
        [0, 0, -axis_length]  # z축
    ]).reshape(-1, 3)


    # 3D 점을 2D 이미지 평면에 투영합니다.
    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)

    if not np.any(np.isnan(imgpts)) and not np.any(np.isinf(imgpts)):
        # imgpts에서 각 축 끝점을 정수로 변환합니다.
        imgpt0 = tuple(map(int, imgpts[0].ravel()))
        imgpt1 = tuple(map(int, imgpts[1].ravel()))
        imgpt2 = tuple(map(int, imgpts[2].ravel()))
        imgpt3 = tuple(map(int, imgpts[3].ravel()))

        cv2.line(image, imgpt0, imgpt1, (255, 0, 0),
                 thickness)  # x축: 빨간색
        cv2.line(image, imgpt0, imgpt2, (0, 255, 0),
                 thickness)  # y축: 초록색
        cv2.line(image, imgpt0, imgpt3, (0, 0, 255),
                 thickness)  # z축: 파란색

    return image


def draw_tag(image, tags):
    # 모든 태그에 대해 축을 그립니다.
    for tag in tags:
        tag_id = tag.tag_id
        center = tag.center
        corners = tag.corners

        center = (int(center[0]), int(center[1]))
        corner_01 = (int(corners[0][0]), int(corners[0][1]))
        corner_02 = (int(corners[1][0]), int(corners[1][1]))
        corner_03 = (int(corners[2][0]), int(corners[2][1]))
        corner_04 = (int(corners[3][0]), int(corners[3][1]))

        # 태그의 중심과 코너에 원과 선을 그립니다.
        cv2.circle(image, center, radius, (0, 0, 255), thickness)
        cv2.line(image, corner_01, corner_02, (255, 0, 0), thickness)
        cv2.line(image, corner_02, corner_03, (255, 0, 0), thickness)
        cv2.line(image, corner_03, corner_04, (0, 255, 0), thickness)
        cv2.line(image, corner_04, corner_01, (0, 255, 0), thickness)

        # 태그의 축을 그립니다.
        if tag_id == 0:
            draw_axes_for_tag(image, tag, corner_02)

        # 태그 ID를 화면에 표시합니다.
        cv2.putText(image, str(tag_id), (center[0] - 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                    2)

    return image


if __name__ == "__main__":
    main()
