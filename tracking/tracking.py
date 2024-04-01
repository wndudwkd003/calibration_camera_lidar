import cv2
import numpy as np

# 마우스 콜백 함수
def select_object(event, x, y, flags, param):
    global point, point_selected, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)

# 화면 크기 설정
frame_width, frame_height = 640, 480

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_object)

point_selected = False
point = ()
old_points = np.array([[]])

# 칼만 필터 설정
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

# 키보드로 포인터를 이동시키기 위한 초기화
speed = 15
key_map = {
    ord('w'): (0, -speed),
    ord('s'): (0, speed),
    ord('a'): (-speed, 0),
    ord('d'): (speed, 0)
}

while True:
    frame = np.zeros((frame_height, frame_width, 3), np.uint8)

    if point_selected:
        cv2.circle(frame, point, 5, (0, 0, 255), 2)

        # 칼만 필터 업데이트
        measurement = np.array([[np.float32(point[0])], [np.float32(point[1])]])
        kalman.correct(measurement)
        prediction = kalman.predict()

        cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 5, (255, 0, 0), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC 키
        break
    elif key in key_map:
        if point_selected:
            dx, dy = key_map[key]
            point = (point[0] + dx, point[1] + dy)

cv2.destroyAllWindows()