import cv2

def find_available_cameras(limit=10):
    available_cameras = []
    for i in range(limit):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

# 카메라 인덱스를 탐색할 최대 개수를 지정합니다. 시스템에 연결된 카메라의 수가 이보다 많지 않다고 예상되는 경우 값을 조정할 수 있습니다.
max_index_to_check = 10
available_cameras = find_available_cameras(max_index_to_check)
print(f"Available camera indexes: {available_cameras}")
