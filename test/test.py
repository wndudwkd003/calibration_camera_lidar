import matplotlib.pyplot as plt
import math

# find_perpendicular_line_endpoints 함수 정의
def find_perpendicular_line_endpoints(A, B, L):
    x1, y1 = A
    x2, y2 = B
    if x2 - x1 == 0:  # 선분 AB가 수직일 경우
        dx = L / 2
        dy = 0
    elif y2 - y1 == 0:  # 선분 AB가 수평일 경우
        dx = 0
        dy = L / 2
    else:
        slope = (y2 - y1) / (x2 - x1)
        angle = math.atan(-1 / slope)
        dx = (L / 2) * math.cos(angle)
        dy = (L / 2) * math.sin(angle)

    endpoint1 = (x2 + dx, y2 + dy)
    endpoint2 = (x2 - dx, y2 - dy)
    return endpoint1, endpoint2


# 시각화 함수 정의
def visualize_perpendicular_line(A, B, L):
    endpoint1, endpoint2 = find_perpendicular_line_endpoints(A, B, L)

    # 선분 AB 그리기
    plt.plot([A[0], B[0]], [A[1], B[1]], 'ro-')

    # 선분 B에서 수직인 선분 그리기
    plt.plot([endpoint1[0], endpoint2[0]], [endpoint1[1], endpoint2[1]], 'bo-')

    # 그래프 설정
    plt.axis('equal')  # x, y축의 비율을 동일하게 설정
    plt.grid(True)  # 격자 표시
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Visualization of Perpendicular Line')

    # 범례 추가
    plt.legend(['AB', 'Perpendicular Line'])

    plt.show()


# 예제 데이터
A = (-2, 4)
B = (2, 3)
L = 3

# 시각화 함수 호출
visualize_perpendicular_line(A, B, L)
