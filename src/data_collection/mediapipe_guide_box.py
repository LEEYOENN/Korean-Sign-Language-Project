import sys
import cv2
import mediapipe as mp
import pandas
import csv

label = 19
# mediapipe의 Hand Landmark 를 추출을 위한 옵션
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode = False, #고정이미지 아님
    max_num_hands = 2,
    min_detection_confidence = 0.3, # 감지확률 0.3 이상
    min_tracking_confidence = 0.3 # 트래킹 확률 0.3 이상
)

vcap = cv2.VideoCapture(0)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"클릭한 좌표: ({x}, {y})")

cv2.namedWindow("webcam")
cv2.setMouseCallback("webcam", mouse_callback)

while True:
    ret, frame = vcap.read()
    if not ret:
        print("웹캠이 작동하지 않습니다.")
        sys.exit()

    # 좌우 반전
    flipped_frame = cv2.flip(frame, 1)
    
    ### Hand Landmark 설정하기 ###

    # 손 감지하기
    results = hands.process(flipped_frame)

    # 손 그리기
    if results.multi_hand_landmarks:
        #print(f"감지된 손 개수: {len(results.multi_hand_landmarks)}")

        for hand_landmarks in results.multi_hand_landmarks:
            height, width, _ = flipped_frame.shape
            # x_min, y_min, x_max, y_max를 초기화합니다.
            x_min, y_min = width, height
            x_max, y_max = 0, 0

            # 21개의 랜드마크를 순회하며 최소/최대 좌표를 찾습니다.
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * width), int(landmark.y * height)

                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y

                mp_drawing.draw_landmarks(
                        flipped_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            # 사각형을 그릴 좌표
            x1 = x_min - 12
            y1 = y_min - 12
            x2 = x_max + 12
            y2 = y_max + 12

            # 여기서 cv2.rectangle() 함수를 사용하여 사각형을 그릴 수 있습니다.
            cv2.rectangle(flipped_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

            # s키 입력 시 저장
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                with open("./data/guide_box.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([x1, y1, x2, y2, label])
                print("좌표가 CSV에 저장되었습니다:", [x1, y1, x2, y2, label])

            # q키 입력 시 종료
            if key == ord('q'):
                break          


    # 색보정
    #flipped_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
    
    #print(frame.shape)

    # 손 그리기 설정
    frame.flags.writeable = False

    # face_box
    cv2.putText(flipped_frame, "FACE", (228, 102), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.rectangle(flipped_frame, (228, 110), (372, 289), (208, 203, 128), 2)

    # number_box
    #cv2.putText(flipped_frame, "NUMBER", (411, 182), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    #cv2.rectangle(flipped_frame, (411, 190), (538, 375), (121, 190, 132), 2)

    # 예 box
    # cv2.putText(flipped_frame, "YES", (351, 162), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    # cv2.rectangle(flipped_frame, (351, 170), (489, 284), (121, 190, 132), 2)

    # 아니오 box 0, 1
    # cv2.putText(flipped_frame, "NO", (79, 304), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    # cv2.rectangle(flipped_frame, (79, 310), (218, 438), (121, 190, 132), 2)
    # cv2.rectangle(flipped_frame, (396, 310), (536, 438), (121, 190, 132), 2)

    # 화면 띄우기
    cv2.imshow("webcam", flipped_frame)

    # 꺼지는 조건
    key = cv2.waitKey(1)
    if key == 27:
        break

vcap.release()
cv2.destroyAllWindows()