import sys
import cv2
import os 
import csv 
import mediapipe as mp
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = 'C:\Potenup\Korean-Sign-Language-Project'
sys.path.append(project_root)

from utils.guide_box import draw_box

# box 데이터 프레임 불러오기
guide_box_df = pd.read_csv("C:\Potenup\Korean-Sign-Language-Project\data\guide_box.csv")
sign_code_df = pd.read_csv("C:\Potenup\Korean-Sign-Language-Project\data\sign_code.csv")
# mediapipe의 Hand Landmark 를 추출을 위한 옵션
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode = False, #고정이미지 아님
    max_num_hands = 2,

    min_detection_confidence = 0.5, #감지 확률 0.5 이상만
    min_tracking_confidence = 0.5 # 트래킹 확률 0.5이상만
)

pose = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

##############################################
######### 🚨 여기를 수정하면 됩니다! 🚨 ########
# 저장할 데이터 설정 
answer_label = 0 # 저장할 라벨을 적어주세요
answer_text = (
    sign_code_df.loc[sign_code_df['label'] == answer_label, 'sign_text']
    .squeeze() if (sign_code_df['label'] == answer_label).any() else None
)
print("========================================")
print(f'{answer_text} 를 저장하기 시작합니다!')
print("========================================")
# s 키를 누르면 저장됩니다!
file_path = f'C:/Potenup/Korean-Sign-Language-Project/data/sign_data.csv'
######### 🚨 여기를 수정하면 됩니다! 🚨 ########
##############################################

# 파일이 없을 경우 생성
if not os.path.exists(file_path):
    with open(file_path, "w") as file:
        writer = csv.writer(file)

vcap = cv2.VideoCapture(0)

count = 0
while True:
    ret, frame = vcap.read()
    if not ret:
        print("웹캠이 작동하지 않습니다.")
        sys.exit()
    
    # 좌우반전
    frame = cv2.flip(frame, 1)

    draw_box(frame, guide_box_df, answer_label)

    # 그리기 설정
    frame.flags.writeable = True
    
    # 저장 데이터 준비    
    result_landmarks = {"Left" : [], 'Right': [], "Face": []}
    data_count = {'Left' : 0, 'Right' : 0, 'Face': 0}

    ###### Pose Landmark 그리기 ######
    pose_results = pose.process(frame)
    if pose_results.pose_landmarks:
        data_count['Face'] += 1
        pose_landmarks = pose_results.pose_landmarks.landmark
        height, width, _ = frame.shape

        pose_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for index, landmark in enumerate(pose_landmarks):
            if index in pose_points:
                point_x = int(landmark.x * width)
                point_y = int(landmark.y * height)

                cv2.circle(frame, (point_x, point_y), 3, (0,0,255), 2)
                result_landmarks['Face'].extend([landmark.x, landmark.y])
    
    ###### Hands Landmark 설정하기 ########
    # 손 감지하기
    hand_results = hands.process(frame)

    # 그리기
    if hand_results.multi_hand_landmarks:        
        for hand in hand_results.multi_handedness:
            data_count[hand.classification[0].label] += 1

        for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            label = hand_results.multi_handedness[i].classification[0].label

            # 좌표 모으기
            for landmark in hand_landmarks.landmark:
                result_landmarks[label].extend([landmark.x, landmark.y])

            # 자동 그리기
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    ready_to_save = False
    if  data_count['Face'] == 1 and data_count['Left'] == 1 and data_count['Right'] == 1:
        ready_to_save = True
    if  data_count['Face'] == 1 and data_count['Left'] == 0 and data_count['Right'] == 1:
        ready_to_save = True
    if  data_count['Face'] == 1 and data_count['Left'] == 1 and data_count['Right'] == 0:
        ready_to_save = True

    if ready_to_save:
        cv2.putText(frame, "Ready to save!", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
        
        # 왼손이 없는 경우
        if data_count['Left'] == 0:
            result_landmarks['Left'] = [0] * 42
    
        # 오른손이 없는 경우
        if data_count['Right'] > 1:
            result_landmarks['Right'] = [0] * 42

        key = cv2.waitKey(1) # ASCII 코드
        if key == ord("s"):
            result = [answer_label]
            result.extend(result_landmarks['Left'])
            result.extend(result_landmarks['Right'])
            result.extend(result_landmarks['Face'])
        
            with open(file_path, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(result)
                cv2.putText(frame, "Save Data!", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

                count += 1
                print('CSV 저장 완료! ', count)
    
    # 화면 띄우기
    cv2.imshow("webcam", frame)

    # 꺼지는 조건
    key = cv2.waitKey(1)
    if key == 27:
        break

vcap.release()
cv2.destroyAllWindows()