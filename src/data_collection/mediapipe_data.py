import sys
import cv2
import os 
import csv 
import mediapipe as mp
import pandas as pd
from pandas.errors import EmptyDataError

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils import mediapipe_util
from utils.guide_box import draw_box

# mediapipe의 Hand Landmark 를 추출을 위한 옵션
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def recolor_style_dict(style_dict, bgr):
    """DrawingStyles(dict) 의 color만 변경한 복제본을 반환"""
    new_dict = {}
    for k, spec in style_dict.items():
        # spec: mp_drawing.DrawingSpec
        new_dict[k] = mp_drawing.DrawingSpec(
            color=bgr,
            thickness=spec.thickness,
            circle_radius=spec.circle_radius
        )
    return new_dict

# 기본 스타일 가져오기
base_landmark_style = mp_drawing_styles.get_default_hand_landmarks_style()
base_conn_style     = mp_drawing_styles.get_default_hand_connections_style()

# 왼/오른손 스타일 만들기 (원하는 색으로 변경)
left_landmark_styles  = recolor_style_dict(base_landmark_style, (0, 255, 0))   # 초록
left_connection_styles= recolor_style_dict(base_conn_style,     (0, 180, 0))
right_landmark_styles = recolor_style_dict(base_landmark_style, (255, 0, 0))   # 파랑
right_connection_styles= recolor_style_dict(base_conn_style,    (180, 0, 0))

HAND_COUNT = 21 * 3
POSE_COUNT = 11 * 3

hands = mp_hands.Hands(
    static_image_mode = False, #고정이미지 아님
    max_num_hands = 2,

    min_detection_confidence = 0.5, #감지 확률 0.5 이상만
    min_tracking_confidence = 0.75 # 트래킹 확률 0.5이상만
)                                                                                  

pose = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.3
)

##############################################
######### 🚨 여기를 수정하면 됩니다! 🚨 ########
# box 데이터 프레임 불러오기
guide_box_df = pd.read_csv("./data/guide_box.csv")
sign_code_df = pd.read_csv("./data/sign_code.csv")

SHOW_GUIDE_BOX = True
SAVE_IMAGE = False

# 저장할 이미지 갯수
MAX_COUNT = 500

COLUM_COUNT = 159

# 저장할 데이터 설정 
# 저장할 라벨을 적어주세요
ANSWER_LABEL = 15
ANSWER_TEXT = (
    sign_code_df.loc[sign_code_df['label'] == ANSWER_LABEL, 'sign_text']
    .squeeze() if (sign_code_df['label'] == ANSWER_LABEL).any() else None
)
print("========================================")
print(f'{ANSWER_TEXT} 를 저장하기 시작합니다!')
print(f's/space 키를 누르면 저장됩니다!')
print("========================================")

FOLDER_PATH = f'./data/sign_images/sign_images_{ANSWER_LABEL}'
FILE_PATH = f'./data/sign_data/sign_data_{ANSWER_LABEL}.csv'
######### 🚨 여기를 수정하면 됩니다! 🚨 ########
##############################################

count = 0

# 폴더 없을 경우 생성
os.makedirs(FOLDER_PATH, exist_ok=True)
jpg_files = [f for f in os.listdir(FOLDER_PATH) if f.lower().endswith(".jpg")]
image_count = len(jpg_files)

# 파일이 없을 경우 생성
if not os.path.exists(FILE_PATH):
    with open(FILE_PATH, "w") as file:
        writer = csv.writer(file)
        colums = ['label']
        colums.extend([i for i in range(COLUM_COUNT)])
        writer.writerow(colums)
else :
    try:
        df = pd.read_csv(FILE_PATH)
        count = len(df)
        print("파일 읽기 성공")
    except EmptyDataError:
        print("파일이 비어 있어서 읽을 수 없습니다.")

    print("========================================")
    print(f'{ANSWER_TEXT} 파일이 이미 존재합니다. 계속 진행해도 될까요? 괜찮으면 Y를 눌러주세요')
    print(f'괜찮으면 Y / 종료하려면 N 을 눌러주세요')
    print("========================================")
    
    while True:
        key = input("계속하려면 y, 종료하려면 n 을 입력하세요: ").strip().lower()
        if key == "y":
            break
        elif key == "n":
            exit()

    print("========================================")
    print(f'{ANSWER_TEXT} 를 저장하기를 정말 시작합니다!')
    print("========================================")

# print(image_count, count)
# if image_count != count:
#     print("이미지와 csv 갯수가 일치하지 않아요...")
#     exit()

vcap = cv2.VideoCapture(0)

while True:
    ret, frame = vcap.read()
    if not ret:
        print("웹캠이 작동하지 않습니다.")
        sys.exit()
    
    # 좌우반전
    frame = cv2.flip(frame, 1)
    origin_frame = frame.copy()

    if SHOW_GUIDE_BOX:
        draw_box(frame, guide_box_df, ANSWER_LABEL)

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
                result_landmarks['Face'].extend([landmark.x, landmark.y, landmark.z])
    
    ###### Hands Landmark 설정하기 ########
    # 손 감지하기
    hand_results = hands.process(frame)
    if hand_results.multi_hand_landmarks:        
        for hand in hand_results.multi_handedness:
            data_count[hand.classification[0].label] += 1

        for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            label = hand_results.multi_handedness[i].classification[0].label

            # 좌표 모으기
            for landmark in hand_landmarks.landmark:
                result_landmarks[label].extend([landmark.x, landmark.y, landmark.z])

            # 자동 그리기
            # mp_drawing.draw_landmarks(
            #     frame,
            #     hand_landmarks,
            #     mp_hands.HAND_CONNECTIONS,
            #     mp_drawing_styles.get_default_hand_landmarks_style(),
            #     mp_drawing_styles.get_default_hand_connections_style()
            # )
            if label == "Left":
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    left_landmark_styles,
                    left_connection_styles
                )
            else:  # "Right"
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    right_landmark_styles,
                    right_connection_styles
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
            result_landmarks['Left'] = [0] * HAND_COUNT
    
        # 오른손이 없는 경우
        if data_count['Right'] > 1:
            result_landmarks['Right'] = [0] * HAND_COUNT

        key = cv2.waitKey(1) # ASCII 코드
        if key == ord("s") or key == 32:
            result = [ANSWER_LABEL]
            result.extend(mediapipe_util.flatten_landmarks(result_landmarks))
        
            with open(FILE_PATH, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(result)
                print(f"CSV 저장 : {count + 1}/{MAX_COUNT}")
                cv2.putText(frame, "Save Data!", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

                if SAVE_IMAGE:
                    cv2.imwrite(os.path.join(FOLDER_PATH, f"{ANSWER_LABEL}_{count}.jpg"), origin_frame)

                    print(f"이미지 저장 : {count + 1}/{MAX_COUNT}")
                count += 1
    
    # 화면 띄우기
    cv2.imshow("webcam", frame)

    # 꺼지는 조건
    key = cv2.waitKey(1)
    if key == 27:
        break

    if count >= MAX_COUNT:
        print("모두 카운팅 완료!")
        break

vcap.release()
cv2.destroyAllWindows()