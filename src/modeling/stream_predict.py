import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"        # TF 로그: ERROR만
os.environ["GLOG_minloglevel"] = "3"            # glog: FATAL만
os.environ["ABSL_LOG_SEVERITY_THRESHOLD"] = "3" # absl: FATAL만

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

import sys
import cv2
import pandas as pd
import joblib
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.guide_box import draw_box
from utils.mediapipe_util import get_landmarks_file, flatten_landmarks
import mediapipe as mp

# box 데이터 프레임 불러오기``
guide_box_df = pd.read_csv("./data/guide_box.csv")
sign_code_df = pd.read_csv("./data/sign_code.csv")

# 저장할 이미지 갯수
MAX_COUNT = 50

# 정답 레이블
ANSWER_LABEL = 0 # 확인할 라벨
ANSWER_TEXT = (
    sign_code_df.loc[sign_code_df['label'] == ANSWER_LABEL, 'sign_text']
    .squeeze() if (sign_code_df['label'] == ANSWER_LABEL).any() else None
)

# 모델 불러오기
MODEL_PATH = "./models/xgb_num_model.pkl"
model = joblib.load(MODEL_PATH)

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
right_landmark_styles = recolor_style_dict(base_landmark_style, (255, 0, 0))   # 빨강
right_connection_styles= recolor_style_dict(base_conn_style,    (180, 0, 0))

HAND_COUNT = 21 * 3
POSE_COUNT = 11 * 3

vcap = cv2.VideoCapture(0)

count = 0
while True:
    ret, frame = vcap.read()
    if not ret:
        print("웹캠이 작동하지 않습니다.")
        sys.exit()
    
    # 좌우반전
    frame = cv2.flip(frame, 1)
    origin_frame = frame.copy()
    
    draw_box(frame, guide_box_df, ANSWER_LABEL)    

    # 예측
    if count > 25:
        # 랜드마크 추출
        landmarks = get_landmarks_file(origin_frame)
        if len(landmarks['Right']) > 0:
            data = flatten_landmarks(landmarks, hand_size=HAND_COUNT, face_size=POSE_COUNT)
            data = np.reshape(data[63:63+63], (1, 63))
            pred = model.predict(data)
            print("=====================")
            print(pred)
            print("=====================")

            cv2.putText(frame, f"predict: {pred[0] + 1}", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
        
        if count == 100:
            count = 0
        
    count += 1

    # 화면 띄우기
    cv2.imshow("webcam", frame)

    # 꺼지는 조건
    key = cv2.waitKey(1)
    if key == 27:
        break

vcap.release()
cv2.destroyAllWindows()