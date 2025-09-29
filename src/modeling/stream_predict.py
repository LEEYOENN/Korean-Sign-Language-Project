import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_LOG_SEVERITY_THRESHOLD"] = "3"

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

import sys
import cv2
import pandas as pd
import joblib
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.guide_box import draw_box
from utils.mediapipe_util import get_landmarks_file, get_landmark_data, LANDMARK_MODE
from utils.image_preprocessed_util import enhance_soft, resize_keep_aspect
from utils.cv2_util import put_korean_text
import mediapipe as mp

# box 데이터 프레임 불러오기
guide_box_df = pd.read_csv("./data/guide_box.csv")
sign_code_df = pd.read_csv("./data/sign_code.csv")

MAX_COUNT = 50
ANSWER_LABEL = 0
ANSWER_TEXT = (
    sign_code_df.loc[sign_code_df['label'] == ANSWER_LABEL, 'sign_text']
    .squeeze() if (sign_code_df['label'] == ANSWER_LABEL).any() else None
)

ANSWER_TEXT_LIST = [1, 2, 3, 4, 5, 6, 7, '좋아요', '싫어요', '맞다', '틀리다'] # ['좋아요', '싫어요', '맞다', '틀리다']]
MODEL_PATH = "./models/xgb_sample_model.pkl"
mode = LANDMARK_MODE.ANGLE_VECTOR_CURV_FACE_NOSE_WRIST
model = joblib.load(MODEL_PATH)

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def recolor_style_dict(style_dict, bgr):
    new_dict = {}
    for k, spec in style_dict.items():
        new_dict[k] = mp_drawing.DrawingSpec(
            color=bgr,
            thickness=spec.thickness,
            circle_radius=spec.circle_radius
        )
    return new_dict

base_landmark_style = mp_drawing_styles.get_default_hand_landmarks_style()
base_conn_style     = mp_drawing_styles.get_default_hand_connections_style()

left_landmark_styles   = recolor_style_dict(base_landmark_style, (0, 255, 0))
left_connection_styles = recolor_style_dict(base_conn_style,     (0, 180, 0))
right_landmark_styles  = recolor_style_dict(base_landmark_style, (255, 0, 0))
right_connection_styles= recolor_style_dict(base_conn_style,     (180, 0, 0))

HAND_COUNT = 21 * 3
POSE_COUNT = 11 * 3

STRIDE    = 3     # 매 3프레임마다만 예측
DET_WIDTH = 512   # 검출용 다운스케일 폭

# 웹캠 설정
cv2.setUseOptimized(True)
vcap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
vcap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
vcap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
vcap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
vcap.set(cv2.CAP_PROP_FPS, 30)

if not vcap.isOpened():
    print("웹캠이 작동하지 않습니다.")
    sys.exit()

count = 0
last_pred_text = None  # ★ 마지막 예측 결과 캐시

while True:
    ret, frame = vcap.read()
    if not ret:
        print("웹캠이 작동하지 않습니다.")
        break

    # 좌우반전 (표시/검출 모두 동일 기준으로)
    frame = cv2.flip(frame, 1)

    det_img = resize_keep_aspect(frame, DET_WIDTH)
    det_img = enhance_soft(det_img)

    # ====== 예측 (프레임 스킵) ======
    if count > 25 and (count % STRIDE == 0):
        landmarks = get_landmarks_file(det_img)
        right_data = landmarks['Right']

        if len(right_data) == HAND_COUNT:
            data = get_landmark_data(landmarks, mode=mode)
            pred = model.predict(data)

            last_pred_text = f"predict: {ANSWER_TEXT_LIST[int(pred[0])]}"

        if count == 100:
            count = 0

    count += 1
    
    display = frame

    # 가이드 박스 및 예측 텍스트 표시
    draw_box(display, guide_box_df, ANSWER_LABEL)

    if last_pred_text:
        display = put_korean_text(
            display,
            last_pred_text,
            org=(10, 50),
            font_size=32,
            color=(255,0,0),
            stroke_color=(0,0,0)
        )

    cv2.imshow("webcam", display)
    if cv2.waitKey(1) == 27:  # ESC
        break

vcap.release()
cv2.destroyAllWindows()
