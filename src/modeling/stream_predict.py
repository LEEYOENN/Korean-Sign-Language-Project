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
MODEL_PATH = "./models/xgb_test_model.pkl"
model = joblib.load(MODEL_PATH)

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

    # quality=90
    # params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    # _, buf = cv2.imencode('jpg', frame, params)
    # b64 = base64.b64encode(buf).decode("utf-8")

    # 랜드마크 추출
    data = flatten_landmarks(get_landmarks_file(origin_frame))

    data = np.reshape(data, (1, 106))

    # 예측
    if count > 25:
        pred = model.predict(data)
        print(pred)

        cv2.putText(frame, f"predict: {pred[0]}", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
        
        if count == 50:
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