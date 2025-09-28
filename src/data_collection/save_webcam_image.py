import sys
import cv2
import mediapipe as mp
import pandas as pd
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.guide_box import draw_box

##############################################
######### 🚨 여기를 수정하면 됩니다! 🚨 ########
# box 데이터 프레임 불러오기
guide_box_df = pd.read_csv("./data/guide_box.csv")
sign_code_df = pd.read_csv("./data/sign_code.csv")

# 저장할 이미지 갯수
MAX_COUNT = 50

# 저장할 데이터 설정 

ANSWER_LABEL = 0 # 저장할 라벨을 적어주세요
ANSWER_TEXT = (
    sign_code_df.loc[sign_code_df['label'] == ANSWER_LABEL, 'sign_text']
    .squeeze() if (sign_code_df['label'] == ANSWER_LABEL).any() else None
)
print("========================================")
print(f'{ANSWER_TEXT} 를 저장하기 시작합니다!')
print(f's/space 키를 누르면 저장됩니다!')
print("========================================")

FOLDER_PATH = f'./data/sign_images/sign_images_{ANSWER_LABEL}'
######### 🚨 여기를 수정하면 됩니다! 🚨 ########
##############################################

count = 0
# 폴더가 없을 경우 생성
os.makedirs(FOLDER_PATH, exist_ok=True)
jpg_files = [f for f in os.listdir(FOLDER_PATH) if f.lower().endswith(".jpg")]
count = len(jpg_files)

vcap = cv2.VideoCapture(0)

while True:
    ret, frame = vcap.read()
    if not ret:
        print("웹캠이 작동하지 않습니다.")
        sys.exit()
    
    # 좌우반전
    frame = cv2.flip(frame, 1)
    origin_frame = frame.copy()
    
    draw_box(frame, guide_box_df, ANSWER_LABEL)

    # 화면 띄우기
    cv2.imshow("webcam", frame)

    key = cv2.waitKey(1) # ASCII 코드
    if key == ord("s") or key == 32:
        cv2.imwrite(os.path.join(FOLDER_PATH, f"{ANSWER_LABEL}_{count}.jpg"), origin_frame) 
        print(f"이미지 저장 : {count + 1}/{MAX_COUNT}")
        count += 1

    # 꺼지는 조건
    key = cv2.waitKey(1)
    if key == 27:
        break

vcap.release()
cv2.destroyAllWindows()