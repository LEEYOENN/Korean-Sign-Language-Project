import os
import sys
import cv2
import mediapipe as mp
import pandas as pd
import glob
import numpy as np
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils import mediapipe_util


##############################################
######### 🚨 여기를 수정하면 됩니다! 🚨 ########
# 저장할 데이터 설정 
ANSWER_LABEL = 15

COLUM_COUNT = 159

FILE_PATH = f'./data/sign_images/sign_images_{ANSWER_LABEL}'
CSV_PATH = f'./data/sign_data/sign_data_{ANSWER_LABEL}.csv'
######### 🚨 여기를 수정하면 됩니다! 🚨 ########
##############################################

colums = ['label']
colums.extend([i for i in range(COLUM_COUNT)])

count = 0
# 파일이 없을 경우 생성
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w") as file:
        writer = csv.writer(file)
        writer.writerow(colums)
else :
    try:
        df = pd.read_csv(CSV_PATH)
        count = len(df)
    except:
        count = 0

exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
files = []
for e in exts:
    files.extend(glob.glob(os.path.join(FILE_PATH, e)))
files.sort()

if not files:
    print("이미지 파일이 없습니다.")
    exit()

for idx, path in enumerate(files, 1):
    if count > idx:
        print("SKIP ", idx)
        continue

    original_image = cv2.imread(path)
    if original_image is None:
        print(f"[SKIP] 로드 실패: {path}")
        continue

    win_name = "Preview (s: save, Space: next, q/ESC: close)"
    cv2.imshow(win_name, original_image)
    landmarks_data = mediapipe_util.get_landmarks(path)

    result_image = mediapipe_util.annotate_landmarks_image(
        path,
        landmarks_data,
        flip_before_draw=True
    )
    cv2.imshow(win_name, result_image)

    while True:
        key = cv2.waitKey(1)

        if key == ord('s'):
            # 저장 후 다음 이미지로 넘어가
            # csv 저장하기
            with open(CSV_PATH, "a", newline="") as file:
                
                result = [ANSWER_LABEL]
                result.extend(mediapipe_util.flatten_landmarks(landmarks_data))

                writer = csv.writer(file)
                writer.writerow(result)

                count += 1
                print('CSV 저장 완료! ', count)
            cv2.waitKey(10)
            cv2.destroyWindow(win_name)
            break

        elif key == 32:  # Space: 추출 없이 다음으로
            cv2.destroyWindow(win_name)
            break

        elif key in (27, ord('q')):  # ESC or q: 종료
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()