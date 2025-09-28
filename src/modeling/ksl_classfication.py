import os
import sys
import cv2
import joblib
import numpy as np
import base64

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.mediapipe_util import get_landmarks_from_base64, flatten_landmarks

MODEL_PATH = 'models/rf_test_model.pkl'
IMAGE_PATH = 'data/images/4_1.jpg'
NUM_FEATURES=106
NUM_LABELS = 11

def test():
    try:
        # 1. 모델 불러오기
        model = joblib.load(MODEL_PATH)
        print("모델이 성공적으로 로드되었습니다.")
    except FileNotFoundError:
        print("모델 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        sys.exit()

    # 테스트를 위한 예시 이미지 준비
    try:
        with open(IMAGE_PATH, "rb") as image_file:
            image_bytes = image_file.read()
            base64_string = base64.b64encode(image_bytes).decode('utf-8')
        print("이미지 파일이 성공적으로 로드되었습니다.")
    except FileNotFoundError:
        print(f"오류: {IMAGE_PATH} 파일을 찾을 수 없습니다.")
        sys.exit()

    # 2. get_landmarks_from_base64 함수를 사용해서 랜드마크 추출
    landmarks_data = get_landmarks_from_base64(base64_string)

    # 3. 모델의 입력에 맞게 랜드마크 데이트를 준비
    # 딕셔너리에서 모든 랜드마크 리스트를 합칩니다.
    result = flatten_landmarks(landmarks_data)

    print(f"추출된 랜드마크 데이터: {result}")

    if len(result) != NUM_FEATURES:
        print(f"오류: 랜드마크 개수가 모델의 피처 수 106과 일치하지 않습니다.")
        print(f"랜드마크 개수: {len(result)}")
    
    features_array = np.array(result).reshape(1, -1)[:,42:85]

    # 4. 모델의 예측 수행
    prediction = model.predict(features_array)
    print(f"모델의 예측 결과: {prediction[0]}")

if __name__ == '__main__':
    test()
