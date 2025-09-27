# # uvicorn src.app.main:app --host 0.0.0.0 --port 8000
from fastapi import FastAPI, WebSocket
from utils.mediapipe_util import get_landmarks_from_base64, flatten_landmarks
from xgboost import XGBClassifier
import base64
import joblib
import numpy as np
import uuid

app = FastAPI()

# 설정
MODEL_PATH = "./models/xgb_test_model.pkl"
model = joblib.load(MODEL_PATH)

@app.websocket("/ws/socket")
async def websocket(websocket: WebSocket):
    '''
        Request  : {"images" : image_bytes, "sign_id" : sign_id}
        Response : {"is_correct" : True/False, "sign_id" : sign_id, "code" : 200/400}
    '''
    await websocket.accept()

    try:
        # 메시지 수신
        request_data = await websocket.receive_json()
        print(request_data, type(request_data))
        images_b64 = request_data.get("images")
        sign_id = request_data.get("sign_id")

        if images_b64 is None or sign_id is None:
            print("not images_b64")
            await websocket.send_json({"is_correct": False, "sign_id": sign_id, "code": 400})
            return

        print(f"images ::: {images_b64} / sign_id ::: {sign_id}")

        if images_b64.startswith("data:image"):
            images_b64 = images_b64.split(",")[1]

        # decoding + landmark 추출 + 전처리 + 예측
        prediction = handle_prediction(images_b64)

        # 추론 결과와 sign_id 비교
        is_correct = check_prediction_match(prediction, sign_id)
    
        await websocket.send_json({"is_correct" : is_correct, "sign_id" : sign_id, "code" : 200})

    except Exception as e:
        print(f"[WebSocket Error] {e}")
        await websocket.send_json({
            "is_correct": False,
            "sign_id": None,
            "code": 400
        })
    
    finally: 
        await websocket.close()
        print("WebSocket 연결 종료")


@app.websocket("/ws/test")
async def test(websocket: WebSocket):
    await websocket.accept()

    try:
        data = await websocket.receive_json()
        print(data, type(data))

        await websocket.send_json({"is_correct" : "True", "sign_id" : "one", "code" : 200})

    except Exception as e:
        print(f"WebSocket 에러 발생: {e}")
    
    finally: 
        await websocket.close()
        print("WebSocket 연결 종료")


def check_prediction_match(pred: np.ndarray, sign_id: str) -> bool:
    # 예측 결과와 클라이언트의 sign_id 비교
    predicted_id = str(pred[0])
    is_match = predicted_id == sign_id
    print(f"Prediction: {predicted_id}, sign_id: {sign_id}, Match: {is_match}")
    return is_match


def handle_prediction(image_base64: str) -> np.ndarray:

    landmarks = get_landmarks_from_base64(image_base64)

    if landmarks is None:
        raise ValueError("Failed to extract landmarks")

    flattened = flatten_landmarks(landmarks)
    input_data = np.reshape(flattened, (1, -1))
    # input_data = np.reshape(flattened, (1, 106))

    return model.predict(input_data)

# 이미지 저장
# file_name = f"output_{uuid.uuid4().hex[:5]}.png"
# with open(file_name, 'wb') as f:
#     f.write(image_data)

