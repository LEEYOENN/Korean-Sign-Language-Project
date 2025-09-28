# uvicorn src.app.main:app --host 0.0.0.0 --port 8000
from fastapi import FastAPI, WebSocket
from utils.mediapipe_util import get_landmarks_from_base64, get_landmark_data
from starlette.websockets import WebSocketState, WebSocketDisconnect
from xgboost import XGBClassifier
import base64
import joblib
import numpy as np
import uuid

app = FastAPI()

# 설정
MODEL_PATH = "./models/xgb_sample_angle_vector_model.pkl"
model = joblib.load(MODEL_PATH)

@app.websocket("/ws/socket")
async def websocket(websocket: WebSocket):
    '''
        Request JSON: 
            {
                "images" : <base64_image_string>,
                "sign_id" : <string>,
                "action" : "end" | null
            }
        Response JSON: 
            {
                "is_correct" : true | false, 
                "sign_id" : <string>, 
                "code" : 200 | 400
            }
    '''

    await websocket.accept()
    print("WebSocket 연결 시작")

    try:
        while True:
            # 메시지 수신
            request_data = await websocket.receive_json()
            print(type(request_data))

            # 종료 조건 처리
            action = request_data.get("action")
            if action == "end":
                print("클라이언트 요청에 의해 WebSocket 종료")
                await websocket.close()
                break

            images_b64 = request_data.get("images")
            sign_id = request_data.get("sign_id")

            if images_b64 is None or sign_id is None:
                print("images_b64 or sign_id is None")
                await websocket.send_json({"is_correct": False, "sign_id": sign_id, "code": 400})
                continue

            print(f"sign_id ::: {sign_id}")

            if images_b64.startswith("data:image"):
                images_b64 = images_b64.split(",")[1]

            # 손 확인 로직 추가
            landmark_result = get_landmarks_from_base64(images_b64, include_z=True)
            if len(landmark_result['Left']) == 0 and len(landmark_result['Right']) == 0:
                print("손이 감지되지 않았습니다.")
                await websocket.send_json({
                    "is_correct": False,
                    "sign_id": sign_id,
                    "code": 400
                })
                continue

            # decoding + landmark 추출 + 전처리 + 예측
            prediction = handle_prediction(images_b64)

            # 추론 결과와 sign_id 비교
            is_correct = check_prediction_match(prediction, sign_id)
        
            await websocket.send_json({"is_correct" : is_correct, "sign_id" : sign_id, "code" : 200})

    except WebSocketDisconnect:
        print("클라이언트가 WebSocket을 정상 종료했습니다.")

    except Exception as e:
        print(f"[WebSocket Error] {e}")
        try:
            if websocket.application_state != WebSocketState.DISCONNECTED:
                await websocket.send_json({
                    "is_correct": False,
                    "sign_id": None,
                    "code": 400
                })
        except RuntimeError as re:
            print((f"[응답 실패] 이미 닫힌 상태 : {re}"))
    
    finally: 
        if websocket.application_state != WebSocketState.DISCONNECTED:
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
    
    data = get_landmark_data(landmarks)
    pred = model.predict(data)

    print(f"pred ::: {pred}")

    return pred

# 이미지 저장
# file_name = f"output_{uuid.uuid4().hex[:5]}.png"
# with open(file_name, 'wb') as f:
#     f.write(image_data)

