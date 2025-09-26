# uvicorn src.app.backend.main:app --host 0.0.0.0 --port 8000
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws/socket")
async def websocket(websocket: WebSocket):
    '''
        Request  : {"images" : image_bytes, "sign_id" : sign_id}
        Response : {"is_success" : True/False, "sign_id" : sign_id}
    '''
    await websocket.accept()

    try:
        data = await websocket.receive_json()
        print(data, type(data))
        images = data["images"]
        sign_id = data["sign_id"]

        print(f"images ::: {images} / sign_id ::: {sign_id}")

        is_success = "True"
        await websocket.send_json({"is_success" : is_success, "sign_id" : sign_id})

    except Exception as e:
        print(f"WebSocket 에러 발생: {e}")
    
    finally: 
        await websocket.close()
        print("WebSocket 연결 종료")


@app.websocket("/ws/test")
async def test(websocket: WebSocket):
    await websocket.accept()

    try:
        data = await websocket.receive_json()
        print(data, type(data))

        await websocket.send_json({"result" : "success"})

    except Exception as e:
        print(f"WebSocket 에러 발생: {e}")
    
    finally: 
        await websocket.close()
        print("WebSocket 연결 종료")
