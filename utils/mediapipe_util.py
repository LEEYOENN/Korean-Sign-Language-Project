import os
import cv2
import mediapipe as mp
import numpy as np
import base64
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import glob
from typing import Union

# 상수 정의
MARGIN = 10 
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

# 모델 경로 정의
HAND_MODEL_PATH = r'./utils/landmarker_tasks/hand_landmarker.task'
POSE_MODEL_PATH = r'./utils/landmarker_tasks/pose_landmarker_full.task'

# 미디어 파이프 설정
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

HAND_OPTIONS = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
num_hands=2,
min_hand_detection_confidence=0.5,
min_hand_presence_confidence=0.5,
)

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
POSE_OPTIONS = PoseLandmarkerOptions(
    num_poses=1,
    base_options=BaseOptions(
        model_asset_path=POSE_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5)

def draw_landmarks_manual(image, landmarks_data, color):
    # Check if landmarks_data is a valid list
    if not landmarks_data:
        return image
    
    annotated_image = np.copy(image)
    height, width, _ = annotated_image.shape
    
    for x, y in landmarks_data:
        # Convert normalized coordinates (0-1) to pixel coordinates
        point_x = int(x * width)
        point_y = int(y * height)
        
        # Draw a filled circle at each landmark point
        cv2.circle(annotated_image, (point_x, point_y), 3, color, -1)
        
    return annotated_image

def draw_landmarks(image, landmarks_data, label, color):
    if not landmarks_data:
        return image
    
    annotated_image = np.copy(image)
    
    landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=p[0], y=p[1], z=0.0) for p in landmarks_data
    ])
    
    # Corrected section for drawing landmarks
    if label in ["Left", "Right"]:
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())
    else:  # Assume 'Face'
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())

    return annotated_image

def flatten_landmarks(result_landmarks: dict,
                      hand_size: int = 63,
                      face_size: int = 33) -> list:
    left  = result_landmarks.get("Left",  [])
    right = result_landmarks.get("Right", [])
    face  = result_landmarks.get("Face",  [])

    # 없을 경우 0으로 패딩
    if not left:
        left = [0.0] * hand_size
    if not right:
        right = [0.0] * hand_size
    if not face:
        face = [0.0] * face_size

    return left + right + face

def _empty_result():
    return {"Left": [], "Right": [], "Face": []}

def _to_mp_image(image_bgr, flip=True):
    if image_bgr is None:
        return None
    if flip:
        image_bgr = cv2.flip(image_bgr, 1)
    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

def _detect_landmarks_core(mp_image, include_z=False, desired_pose_landmarks=None):
    """
    mp_image: mp.Image (RGB, SRGB)
    include_z: True면 [x,y,z], False면 [x,y]
    desired_pose_landmarks: 포즈에서 저장할 인덱스 리스트 (기본 0~10)
    """
    if mp_image is None:
        return _empty_result()

    if desired_pose_landmarks is None:
        desired_pose_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    result = _empty_result()

    # --- 손 랜드마커 ---
    with HandLandmarker.create_from_options(HAND_OPTIONS) as landmarker:
        hand_res = landmarker.detect(mp_image)
        # 손이 없을 수도 있음
        for hand_lms, handedness in zip(hand_res.hand_landmarks, hand_res.handedness):
            coords = []
            for lm in hand_lms:
                if include_z:
                    coords.extend([lm.x, lm.y, lm.z])
                else:
                    coords.extend([lm.x, lm.y])

            label = handedness[0].category_name  # 'Left' or 'Right'
            if label == 'Left':
                result['Left'].extend(coords)
            elif label == 'Right':
                result['Right'].extend(coords)

    # --- 포즈 랜드마커 (얼굴/상체 일부) ---
    with PoseLandmarker.create_from_options(POSE_OPTIONS) as landmarker:
        pose_res = landmarker.detect(mp_image)
        if pose_res.pose_landmarks:
            face_coords = []
            for idx, lm in enumerate(pose_res.pose_landmarks[0]):
                if idx in desired_pose_landmarks:
                    if include_z:
                        face_coords.extend([lm.x, lm.y, lm.z])
                    else:
                        face_coords.extend([lm.x, lm.y])
            result['Face'] = face_coords

    return result

# 랜드마크 추출
def get_landmarks(image_path, include_z = True):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return _empty_result()
    mp_image = _to_mp_image(img, flip=True)
    return _detect_landmarks_core(mp_image, include_z=include_z)

def get_landmarks_file(image, include_z = True):
    mp_image = _to_mp_image(image, flip=True)
    return _detect_landmarks_core(mp_image, include_z=include_z)

def get_landmarks_from_base64(base64_string, include_z = True):
    try:
        image_bytes = base64.b64decode(base64_string)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            print("Error: Could not decode Base64 image.")
            return _empty_result()
    except Exception as e:
        print(f"Error decoding Base64 string: {e}")
        return _empty_result()

    mp_image = _to_mp_image(img, flip=True)
    return _detect_landmarks_core(mp_image, include_z=include_z)

# 폴더 내 모든 랜드마크 추출
def get_all_landmarks(folder_path):
    results = []
    
    image_files = glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png"))
    
    for img_path in image_files:
        landmarks = get_landmarks(img_path)
        results.append(landmarks)
    
    return results

def annotate_landmarks_image(
    image_or_path: Union[str, np.ndarray],
    landmarks_data: dict,
    flip_before_draw: bool = True,
    left_color=(0, 255, 0),
    right_color=(255, 0, 0),
    face_color=(0, 0, 255)
) -> np.ndarray:    
    # 1) 입력 이미지 로드/검증
    if isinstance(image_or_path, str):
        original_image = cv2.imread(image_or_path)
    else:
        original_image = image_or_path

    if original_image is None:
        raise ValueError("Error: Could not load the original image.")

    # 2) (옵션) 좌우 반전 후 RGB 변환
    working = cv2.flip(original_image, 1) if flip_before_draw else original_image.copy()
    rgb_image = cv2.cvtColor(working, cv2.COLOR_BGR2RGB)
    annotated = np.copy(rgb_image)

    # 3) 안전한 리스트 변환
    left_vals  = landmarks_data.get('Left',  []) or []
    right_vals = landmarks_data.get('Right', []) or []
    face_vals  = landmarks_data.get('Face',  []) or []

    # 4) (x,y) 점 리스트로 변환
    def to_points(vals):
        return [(vals[i], vals[i+1]) for i in range(0, len(vals), 2)]

    left_points  = to_points(left_vals)
    right_points = to_points(right_vals)
    face_points  = to_points(face_vals)

    # 5) 랜드마크 그리기 (사용자 제공 draw_landmarks_manual 사용)
    if left_points:
        annotated = draw_landmarks_manual(annotated, left_points,  left_color)
    if right_points:
        annotated = draw_landmarks_manual(annotated, right_points, right_color)
    if face_points:
        annotated = draw_landmarks_manual(annotated, face_points, face_color)

    # 6) BGR로 되돌리고, (옵션) 다시 좌우 반전해서 원본 좌표계와 정렬
    final_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    if flip_before_draw:
        final_bgr = cv2.flip(final_bgr, 1)

    return final_bgr


if __name__ == "__main__":
    test_image_path = r'C:\\Potenup\\Korean-Sign-Language-Project\data\\images\\test.jpg'
    result_image_path = r'./data/images/easy_test_result.jpg'
    
    # TEST 1
    landmarks_data = get_landmarks(test_image_path)
    result = flatten_landmarks(landmarks_data)
    print(result)
    print(len(result))
    
    original_image = cv2.imread(test_image_path)
    if original_image is None:
        print("Error: Could not load the original image.")
        exit()

    # TEST 2
    # _, buffer = cv2.imencode('.jpg', original_image)
    # base64_string = base64.b64encode(buffer).decode('utf-8')
    # landmarks_data = get_landmarks_from_base64(base64_string)
    
    result_image = annotate_landmarks_image(
        test_image_path,
        landmarks_data,
        flip_before_draw=True
    )
    
    cv2.imshow('result_image.jpg', result_image)
    cv2.imwrite(result_image_path, result_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()