import cv2
import mediapipe as mp
import numpy as np
import base64
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# 상수 정의
MARGIN = 10 
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

# 모델 경로 정의 (실제 경로로 수정 필요)
HAND_MODEL_PATH = r'C:\\Potenup\\Korean-Sign-Language-Project\\utils\\landmarker_tasks\\hand_landmarker.task'
POSE_MODEL_PATH = r'C:\\Potenup\\Korean-Sign-Language-Project\\utils\\landmarker_tasks\\pose_landmarker_full.task'

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

def get_landmarks(image_path):
    result_landmarks = {"Left" : [], 'Right': [], "Face": []}

    # 이미지 불러오기
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return result_landmarks
        
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # 미디어 파이프 기본 설정
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    # 손 랜드마커
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2)

    with HandLandmarker.create_from_options(options) as landmarker:
        hand_landmarker_result = landmarker.detect(mp_image)
        
        for hand_landmarks, handedness in zip(hand_landmarker_result.hand_landmarks, hand_landmarker_result.handedness):
            landmarks = []
            for landmark in hand_landmarks:
                landmarks.extend([landmark.x, landmark.y])
            
            hand_label = handedness[0].category_name
            if hand_label == 'Left':
                result_landmarks['Left'].extend(landmarks)
            elif hand_label == 'Right':
                result_landmarks['Right'].extend(landmarks)

    # 포즈 랜드마커
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE)
        
    with PoseLandmarker.create_from_options(options) as landmarker:
        pose_landmarker_result = landmarker.detect(mp_image)
        
        desired_pose_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        if pose_landmarker_result.pose_landmarks:
            landmarks_to_save = []
            for idx, landmark in enumerate(pose_landmarker_result.pose_landmarks[0]):
                if idx in desired_pose_landmarks:
                    landmarks_to_save.extend([landmark.x, landmark.y])
            result_landmarks['Face'] = landmarks_to_save
    
    return result_landmarks

def get_landmarks_from_base64(base64_string):
    result_landmarks = {"Left": [], 'Right': [], "Face": []}

    try:
        image_bytes = base64.b64decode(base64_string)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            print("Error: Could not decode Base64 image.")
            return result_landmarks
    except Exception as e:
        print(f"Error decoding Base64 string: {e}")
        return result_landmarks
    
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        hand_landmarker_result = landmarker.detect(mp_image)
        for hand_landmarks, handedness in zip(hand_landmarker_result.hand_landmarks, hand_landmarker_result.handedness):
            landmarks = []
            for landmark in hand_landmarks:
                landmarks.extend([landmark.x, landmark.y])
            
            hand_label = handedness[0].category_name
            if hand_label == 'Left':
                result_landmarks['Left'].extend(landmarks)
            elif hand_label == 'Right':
                result_landmarks['Right'].extend(landmarks)

    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        pose_landmarker_result = landmarker.detect(mp_image)
        
        desired_pose_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        if pose_landmarker_result.pose_landmarks:
            landmarks_to_save = []
            for idx, landmark in enumerate(pose_landmarker_result.pose_landmarks[0]):
                if idx in desired_pose_landmarks:
                    landmarks_to_save.extend([landmark.x, landmark.y])
            result_landmarks['Face'] = landmarks_to_save
    
    result = []
    result.extend(result_landmarks['Left'])
    result.extend(result_landmarks['Right'])
    result.extend(result_landmarks['Face'])

    return result_landmarks

if __name__ == "__main__":
    test_image_path = r'C:/Potenup/Korean-Sign-Language-Project/data/images/test.jpg'
    result_image_path = r'C:/Potenup/Korean-Sign-Language-Project/data/images/test_result.jpg'
    
    # TEST 1
    # landmarks_data = get_landmarks(test_image_path)
    
    original_image = cv2.imread(test_image_path)
    if original_image is None:
        print("Error: Could not load the original image.")
        exit()

    # TEST 2
    _, buffer = cv2.imencode('.jpg', original_image)
    base64_string = base64.b64encode(buffer).decode('utf-8')
    landmarks_data = get_landmarks_from_base64(base64_string)
    
    # 좌우 반전
    flipped_image = cv2.flip(original_image, 1)
    rgb_image = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB)
    annotated_image = np.copy(rgb_image)
    
    left_hand_points = [(landmarks_data['Left'][i], landmarks_data['Left'][i+1]) for i in range(0, len(landmarks_data['Left']), 2)]
    # annotated_image = draw_landmarks(annotated_image, left_hand_points, "Left", (0, 255, 0))
    annotated_image = draw_landmarks_manual(annotated_image, left_hand_points,  (0, 255, 0))

    right_hand_points = [(landmarks_data['Right'][i], landmarks_data['Right'][i+1]) for i in range(0, len(landmarks_data['Right']), 2)]
    # annotated_image = draw_landmarks(annotated_image, right_hand_points, "Right", (255, 0, 0))
    annotated_image = draw_landmarks_manual(annotated_image, right_hand_points,  (255, 0, 0))

    face_points = [(landmarks_data['Face'][i], landmarks_data['Face'][i+1]) for i in range(0, len(landmarks_data['Face']), 2)]
    # annotated_image = draw_landmarks(annotated_image, face_points, "Face", (0, 0, 255))
    annotated_image = draw_landmarks_manual(annotated_image, face_points,  (0, 0, 2552))
    
    final_bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    result_image = cv2.flip(final_bgr_image, 1)
    cv2.imshow('result_image.jpg', result_image)
    cv2.imwrite(result_image_path, result_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()