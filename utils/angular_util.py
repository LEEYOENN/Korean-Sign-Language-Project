import numpy as np
from math import acos, degrees

# MediaPipe Hands index 참고 (0~20)
# 엄지: 0-1-2-3-4
# 검지: 0-5-6-7-8
# 중지: 0-9-10-11-12
# 약지: 0-13-14-15-16
# 소지: 0-17-18-19-20

FINGER_CHAINS = {
    "thumb":  [0, 1, 2, 3, 4],
    "index":  [0, 5, 6, 7, 8],
    "middle": [0, 9,10,11,12],
    "ring":   [0,13,14,15,16],
    "pinky":  [0,17,18,19,20],
    "curv":   [2, 5, 9,13,17]
}

def _to_xyz_array(flat_landmarks, length = 63, dim = 3):
    """[x1,y1,z1,...] -> (21,3) numpy array"""
    arr = np.asarray(flat_landmarks, dtype=float)
    if arr.shape[0] != length:
        raise ValueError(f"랜드마크 길이는 {length}이어야 합니다. ({length/dim}점 x 3좌표)")
    return arr.reshape(int(length/dim), dim)

def _safe_unit(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v), 0.0
    return v / n, n

import numpy as np

def flatten_vectors(vectors_dict, order=None, dim=3):
    """
    vectors_dict: {(i,j): [vx,vy,vz] 또는 np.array}
    order      : [(i,j), ...]  # 고정 순서. 없으면 키 정렬 사용
    dim        : 벡터 차원(기본 3)

    return     : [vx,vy,vz, vx,vy,vz, ...]  # 길이 = len(order)*dim
    """
    if not vectors_dict:
        return []

    if order is None:
        # (i,j) 튜플 기준 정렬 → 일관된 순서 보장
        order = sorted(vectors_dict.keys())

    flat = []
    for key in order:
        v = vectors_dict.get(key, None)
        if v is None:
            flat.extend([0.0] * dim)
            continue

        v = np.asarray(v, dtype=float).reshape(-1)
        # pad / truncate
        if v.size < dim:
            v = np.concatenate([v, np.zeros(dim - v.size, dtype=float)])
        elif v.size > dim:
            v = v[:dim]

        flat.extend(v.tolist())

    return flat

def compute_connected_unit_vectors(flat_landmarks_63, eps=1e-8, return_flat=False):
    """
    연결된 랜드마크 쌍(손가락 체인)의 '단위벡터'만 계산.
    - 입력: [x1,y1,z1, x2,y2,z2, ..., x21,y21,z21] (길이 63)
    - 출력:
        unit_vectors: {(i,j): np.ndarray([ux,uy,uz])}  # 크기=1의 방향벡터
        order:        [(i,j), ...]                     # 체인 순서대로

    주의: 두 점이 같아 벡터 길이가 eps 미만이면 [0,0,0]을 반환.
    """
    P = _to_xyz_array(flat_landmarks_63)  # (21,3)
    unit_vectors = {}
    order = []

    for chain in FINGER_CHAINS.values():
        for a, b in zip(chain[:-1], chain[1:]):
            v = P[b] - P[a]                 # a->b
            n = np.linalg.norm(v)
            if n < eps:
                u = np.zeros(3, dtype=float)
            else:
                u = v / n
            unit_vectors[(a, b)] = u
            order.append((a, b))
    
    if return_flat:
        flat = flatten_vectors(unit_vectors)
        return unit_vectors, order, flat
    else:
        return unit_vectors, order


def compute_connected_vectors(flat_landmarks_63):
    """
    1단계: 연결된 랜드마크 쌍(각 손가락 체인 기준)의 벡터를 계산.
    반환:
      - vectors_dict: {(i,j): vector_ij (3D)}, i->j (j - i)
      - order: [(i,j), ...] 고정 순서 리스트 (체인 순서대로)
    """
    P = _to_xyz_array(flat_landmarks_63)
    vectors_dict = {}
    order = []

    for chain in FINGER_CHAINS.values():
        for a, b in zip(chain[:-1], chain[1:]):
            v = P[b] - P[a]          # a->b
            vectors_dict[(a, b)] = v
            order.append((a, b))

    return vectors_dict, order

def compute_joint_angles(flat_landmarks_63, angle_unit="deg"):
    """
    2단계: 같은 체인에서 '연속된 벡터들' 사이의 각도를 계산.
    예) 체인 [0,1,2,3,4]이면, (0->1)∠(1->2), (1->2)∠(2->3), (2->3)∠(3->4)
    반환:
      - angles_list: [각도, ...] (엄지→검지→중지→약지→소지 순, 관절 순서대로)
      - angles_info: [{'finger':..., 'triplet':(a,b,c), 'angle':...}, ...]
      - angles_dict: {(a,b,c): angle, ...}
    """
    P = _to_xyz_array(flat_landmarks_63)
    angles_list = []
    angles_info = []
    angles_dict = {}

    for finger_name, chain in FINGER_CHAINS.items():
        if finger_name == 'curv':
            continue
        # 연속 세 점 (a, b, c) => 벡터 u=(a->b), v=(b->c), 관절은 b에서의 굽힘 각
        for a, b, c in zip(chain[:-2], chain[1:-1], chain[2:]):
            u = P[b] - P[a]
            v = P[c] - P[b]
            u_hat, nu = _safe_unit(u)
            v_hat, nv = _safe_unit(v)

            if nu == 0.0 or nv == 0.0:
                angle = 0.0
            else:
                # 수치 안정화: dot 범위 클램프
                dot = float(np.clip(np.dot(u_hat, v_hat), -1.0, 1.0))
                angle = acos(dot)  # radians

            if angle_unit == "deg":
                angle = degrees(angle)

            angles_list.append(angle)
            info = {"finger": finger_name, "triplet": (a, b, c), "angle": angle}
            angles_info.append(info)
            angles_dict[(a, b, c)] = angle

    return angles_list, angles_info, angles_dict

def compute_hand_features(flat_landmarks_63, angle_unit="deg"):
    """
    벡터와 각도 한 번에 계산해서 묶어서 반환.
    반환:
      {
        'vectors_order': [(i,j), ...],
        'vectors': {(i,j): np.array([vx,vy,vz]), ...},
        'angles_order': [(a,b,c), ...],
        'angles': {(a,b,c): angle, ...},
        'angles_list': [angle,...],        # 체인/관절 고정 순서
        'angles_info': [{'finger':..., 'triplet':(a,b,c), 'angle':...}, ...]
      }
    """
    vectors, v_order = compute_connected_vectors(flat_landmarks_63)
    angles_list, angles_info, angles_dict = compute_joint_angles(flat_landmarks_63, angle_unit=angle_unit)

    # angles_order는 info에서 triplet만 추출 (이미 체인 순서)
    a_order = [x["triplet"] for x in angles_info]

    return {
        "vectors_order": v_order,
        "vectors": vectors,
        "angles_order": a_order,
        "angles": angles_dict,
        "angles_list": angles_list,
        "angles_info": angles_info,
    }


FACE_IDS = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "mouth_left": 9,
    "mouth_right": 10,
}
DEFAULT_FACE_ANCHORS = ["nose", "eye_center", "mouth_center"]
DEFAULT_HAND_POINTS = [0, 4, 8, 12, 16, 20]  # wrist + five tips

def _build_face_anchors(face_xyz):
    """
    face_xyz: (>=11, 3) 를 기대. (0..10) 중 필요한 인덱스 사용.
    반환: dict 이름->(3,)
    """
    anchors = {}
    # 개별 포인트 존재 가정(사전 체크는 생략했지만 필요하면 인덱스 범위 검사 추가)
    nose = face_xyz[FACE_IDS["nose"]]
    le   = face_xyz[FACE_IDS["left_eye"]]
    re   = face_xyz[FACE_IDS["right_eye"]]
    ml   = face_xyz[FACE_IDS["mouth_left"]]
    mr   = face_xyz[FACE_IDS["mouth_right"]]

    anchors["nose"]         = nose
    anchors["eye_center"]   = (le + re) * 0.5
    anchors["mouth_center"] = (ml + mr) * 0.5
    return anchors

def compute_face_hand_vectors(
    face_flat,
    hand_flat,
    face_anchor_names=DEFAULT_FACE_ANCHORS,
    hand_indices=DEFAULT_HAND_POINTS,
    return_flat=True,
):
    """
    얼굴 앵커(코, 눈중심, 입중심)와 손 포인트(손목+5손끝) 사이의 벡터 계산.
    - 벡터 정의: anchor -> hand_point  (hand - anchor)
    - 입력: face_flat, hand_flat : [x,y,(z)]*N 형태
    - 출력:
        unit_vecs: {(anchor_name, hand_idx): np.array([ux,uy,uz])}
        order    : [(anchor_name, hand_idx), ...]
        flat     : [ux,uy,uz, ux,uy,uz, ...] (옵션)
    """
    face_xyz = _to_xyz_array(face_flat, length=33)
    hand_xyz = _to_xyz_array(hand_flat)

    # 얼굴 앵커 산출(코/눈중심/입중심)
    face_anchors = _build_face_anchors(face_xyz)

    vectors = {}
    order = []

    # 고정 순서: face_anchor_names 순회 → hand_indices 오름차순
    for an in face_anchor_names:
        a = face_anchors[an]
        for h_idx in hand_indices:
            if h_idx >= hand_xyz.shape[0]:
                # 손 랜드마크가 부족하면 0벡터
                u = np.zeros(3, dtype=float)
            else:
                v = hand_xyz[h_idx] - a
                # u_hat, nu = _safe_unit(u)
            vectors[(an, h_idx)] = v
            order.append((an, h_idx))

    if return_flat:
        flat = flatten_vectors(vectors)
        return vectors, order, flat
    else:
        return vectors, order