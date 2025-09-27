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
}

def _to_xyz_array(flat_landmarks_63):
    """[x1,y1,z1,...] -> (21,3) numpy array"""
    arr = np.asarray(flat_landmarks_63, dtype=float)
    if arr.shape[0] != 63:
        raise ValueError("랜드마크 길이는 63이어야 합니다. (21점 x 3좌표)")
    return arr.reshape(21, 3)

def _safe_unit(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v), 0.0
    return v / n, n

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
