import cv2
import numpy as np

def resize_keep_aspect(img, target_w=512):
    h, w = img.shape[:2]
    if w <= target_w:
        return img
    new_h = int(h * (target_w / w))
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)

def enhance_contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(32, 32))
    enhanced = clahe.apply(gray)
    inverted = cv2.bitwise_not(enhanced)   # 흑백 반전
    return cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)  # 3채널로 복원

def enhance_soft(img, clahe_clip=2.0, clahe_tile=8, sharp_amt=0.4, blend=0.6):
    # 1) 조명 보정(간단): 큰 블러로 배경 조도 제거 후 보정
    blur = cv2.GaussianBlur(img, (0,0), 21)
    illum = cv2.addWeighted(img, 1.4, blur, -0.4, 0)  # 부드럽게 대비↑

    # 2) LAB에서 L 채널만 CLAHE
    lab = cv2.cvtColor(illum, cv2.COLOR_BGR2Lab)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    out = cv2.cvtColor(lab2, cv2.COLOR_Lab2BGR)

    # 3) 약한 언샤프 마스크(엣지만 살짝)
    gauss = cv2.GaussianBlur(out, (0,0), 1.0)
    unsharp = cv2.addWeighted(out, 1 + sharp_amt, gauss, -sharp_amt, 0)

    # 4) 원본과 소프트 블렌딩(과변형 방지)
    final = cv2.addWeighted(unsharp, blend, img, 1.0 - blend, 0)
    return final

def enhance_retinex(img, sigma=30, edge_gain=0.15, blend=0.7):
    img32 = img.astype(np.float32) + 1.0
    base = cv2.GaussianBlur(img32, (0,0), sigma)
    ret = cv2.log(img32) - cv2.log(base)          # 간단 MSR 느낌
    ret = cv2.normalize(ret, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 약한 엣지 강화
    lap = cv2.Laplacian(ret, cv2.CV_16S, ksize=3)
    lap = cv2.convertScaleAbs(lap)
    sharpen = cv2.addWeighted(ret, 1.0, lap, edge_gain, 0)

    return cv2.addWeighted(sharpen, blend, img, 1.0 - blend, 0)
