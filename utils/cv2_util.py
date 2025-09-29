import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = r"C:/Windows/Fonts/H2MJRE.TTF"

def put_korean_text(img_bgr, text, org=(10, 50), font_path=FONT_PATH,
                    font_size=32, color=(255,0,0), stroke_width=0, stroke_color=(0,0,0)):
    # BGR -> RGB (Pillow는 RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)

    # Pillow는 RGB, OpenCV는 BGR이므로 색상 순서 뒤집기
    fill = (color[2], color[1], color[0])
    stroke_fill = (stroke_color[2], stroke_color[1], stroke_color[0])

    draw.text(org, text, font=font, fill=fill,
              stroke_width=stroke_width, stroke_fill=stroke_fill)

    # RGB -> BGR
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
