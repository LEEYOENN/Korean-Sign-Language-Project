import pandas
import cv2
import sys
import time

df = pandas.read_csv("C:\Potenup\Korean-Sign-Language-Project\data\guide_box.csv")

#print(df)

def draw_box(frame, box_df, label):
    label_box = box_df[box_df["label"] == label]
    face_box = box_df[box_df["label"] == 25]
    
    # 학습시킬 박스 표시
    for _, row in label_box.iterrows():
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (230, 220, 142), 2)
        cv2.putText(frame, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 220, 142), 2)
    x1, y1, x2, y2 = int(face_box["x1"].iloc[0]), int(face_box["y1"].iloc[0]), int(face_box["x2"].iloc[0]), int(face_box["y2"].iloc[0])
    
    # 얼굴영역 박스 표시
    cv2.rectangle(frame, (x1, y1), (x2, y2), (121, 190, 132), 2)
    cv2.putText(frame, "face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (121, 190, 132), 2)

    return frame

if __name__ == "__main__":
    label = 20

    vcap = cv2.VideoCapture(0)

    #print(grouped_df)

    while True:
        ret, frame = vcap.read()
        if not ret:
            print("웹캠이 작동하지 않습니다.")
            sys.exit()

        # 좌우 반전
        flipped_frame = cv2.flip(frame, 1)


        frflipped_frameame = draw_box(flipped_frame, df, label)
        cv2.imshow("webcam", flipped_frame)

        # 꺼지는 조건
        key = cv2.waitKey(1)
        if key == 27:
            break

    vcap.release()
    cv2.destroyAllWindows()