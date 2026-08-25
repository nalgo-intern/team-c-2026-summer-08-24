import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO

class PoseEstimator:
    def __init__(self):
        yolo_model: YOLO
        self.frame_order = 0
        self.records = []

        self.yolo_model = YOLO('yolov8n.pt')
        self.current_offset = (0,0)
        self.current_bbox = None


    # YOLOでクロップしてmediapipeで推論、角度計算をする
    def process_frame(self, frame):
        self.frame_order += 1

        # YOLOによる人物の検出
        res = self.yolo_model(frame, classes=[0], verbose=False)

        if len(res[0].boxes) == 0:
            return None

        box = res[0].boxes[0]
        x1, y1, x2, y2 = (int(x) for x in box.xyxy[0])
        self.current_bbox = (x1, y1, x2, y2)
        self.current_offset = (x1, y1)

        cropped_frame = frame[y1:y2, x1:x2]

        angles = {
            'frame_order': self.frame_order,
            'right_elbow_right_shoulder_right_hip': np.random.uniform(90, 180),
            'left_elbow_left_shoulder_left_hip': np.random.uniform(90, 180),
            'right_knee_mid_hip_left_knee': np.random.uniform(30, 90),
            'right_hip_right_knee_right_ankle': np.random.uniform(90, 180),
            'left_hip_left_knee_left_ankle': np.random.uniform(90, 180),
            'right_wrist_right_elbow_right_shoulder': np.random.uniform(90, 180),
            'left_wrist_left_elbow_left_shoulder': np.random.uniform(90, 180),
        }
        
        self.records.append(angles)
        return angles


    # 動画から全フレームを処理する
    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
            
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame)

            frame = self.draw_landmarks(frame)
            cv2.imshow("test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()

    # 骨格点を描画する
    def draw_landmarks(self, frame):
        if self.current_bbox is not None:
            x1, y1, x2, y2 = self.current_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return frame

    # pandasのデータフレームとして角度の情報を返す
    def get_dataframe(self):
        return pd.DataFrame(self.records)

    # 角度を計算する
    @staticmethod
    def calculate_angle(a, b, c):
        return 0.0