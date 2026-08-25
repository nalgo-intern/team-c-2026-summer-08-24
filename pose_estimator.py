import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PoseEstimator:
    def __init__(self, model_path='pose_landmarker_full.task'):
        yolo_model: YOLO

        self.frame_order = 0
        self.records = []

        #Yoloの初期化
        self.yolo_model = YOLO('yolov8n.pt')

        # MediaPipeの初期化
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1
        )

        self.POSE_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
            (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
        ]
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        self.current_offset = (0,0)
        self.current_bbox = None
        self.current_landmarks = None


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

        if cropped_frame.size == 0:
            return None

        img_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        results = self.pose_landmarker.detect(mp_image)

        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            self.current_landmarks = results.pose_landmarks[0]

        def pt(idx):
            if self.current_landmarks:
                return np.array([self.current_landmarks[idx].x, self.current_landmarks[idx].y])
            return np.zeros(2)
        
        r_shoulder = pt(12)
        l_shoulder = pt(11)

        r_elbow    = pt(14)
        l_elbow    = pt(13)
        
        r_hip      = pt(24)
        l_hip      = pt(23)
        m_hip      = (l_hip + r_hip) / 2.0

        r_knee     = pt(26)
        l_knee     = pt(25)

        r_ankle    = pt(28)
        l_ankle    = pt(27)

        r_wrist    = pt(16)
        l_wrist    = pt(15)

        angles = {
            'frame_order': self.frame_order,
            'right_elbow_right_shoulder_right_hip':     self.calculate_angle(r_elbow, r_shoulder, r_hip),
            'left_elbow_left_shoulder_left_hip':        self.calculate_angle(l_elbow, l_shoulder, l_hip),
            'right_knee_mid_hip_left_knee':             self.calculate_angle(r_knee, m_hip, l_knee),
            'right_hip_right_knee_right_ankle':         self.calculate_angle(r_hip, r_knee, r_ankle),
            'left_hip_left_knee_left_ankle':            self.calculate_angle(l_hip, l_knee, l_ankle),
            'right_wrist_right_elbow_right_shoulder':   self.calculate_angle(r_wrist, r_elbow, r_shoulder),
            'left_wrist_left_elbow_left_shoulder':      self.calculate_angle (l_wrist, l_elbow, l_shoulder),
        }
        
        self.records.append(angles)
        return angles


    # 動画から全フレームを処理する
    def process_video(self, video_path: str, show_video: bool = False):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame)

            # デバッグ用 ============================
            if show_video:
                frame = self.draw_landmarks(frame)
                cv2.imshow("test", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # =======================================
            
        cap.release()
        if show_video:
            cv2.destroyAllWindows()

    # 骨格点を描画する
    def draw_landmarks(self, frame):
        if self.current_bbox is not None:
            x1, y1, x2, y2 = self.current_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if self.current_landmarks:
            crop = frame[y1:y2, x1:x2].copy()
            h, w, _ = crop.shape
            
            # 0.0〜1.0の座標をピクセルに変換
            pts = [(int(l.x * w), int(l.y * h)) for l in self.current_landmarks]
            
            # 骨格の線を引く
            for a, b in self.POSE_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(crop, pts[a], pts[b], (255, 255, 255), 2)
            
            # 関節の点を打つ
            for p in pts:
                cv2.circle(crop, p, 4, (0, 255, 0), -1)
                
            frame[y1:y2, x1:x2] = crop

        return frame

    # pandasのデータフレームとして角度の情報を返す
    def get_dataframe(self):
        return pd.DataFrame(self.records)

    # 角度を計算する
    @staticmethod
    def calculate_angle(a, b, c):
        if np.all(a == 0.0) or np.all(b == 0.0) or np.all(c == 0.0):
            return 0.0

        v1 = a - b
        v2 = c - b

        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)

        angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        return float(angle)