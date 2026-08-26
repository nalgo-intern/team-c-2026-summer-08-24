import os
import pandas as pd
import numpy as np
import cv2
import urllib
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PoseEstimator:
    def __init__(self, model_path='pose_landmarker_lite.task'):
        
        if not os.path.exists(model_path):
            url = f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            urllib.request.urlretrieve(url, model_path)

        self.frame_order = 0
        self.records = []

        # MediaPipeの初期化
        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.GPU
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)

        self.POSE_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
            (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
        ]

        self.current_landmarks = None


    # YOLOでクロップしてmediapipeで推論、角度計算をする
    def process_frame(self, frame, timestamp_ms):
        self.frame_order += 1

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        results = self.pose_landmarker.detect_for_video(mp_image, int(timestamp_ms))

        self.current_landmarks = results.pose_landmarks[0] if results.pose_landmarks else None

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
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps)
        if fps == 0: fps = 30
        frame_index = 0

        while cap.isOpened():
            if frame_index % 5 != 0:
                frame_index += 1
                continue
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = (frame_index / fps) * 1000

            self.process_frame(frame, timestamp_ms)

            print(timestamp_ms / 1000)

            # デバッグ用 ============================
            if show_video:
                frame = self.draw_landmarks(frame)
            # ==========================================
                if frame_index < len(self.records):
                    record = self.records[frame_index]
                    
                    # 左右の膝の角度を取得 (取得できない場合は0.0にする)
                    l_knee = record.get('left_hip_left_knee_left_ankle', 0.0)
                    r_knee = record.get('right_hip_right_knee_right_ankle', 0.0)
                    avg_knee = (l_knee + r_knee) / 2.0
                    
                    # OpenCVで画面の左上にテキストを描画 (黄色とピンク)
                    cv2.putText(frame, f"L Knee: {l_knee:.1f}", (10, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"R Knee: {r_knee:.1f}", (10, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Avg Knee: {avg_knee:.1f}", (10, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2, cv2.LINE_AA)
                # ==========================================
                
            out.write(frame)
                
            frame_index += 1
        cap.release()
        if show_video:
            cv2.destroyAllWindows()

    # 骨格点を描画する
    def draw_landmarks(self, frame):

        if self.current_landmarks:
            h, w, _ = frame.shape
            
            # 0.0〜1.0の座標をピクセルに変換
            pts = [(int(l.x * w), int(l.y * h)) for l in self.current_landmarks]
            
            # 骨格の線を引く
            for a, b in self.POSE_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
            
            # 関節の点を打つ
            for p in pts:
                cv2.circle(frame, p, 4, (0, 255, 0), -1)

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