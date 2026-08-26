import os
import pandas as pd
import numpy as np
import cv2
import urllib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from moviepy import VideoFileClip

class PoseEstimator:
    def __init__(self, model_path=r"pose_landmarker_lite.task"):
        
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
    def process_video(self, video_path: str):
        self.frame_order = 0
        self.records = []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        frame_index = 0

        video_landmarks = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = (frame_index / fps) * 1000

            self.process_frame(frame, timestamp_ms)

            video_landmarks.append(self.current_landmarks)
        
            frame_index += 1
            
        cap.release()
        
        return video_landmarks

    def render_video(self, input_video_path: str, video_landmarks: list, frame_counts: list, frame_evals: list, output_video_path: str = "output.mp4"):
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            return
            
        # 元動画の情報を取得
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        
        temp_output = "temp_" + os.path.basename(output_video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        
        frame_index = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_index < len(video_landmarks):
                self.current_landmarks = video_landmarks[frame_index]
                frame = self.draw_landmarks(frame)
                
                # ===== テキストの描画処理 =====
                record = self.records[frame_index]
                count = frame_counts[frame_index]
                
                # 膝の角度を取得 (取得できない場合は0.0にする)
                l_knee_angle = record.get('left_hip_left_knee_left_ankle', 0.0)
                r_knee_angle = record.get('right_hip_right_knee_right_ankle', 0.0)
                
                # --- 動画の左上にテキストを描画 ---
                cv2.putText(frame, f"Count: {count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.putText(frame, f"L-Knee: {l_knee_angle:.1f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.putText(frame, f"R-Knee: {r_knee_angle:.1f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                # --- 追加箇所: 評価指標値の描画 ---
                if frame_index < len(frame_evals) and frame_evals[frame_index] is not None:
                    eval_data = frame_evals[frame_index]
                    eval_angle = eval_data.get("knee_angle", 0.0)
                    
                    # 判定に使っている角度を描画（色は赤など目立つ色に）
                    cv2.putText(frame, f"Eval Angle: {eval_angle:.1f}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                # ===============================================

            out.write(frame)
                
            frame_index += 1
            
        cap.release()
        out.release()
        
        # --- ここから moviepy で H.264 に変換 ---
        try:
            clip = VideoFileClip(temp_output)
            clip.write_videofile(output_video_path, codec="libx264", audio=False, logger=None)
            clip.close()
            
            if os.path.exists(temp_output):
                os.remove(temp_output)
                
        except Exception as e:
            print(f"動画の変換に失敗しました: {e}")
            return temp_output  
        
        return output_video_path
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