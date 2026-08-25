import pandas as pd
import numpy as np

class PoseEstimator:
    def __init__(self):
        self.frame_order = 0
        self.records = []

    # YOLOでクロップしてmediapipeで推論、角度計算をする
    def process_frame(self, frame):
        self.frame_order += 1
        
        dummy_angles = {
            'frame_order': self.frame_order,
            'right_elbow_right_shoulder_right_hip': np.random.uniform(90, 180),
            'left_elbow_left_shoulder_left_hip': np.random.uniform(90, 180),
            'right_knee_mid_hip_left_knee': np.random.uniform(30, 90),
            'right_hip_right_knee_right_ankle': np.random.uniform(90, 180),
            'left_hip_left_knee_left_ankle': np.random.uniform(90, 180),
            'right_wrist_right_elbow_right_shoulder': np.random.uniform(90, 180),
            'left_wrist_left_elbow_left_shoulder': np.random.uniform(90, 180),
        }
        
        self.records.append(dummy_angles)
        return dummy_angles

    # 骨格点を描画する
    def draw_landmarks(self, frame):
        return frame

    # pandasのデータフレームとして角度の情報を返す
    def get_dataframe(self):
        return pd.DataFrame(self.records)

    # 角度を計算する
    @staticmethod
    def calculate_angle(a, b, c):
        return 0.0