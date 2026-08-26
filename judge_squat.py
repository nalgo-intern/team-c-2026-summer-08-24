import math


class SquatEvaluator:
    """
    スクワットのフォーム判定のみを行う（動画読み込み・姿勢推定は行わない）。
    呼び出し側が1フレームごとに関節座標を渡し、judge_frame()で判定する。

    landmarks の形式: MediaPipeのPoseLandmarker形式（例: PoseEstimator.current_landmarks）
      各要素が .x, .y 属性を持つオブジェクトのリスト（正規化座標 0〜1）で、
      添字がそのままMediaPipeのランドマーク番号と一致している想定。
      使用する番号: 23(左腰), 25(左膝), 27(左足首)
    """

    def __init__(self):
        self.feedback_logs = []  # [{"time": float, "message": str}]
        self.min_knee_angle = 180.0

    def judge_frame(self, current_second: float, landmarks) -> dict | None:
        """
        1フレーム分の関節座標(MediaPipe形式)を渡して判定する。
        戻り値: そのフレームで計算した指標値（グラフ描画などに利用可能）。判定不能な場合はNone。
        """
        if not landmarks or len(landmarks) <= 27:
            return None

        hip = (landmarks[23].x, landmarks[23].y)
        knee = (landmarks[25].x, landmarks[25].y)
        ankle = (landmarks[27].x, landmarks[27].y)

        knee_angle = self._calculate_angle(hip, knee, ankle)

        if knee_angle < self.min_knee_angle:
            self.min_knee_angle = knee_angle

        if 90 < knee_angle < 140:
            self._append_log(
                "しゃがみが浅いです（膝を90度以下まで曲げましょう）",
                current_second,
            )

        return {"knee_angle": knee_angle}

    def get_result(self) -> dict:
        """全フレーム処理後に呼び出し、最終的な判定結果を取得する"""
        if self.min_knee_angle <= 90:
            summary = "GOOD: 十分な深さまでしゃがめています"
        else:
            summary = "WARN: 全体的にしゃがみが浅い傾向があります"

        return {
            "min_knee_angle": self.min_knee_angle,
            "feedback_logs": self.feedback_logs,
            "summary": summary,
        }

    @staticmethod
    def _calculate_angle(a: tuple, b: tuple, c: tuple) -> float:
        """3点a-b-cから、bを頂点とした角度(度)を計算する"""
        angle = math.degrees(
            math.atan2(c[1] - b[1], c[0] - b[0])
            - math.atan2(a[1] - b[1], a[0] - b[0])
        )
        angle = abs(angle)
        if angle > 180:
            angle = 360 - angle
        return angle

    def _append_log(self, message: str, current_second: float):
        """重複を避けつつログを追加（2秒以内の同一メッセージはスキップ）"""
        if (
            not self.feedback_logs
            or self.feedback_logs[-1]["message"] != message
            or (current_second - self.feedback_logs[-1]["time"]) > 2.0
        ):
            self.feedback_logs.append({"time": current_second, "message": message})