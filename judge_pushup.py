import math


class PushupEvaluator:
    """
    腕立て伏せのフォーム判定のみを行う（動画読み込み・姿勢推定は行わない）。
    呼び出し側が1フレームごとに関節座標を渡し、judge_frame()で判定する。

    landmarks の形式: MediaPipeのPoseLandmarker形式（例: PoseEstimator.current_landmarks）
      各要素が .x, .y 属性を持つオブジェクトのリスト（正規化座標 0〜1）で、
      添字がそのままMediaPipeのランドマーク番号と一致している想定。
      使用する番号: 11(左肩), 13(左肘), 15(左手首), 23(左腰), 27(左足首)
    """

    def __init__(self):
        self.feedback_logs = []  # [{"time": float, "message": str}]

    def judge_frame(self, current_second: float, landmarks) -> dict | None:
        """
        1フレーム分の関節座標(MediaPipe形式)を渡して判定する。
        戻り値: そのフレームで計算した指標値（グラフ描画などに利用可能）。判定不能な場合はNone。
        """
        if not landmarks or len(landmarks) <= 27:
            return None

        shoulder = (landmarks[11].x, landmarks[11].y)
        elbow = (landmarks[13].x, landmarks[13].y)
        hip = (landmarks[23].x, landmarks[23].y)
        ankle = (landmarks[27].x, landmarks[27].y)

        shoulder_angle = self._calculate_angle(elbow, shoulder, hip)
        back_angle = self._calculate_angle(shoulder, hip, ankle)

        if back_angle < 150 or back_angle > 210:
            self._append_log(
                "背中が曲がっています（一直線を意識してください）",
                current_second,
            )
        if shoulder_angle < 20:
            self._append_log(
                "身体を深く下ろしすぎています",
                current_second,
            )

        tilt = None
        if len(landmarks) > 15:
            wrist = (landmarks[15].x, landmarks[15].y)
            tilt = self._vertical_deviation(wrist, shoulder)

            if tilt > 20:
                self._append_log(
                    "手の位置が肩の真下からずれています",
                    current_second,
                )

        return {
            "shoulder_angle": shoulder_angle,
            "back_angle": back_angle,
            "wrist_tilt": tilt,
        }

    def get_result(self) -> dict:
        """全フレーム処理後に呼び出し、最終的な判定結果を取得する"""
        if not self.feedback_logs:
            summary = "GOOD: 大きなフォームの乱れは検出されませんでした"
        else:
            summary = f"WARN: {len(self.feedback_logs)}件の問題が検出されました"

        return {
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

    @staticmethod
    def _vertical_deviation(lower_point: tuple, upper_point: tuple) -> float:
        """
        下の点(手首)から上の点(肩)へのベクトルが、真上方向からどれだけ傾いているかを角度(度)で返す。
        0度に近いほど垂直（手が肩の真下にある）。
        """
        dx = upper_point[0] - lower_point[0]
        dy = upper_point[1] - lower_point[1]
        angle_rad = math.atan2(abs(dx), abs(dy))
        return math.degrees(angle_rad)

    def _append_log(self, message: str, current_second: float):
        """重複を避けつつログを追加（2秒以内の同一メッセージはスキップ）"""
        if (
            not self.feedback_logs
            or self.feedback_logs[-1]["message"] != message
            or (current_second - self.feedback_logs[-1]["time"]) > 2.0
        ):
            self.feedback_logs.append({"time": current_second, "message": message})