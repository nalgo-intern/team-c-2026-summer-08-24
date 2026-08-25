class PushupEvaluator:
    """
    腕立て伏せのフォーム判定のみを行う。
    外部プログラムで計算された角度データを受け取り、judge_frame()で判定する。
    """

    def __init__(self):
        self.feedback_logs = []  # [{"time": float, "message": str}]

    def judge_frame(
        self,
        current_second: float,
        shoulder_angle: float | None = None,
        back_angle: float | None = None,
        tilt: float | None = None,
    ):
        """
        1フレーム分の角度・傾きデータを受け取って判定する。
        
        :param current_second: 現在の動画位置（秒）
        :param shoulder_angle: 肘-肩-腰の角度
        :param back_angle: 肩-腰-足首の角度
        :param tilt: 手首から肩へのベクトルの垂直からの傾き角度
        """
        if back_angle is not None:
            if back_angle < 150 or back_angle > 210:
                self._append_log(
                    "背中が曲がっています（一直線を意識してください）",
                    current_second,
                )

        if shoulder_angle is not None:
            if shoulder_angle < 20:
                self._append_log(
                    "身体を深く下ろしすぎています",
                    current_second,
                )

        if tilt is not None:
            if tilt > 20:
                self._append_log(
                    "手の位置が肩の真下からずれています",
                    current_second,
                )

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

    def _append_log(self, message: str, current_second: float):
        """重複を避けつつログを追加（2秒以内の同一メッセージはスキップ）"""
        if (
            not self.feedback_logs
            or self.feedback_logs[-1]["message"] != message
            or (current_second - self.feedback_logs[-1]["time"]) > 2.0
        ):
            self.feedback_logs.append({"time": current_second, "message": message})