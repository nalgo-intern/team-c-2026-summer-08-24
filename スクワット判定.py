class SquatEvaluator:
    """
    スクワットのフォーム判定のみを行う。
    外部プログラムで計算された膝の角度データを受け取り、judge_frame()で判定する。
    """

    def __init__(self):
        self.feedback_logs = []  # [{"time": float, "message": str}]
        self.min_knee_angle = 180.0

    def judge_frame(self, current_second: float, knee_angle: float | None) -> dict | None:
        """
        1フレーム分の膝の角度を渡して判定する。
        
        :param current_second: 現在の動画位置（秒）
        :param knee_angle: 腰-膝-足首の角度
        :return: グラフ描画用の辞書（判定不能な場合はNone）
        """
        if knee_angle is None:
            return None

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

    def _append_log(self, message: str, current_second: float):
        """重複を避けつつログを追加（2秒以内の同一メッセージはスキップ）"""
        if (
            not self.feedback_logs
            or self.feedback_logs[-1]["message"] != message
            or (current_second - self.feedback_logs[-1]["time"]) > 2.0
        ):
            self.feedback_logs.append({"time": current_second, "message": message})