from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.pose_estimator import PoseEstimator


LEFT_KNEE_ANGLE = "left_hip_left_knee_left_ankle"
RIGHT_KNEE_ANGLE = "right_hip_right_knee_right_ankle"


@dataclass
class ExponentialMovingAverage:
    """フレーム間の小さな角度の揺れを抑える。"""

    alpha: float = 0.3
    value: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alphaは0より大きく1以下にしてください。")

    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = float(new_value)
        else:
            self.value = self.alpha * float(new_value) + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


@dataclass
class SquatCounter:
    """膝角度が UP -> DOWN -> UP と変化したときに1回加算する。"""

    stand_angle: float = 130.0
    squat_angle: float = 120.0
    stable_frames: int = 3
    count: int = field(default=0, init=False)
    stage: str = field(default="WAITING", init=False)
    _candidate: str | None = field(default=None, init=False, repr=False)
    _candidate_frames: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.squat_angle < self.stand_angle < 180.0:
            raise ValueError("0 < squat_angle < stand_angle < 180 にしてください。")
        if self.stable_frames < 1:
            raise ValueError("stable_framesは1以上にしてください。")

    def update(self, angle: float | None) -> str:
        """新しい膝角度を受け取り、現在の段階を返す。"""
        posture: str | None = None
        if angle is not None and np.isfinite(angle) and angle >= self.stand_angle:
            posture = "UP"
        elif angle is not None and np.isfinite(angle) and 0.0 < angle <= self.squat_angle:
            posture = "DOWN"

        return self._update_posture(posture)

    def update_bilateral(
        self, left_angle: float | None, right_angle: float | None
    ) -> str:
        """左右両方の膝角度が条件を満たしたときだけ状態を変更する。"""
        posture: str | None = None
        if (
            left_angle is not None
            and right_angle is not None
            and np.isfinite(left_angle)
            and np.isfinite(right_angle)
            and left_angle > 0.0
            and right_angle > 0.0
        ):
            if min(left_angle, right_angle) >= self.stand_angle:
                posture = "UP"
            elif max(left_angle, right_angle) <= self.squat_angle:
                posture = "DOWN"

        return self._update_posture(posture)

    def update_from_pose_angles(
        self, angles: Mapping[str, float] | None
    ) -> str:
        """PoseEstimatorが返した角度情報から左右の膝角度を取得する。"""
        if angles is None:
            return self.update_bilateral(None, None)

        return self.update_bilateral(
            angles.get(LEFT_KNEE_ANGLE),
            angles.get(RIGHT_KNEE_ANGLE),
        )

    def update_from_frame(
        self, pose_estimator: PoseEstimator, frame: Any
    ) -> str:
        """PoseEstimatorでフレームを処理し、その計算結果で状態を更新する。"""
        return self.update_from_pose_angles(pose_estimator.process_frame(frame))

    def _update_posture(self, posture: str | None) -> str:
        if posture is None:
            self._clear_candidate()
            return self.stage

        # 2つのしきい値の間では状態を変えない。
        if posture == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate = posture
            self._candidate_frames = 1

        if self._candidate_frames < self.stable_frames:
            return self.stage

        if self.stage == "WAITING" and posture == "UP":
            self.stage = "UP"
        elif self.stage == "UP" and posture == "DOWN":
            self.stage = "DOWN"
        elif self.stage == "DOWN" and posture == "UP":
            self.count += 1
            self.stage = "UP"

        return self.stage

    def _clear_candidate(self) -> None:
        self._candidate = None
        self._candidate_frames = 0