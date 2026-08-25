from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np


Point: TypeAlias = Sequence[float]


def calculate_angle(point_a: Point, vertex: Point, point_c: Point) -> float:
    """2次元または3次元の3点から、vertexを頂点とする角度を返す。"""
    vector_a = np.asarray(point_a, dtype=float) - np.asarray(vertex, dtype=float)
    vector_c = np.asarray(point_c, dtype=float) - np.asarray(vertex, dtype=float)

    if vector_a.ndim != 1 or vector_c.ndim != 1 or vector_a.shape != vector_c.shape:
        raise ValueError("3点は同じ次元の座標にしてください。")

    length_product = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_c))
    if length_product == 0.0:
        raise ValueError("同じ座標の点があり、角度を計算できません。")

    cosine = float(np.dot(vector_a, vector_c) / length_product)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


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

    stand_angle: float = 120.0
    squat_angle: float = 100.0
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
        if angle is not None and angle >= self.stand_angle:
            posture = "UP"
        elif angle is not None and angle <= self.squat_angle:
            posture = "DOWN"

        return self._update_posture(posture)

    def update_bilateral(
        self, left_angle: float | None, right_angle: float | None
    ) -> str:
        """左右両方の膝角度が条件を満たしたときだけ状態を変更する。"""
        posture: str | None = None
        if left_angle is not None and right_angle is not None:
            if min(left_angle, right_angle) >= self.stand_angle:
                posture = "UP"
            elif max(left_angle, right_angle) <= self.squat_angle:
                posture = "DOWN"

        return self._update_posture(posture)

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
