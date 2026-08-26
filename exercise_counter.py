from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pushup_counter import (
    LEFT_ELBOW_ANGLE,
    RIGHT_ELBOW_ANGLE,
    PushupCounter,
)
from squat_counter import LEFT_KNEE_ANGLE, RIGHT_KNEE_ANGLE, SquatCounter


FEATURE_COLUMNS = (
    "right_elbow_right_shoulder_right_hip",
    "left_elbow_left_shoulder_left_hip",
    "right_knee_mid_hip_left_knee",
    RIGHT_KNEE_ANGLE,
    LEFT_KNEE_ANGLE,
    RIGHT_ELBOW_ANGLE,
    LEFT_ELBOW_ANGLE,
)


@dataclass(frozen=True)
class ExercisePrediction:
    label: str
    confidence: float


class LSTMClassifier(nn.Module):
    """学習時と同じ構造のLSTMモデル。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


class ExerciseCounter:
    """LSTMで種目を判定し、対応するカウンターだけを動かす。"""

    def __init__(
        self,
        model_path: str | Path,
        sequence_length: int = 20,
        min_confidence: float = 0.6,
        confirm_frames: int = 3,
        replay_frames: int = 60,
        min_squat_knee_spread: float = 60.0,
        device: str | None = None,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_lengthは1以上にしてください。")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidenceは0以上1以下にしてください。")
        if confirm_frames < 1:
            raise ValueError("confirm_framesは1以上にしてください。")
        if replay_frames < sequence_length:
            raise ValueError(
                "replay_framesはsequence_length以上にしてください。"
            )
        if not 0.0 <= min_squat_knee_spread < 180.0:
            raise ValueError(
                "min_squat_knee_spreadは0以上180未満にしてください。"
            )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(
            Path(model_path),
            map_location=self.device,
            weights_only=True,
        )
        required = {
            "model_state_dict",
            "input_size",
            "hidden_size",
            "num_layers",
            "num_classes",
            "dropout",
            "class_names",
        }
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError(f"モデルに必要な情報がありません: {sorted(missing)}")
        if checkpoint["input_size"] != len(FEATURE_COLUMNS):
            raise ValueError(
                "モデルの特徴量数と角度の種類数が一致していません。"
            )

        self.class_names = tuple(checkpoint["class_names"])
        if not {"push_up", "squat"}.issubset(self.class_names):
            raise ValueError("モデルにpush_upとsquatのクラスが必要です。")

        self.model = LSTMClassifier(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            num_classes=checkpoint["num_classes"],
            dropout=checkpoint["dropout"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.sequence_length = sequence_length
        self.min_confidence = min_confidence
        self.confirm_frames = confirm_frames
        self.min_squat_knee_spread = min_squat_knee_spread
        self.squat_counter = SquatCounter()
        self.pushup_counter = PushupCounter()
        self.active_exercise: str | None = None
        self.last_prediction: ExercisePrediction | None = None
        self._features: deque[list[float]] = deque(maxlen=sequence_length)
        self._angle_history: deque[dict[str, float]] = deque(
            maxlen=replay_frames
        )
        self._candidate: str | None = None
        self._candidate_frames = 0

    @property
    def squat_count(self) -> int:
        return self.squat_counter.count

    @property
    def pushup_count(self) -> int:
        return self.pushup_counter.count

    def update_from_pose_angles(
        self, angles: Mapping[str, float] | None
    ) -> ExercisePrediction | None:
        if angles is not None:
            self._angle_history.append(dict(angles))

        prediction = self._predict(angles)
        self.last_prediction = prediction
        candidate = self._get_down_position_candidate(prediction, angles)
        switched = self._update_active_exercise(candidate)

        if switched:
            counter = self._replace_active_counter()
            for previous_angles in self._angle_history:
                counter.update_from_pose_angles(previous_angles)
        elif self.active_exercise == "squat":
            self.squat_counter.update_from_pose_angles(angles)
        elif self.active_exercise == "pushup":
            self.pushup_counter.update_from_pose_angles(angles)

        return prediction

    def _predict(
        self, angles: Mapping[str, float] | None
    ) -> ExercisePrediction | None:
        row = self._make_feature_row(angles)
        if row is None:
            self._features.clear()
            return None

        self._features.append(row)
        if len(self._features) < self.sequence_length:
            return None

        array = np.asarray(self._features, dtype=np.float32)
        tensor = torch.from_numpy(array).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]

        class_index = int(torch.argmax(probabilities).item())
        return ExercisePrediction(
            label=self.class_names[class_index],
            confidence=float(probabilities[class_index].item()),
        )

    def _get_down_position_candidate(
        self,
        prediction: ExercisePrediction | None,
        angles: Mapping[str, float] | None,
    ) -> str | None:
        if (
            prediction is None
            or prediction.confidence < self.min_confidence
            or angles is None
        ):
            return None

        label = "pushup" if prediction.label == "push_up" else prediction.label
        if label == "pushup" and self._both_below(
            angles,
            LEFT_ELBOW_ANGLE,
            RIGHT_ELBOW_ANGLE,
            self.pushup_counter.bent_angle,
        ):
            return label
        if label == "squat":
            knee_spread = angles.get("right_knee_mid_hip_left_knee")
            if (
                knee_spread is not None
                and np.isfinite(knee_spread)
                and knee_spread >= self.min_squat_knee_spread
                and self._both_below(
                    angles,
                    LEFT_KNEE_ANGLE,
                    RIGHT_KNEE_ANGLE,
                    self.squat_counter.squat_angle,
                )
            ):
                return label
        return None

    def _update_active_exercise(self, candidate: str | None) -> bool:
        if candidate is None or candidate == self.active_exercise:
            self._candidate = None
            self._candidate_frames = 0
            return False

        if candidate == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate = candidate
            self._candidate_frames = 1

        if self._candidate_frames < self.confirm_frames:
            return False

        self.active_exercise = candidate
        self._candidate = None
        self._candidate_frames = 0
        return True

    def _replace_active_counter(self) -> SquatCounter | PushupCounter:
        if self.active_exercise == "squat":
            previous_count = self.squat_counter.count
            self.squat_counter = SquatCounter()
            self.squat_counter.count = previous_count
            return self.squat_counter
        if self.active_exercise == "pushup":
            previous_count = self.pushup_counter.count
            self.pushup_counter = PushupCounter()
            self.pushup_counter.count = previous_count
            return self.pushup_counter
        raise RuntimeError("有効な種目が選択されていません。")

    @staticmethod
    def _make_feature_row(
        angles: Mapping[str, float] | None,
    ) -> list[float] | None:
        if angles is None:
            return None
        row = [angles.get(name) for name in FEATURE_COLUMNS]
        if any(
            value is None or not np.isfinite(value) or value <= 0.0
            for value in row
        ):
            return None
        return [float(value) for value in row]

    @staticmethod
    def _both_below(
        angles: Mapping[str, float],
        left_key: str,
        right_key: str,
        threshold: float,
    ) -> bool:
        left = angles.get(left_key)
        right = angles.get(right_key)
        return bool(
            left is not None
            and right is not None
            and np.isfinite(left)
            and np.isfinite(right)
            and 0.0 < left <= threshold
            and 0.0 < right <= threshold
        )
