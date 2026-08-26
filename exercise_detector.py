import cv2
import numpy as np
import torch

from pathlib import Path
from collections import deque

from pose_estimator_cpu import PoseEstimator
from lstm_model import LSTMClassifier


class ExerciseDetector:
    """
    腕立て・スクワット検出クラス

    判定値:
        0 = unknown
        1 = push_up
        2 = squat

    使用方法:

        detector = ExerciseDetector(
            model_path="models/lstm_binary_pushup_squat.pth",
            pose_model_path="pose_landmarker_lite.task",
            pushup_threshold=0.90,
            squat_threshold=0.10,
        )

        result = detector.process_frame(
            frame,
            timestamp_ms
        )

    result:
        0 = unknown
        1 = push_up
        2 = squat
    """

    # ============================================================
    # 初期化
    # ============================================================
    def __init__(
        self,
        model_path,
        pose_model_path,
        pushup_threshold=0.90,
        squat_threshold=0.10,
        required_consecutive=3,
        sequence_length=20,
    ):

        # --------------------------------------------------------
        # パス
        # --------------------------------------------------------
        self.model_path = Path(model_path)

        self.pose_model_path = Path(
            pose_model_path
        )

        # --------------------------------------------------------
        # LSTM設定
        # --------------------------------------------------------
        self.input_size = 7

        self.sequence_length = (
            sequence_length
        )

        self.hidden_size = 64

        self.num_layers = 2

        self.dropout = 0.3

        self.num_classes = 2

        self.class_names = [
            "push_up",
            "squat"
        ]

        # --------------------------------------------------------
        # クラスごとの信頼度閾値
        # --------------------------------------------------------
        self.pushup_threshold = (
            pushup_threshold
        )

        self.squat_threshold = (
            squat_threshold
        )

        # --------------------------------------------------------
        # 連続判定
        # --------------------------------------------------------
        self.required_consecutive = (
            required_consecutive
        )

        # --------------------------------------------------------
        # 身体姿勢判定
        # --------------------------------------------------------
        self.lying_y_threshold = 0.15

        self.standing_y_threshold = 0.15

        # --------------------------------------------------------
        # 特徴量
        # --------------------------------------------------------
        self.feature_columns = [
            "right_elbow_right_shoulder_right_hip",
            "left_elbow_left_shoulder_left_hip",
            "right_knee_mid_hip_left_knee",
            "right_hip_right_knee_right_ankle",
            "left_hip_left_knee_left_ankle",
            "right_wrist_right_elbow_right_shoulder",
            "left_wrist_left_elbow_left_shoulder"
        ]

        # --------------------------------------------------------
        # デバイス
        # --------------------------------------------------------
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "Device:",
            self.device
        )

        if torch.cuda.is_available():

            print(
                "GPU:",
                torch.cuda.get_device_name(0)
            )

        # --------------------------------------------------------
        # LSTMモデル
        # --------------------------------------------------------
        self.model, self.class_names = (
            self.load_model()
        )

        # --------------------------------------------------------
        # PoseEstimator
        # --------------------------------------------------------
        self.pose_estimator = PoseEstimator(
            model_path=str(
                self.pose_model_path
            )
        )

        print(
            "PoseEstimator初期化完了"
        )

        # ========================================================
        # ★ リアルタイム推論用の状態
        # ========================================================

        # --------------------------------------------------------
        # 20フレームバッファ
        # --------------------------------------------------------
        self.sequence_buffer = deque(
            maxlen=self.sequence_length
        )

        # --------------------------------------------------------
        # 連続判定
        # --------------------------------------------------------
        self.previous_prediction = None

        self.consecutive_count = 0

        # --------------------------------------------------------
        # 現在確定している運動
        # --------------------------------------------------------
        self.current_exercise = "unknown"

    # ============================================================
    # LSTMモデル読み込み
    # ============================================================
    def load_model(self):

        print()
        print(
            "========================================"
        )
        print(
            "学習済みLSTMモデル読み込み"
        )
        print(
            "========================================"
        )

        if not self.model_path.is_file():

            raise FileNotFoundError(
                f"学習済みモデルが見つかりません:\n"
                f"{self.model_path}"
            )

        print(
            "Model path:",
            self.model_path
        )

        print(
            "Model size:",
            self.model_path.stat().st_size,
            "bytes"
        )

        # --------------------------------------------------------
        # checkpoint
        # --------------------------------------------------------
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device
        )

        # --------------------------------------------------------
        # 学習時設定
        # --------------------------------------------------------
        input_size = checkpoint.get(
            "input_size",
            self.input_size
        )

        hidden_size = checkpoint.get(
            "hidden_size",
            self.hidden_size
        )

        num_layers = checkpoint.get(
            "num_layers",
            self.num_layers
        )

        num_classes = checkpoint.get(
            "num_classes",
            self.num_classes
        )

        dropout = checkpoint.get(
            "dropout",
            self.dropout
        )

        class_names = checkpoint.get(
            "class_names",
            self.class_names
        )

        # --------------------------------------------------------
        # モデル
        # --------------------------------------------------------
        model = LSTMClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        )

        # --------------------------------------------------------
        # 重み
        # --------------------------------------------------------
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.to(
            self.device
        )

        model.eval()

        print()
        print(
            "LSTMモデル読み込み完了"
        )

        print(
            "Classes:",
            class_names
        )

        print(
            "Input size:",
            input_size
        )

        print(
            "Hidden size:",
            hidden_size
        )

        print(
            "Layers:",
            num_layers
        )

        print(
            "Push-up threshold:",
            self.pushup_threshold
        )

        print(
            "Squat threshold:",
            self.squat_threshold
        )

        return (
            model,
            class_names
        )

    # ============================================================
    # 状態リセット
    #
    # 新しい動画・新しいカメラ入力を開始するときに使用
    # ============================================================
    def reset(self):

        self.sequence_buffer.clear()

        self.previous_prediction = None

        self.consecutive_count = 0

        self.current_exercise = "unknown"

    # ============================================================
    # 角度データ → numpy
    # ============================================================
    def angles_to_array(
        self,
        angles
    ):

        values = []

        for column in self.feature_columns:

            value = angles.get(
                column,
                0.0
            )

            values.append(
                float(value)
            )

        return np.array(
            values,
            dtype=np.float32
        )

    # ============================================================
    # LSTM推論
    # ============================================================
    def predict_sequence(
        self,
        sequence
    ):

        x = torch.tensor(
            sequence,
            dtype=torch.float32
        ).unsqueeze(0)

        x = x.to(
            self.device
        )

        with torch.no_grad():

            outputs = self.model(
                x
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, prediction = (
                probabilities.max(
                    dim=1
                )
            )

        prediction = (
            prediction.item()
        )

        confidence = (
            confidence.item()
        )

        probabilities = (
            probabilities[0]
            .cpu()
            .numpy()
        )

        return (
            prediction,
            confidence,
            probabilities
        )

    # ============================================================
    # 身体姿勢判定
    # ============================================================
    def classify_body_position(self):

        landmarks = (
            self.pose_estimator.current_landmarks
        )

        if landmarks is None:

            return "unknown"

        try:

            left_shoulder = landmarks[11]

            right_shoulder = landmarks[12]

            left_hip = landmarks[23]

            right_hip = landmarks[24]

        except IndexError:

            return "unknown"

        shoulder_y = (
            left_shoulder.y
            + right_shoulder.y
        ) / 2.0

        hip_y = (
            left_hip.y
            + right_hip.y
        ) / 2.0

        y_difference = abs(
            hip_y - shoulder_y
        )

        # --------------------------------------------------------
        # 寝転び
        # --------------------------------------------------------
        if (
            y_difference
            <
            self.lying_y_threshold
        ):

            return "lying"

        # --------------------------------------------------------
        # 立ち
        # --------------------------------------------------------
        if (
            shoulder_y < hip_y
            and
            y_difference
            >=
            self.standing_y_threshold
        ):

            return "standing"

        return "unknown"

    # ============================================================
    # 身体姿勢フィルタ
    # ============================================================
    def filter_prediction(
        self,
        predicted_class,
        body_position
    ):

        # --------------------------------------------------------
        # 立っている
        #
        # push_upを無効
        # --------------------------------------------------------
        if body_position == "standing":

            if predicted_class == "push_up":

                return "unknown"

            return predicted_class

        # --------------------------------------------------------
        # 寝転んでいる
        #
        # squatを無効
        # --------------------------------------------------------
        if body_position == "lying":

            if predicted_class == "squat":

                return "unknown"

            return predicted_class

        return "unknown"

    # ============================================================
    # クラスごとの閾値
    # ============================================================
    def get_confidence_threshold(
        self,
        predicted_class
    ):

        if predicted_class == "push_up":

            return self.pushup_threshold

        if predicted_class == "squat":

            return self.squat_threshold

        return 1.0

    # ============================================================
    # ★ 1フレーム処理
    #
    # 外部からリアルタイムに呼び出すためのメソッド
    #
    # 入力:
    #     frame
    #     timestamp_ms
    #
    # 戻り値:
    #     0 = unknown
    #     1 = push_up
    #     2 = squat
    # ============================================================
    def process_frame(
        self,
        frame,
        timestamp_ms
    ):

        # ========================================================
        # MediaPipe Pose
        # ========================================================
        angles = (
            self.pose_estimator.process_frame(
                frame,
                timestamp_ms
            )
        )

        # ========================================================
        # 身体姿勢
        # ========================================================
        body_position = (
            self.classify_body_position()
        )

        # ========================================================
        # 7特徴量
        # ========================================================
        feature = (
            self.angles_to_array(
                angles
            )
        )

        # ========================================================
        # バッファへ追加
        # ========================================================
        self.sequence_buffer.append(
            feature
        )

        # ========================================================
        # 20フレーム未満
        # ========================================================
        if (
            len(self.sequence_buffer)
            <
            self.sequence_length
        ):

            return 0

        # ========================================================
        # LSTM入力
        # ========================================================
        sequence = np.array(
            self.sequence_buffer,
            dtype=np.float32
        )

        # ========================================================
        # LSTM推論
        # ========================================================
        (
            prediction,
            confidence,
            probabilities
        ) = self.predict_sequence(
            sequence
        )

        predicted_class = (
            self.class_names[
                prediction
            ]
        )

        # ========================================================
        # クラス別閾値
        # ========================================================
        threshold = (
            self.get_confidence_threshold(
                predicted_class
            )
        )

        # ========================================================
        # 信頼度不足
        # ========================================================
        if confidence < threshold:

            self.consecutive_count = 0

            self.previous_prediction = (
                None
            )

            return self._get_result_code()

        # ========================================================
        # 身体姿勢フィルタ
        # ========================================================
        detected_class = (
            self.filter_prediction(
                predicted_class,
                body_position
            )
        )

        # ========================================================
        # 無効
        # ========================================================
        if detected_class == "unknown":

            self.consecutive_count = 0

            self.previous_prediction = (
                None
            )

            # ----------------------------------------------------
            # 身体姿勢がunknownの場合は
            # 現在の運動も解除
            # ----------------------------------------------------
            if body_position == "unknown":

                self.current_exercise = (
                    "unknown"
                )

            # ----------------------------------------------------
            # 立ち状態でpush_up
            # ----------------------------------------------------
            if (
                body_position == "standing"
                and
                self.current_exercise == "push_up"
            ):

                self.current_exercise = (
                    "unknown"
                )

            # ----------------------------------------------------
            # 寝転び状態でsquat
            # ----------------------------------------------------
            if (
                body_position == "lying"
                and
                self.current_exercise == "squat"
            ):

                self.current_exercise = (
                    "unknown"
                )

            return self._get_result_code()

        # ========================================================
        # 連続判定
        # ========================================================
        if (
            detected_class
            ==
            self.previous_prediction
        ):

            self.consecutive_count += 1

        else:

            self.consecutive_count = 1

        self.previous_prediction = (
            detected_class
        )

        # ========================================================
        # 一定回数連続したら確定
        # ========================================================
        if (
            self.consecutive_count
            >=
            self.required_consecutive
        ):

            self.current_exercise = (
                detected_class
            )

        # ========================================================
        # 最終フィルタ
        # ========================================================
        if (
            body_position == "unknown"
        ):

            self.current_exercise = (
                "unknown"
            )

            self.consecutive_count = 0

            self.previous_prediction = (
                None
            )

        # --------------------------------------------------------
        # standing + push_up
        # --------------------------------------------------------
        if (
            body_position == "standing"
            and
            self.current_exercise == "push_up"
        ):

            self.current_exercise = (
                "unknown"
            )

        # --------------------------------------------------------
        # lying + squat
        # --------------------------------------------------------
        if (
            body_position == "lying"
            and
            self.current_exercise == "squat"
        ):

            self.current_exercise = (
                "unknown"
            )

        # ========================================================
        # 0 / 1 / 2
        # ========================================================
        return self._get_result_code()

    # ============================================================
    # 結果を0 / 1 / 2に変換
    # ============================================================
    def _get_result_code(self):

        if self.current_exercise == "push_up":

            return 1

        if self.current_exercise == "squat":

            return 2

        return 0

    # ============================================================
    # 表示サイズ
    # ============================================================
    def calculate_display_size(
        self,
        width,
        height
    ):

        aspect_ratio = (
            width / height
        )

        if (
            width <= 1280
            and
            height <= 720
        ):

            return (
                width,
                height
            )

        if aspect_ratio >= 1:

            display_width = 1280

            display_height = int(
                display_width
                / aspect_ratio
            )

        else:

            display_height = 720

            display_width = int(
                display_height
                * aspect_ratio
            )

        return (
            display_width,
            display_height
        )

    # ============================================================
    # ★ 動画処理
    #
    # 動画表示はしない
    #
    # 結果動画だけ保存する
    #
    # 戻り値:
    #
    #     numpy.ndarray
    #
    #     [0, 0, 1, 1, 1, 0, 2, 2, ...]
    # ============================================================
    def process_video(
        self,
        video_path,
        output_path
    ):

        video_path = Path(
            video_path
        )

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # ========================================================
        # 新しい動画なので状態リセット
        # ========================================================
        self.reset()

        # ========================================================
        # 動画
        # ========================================================
        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():

            raise RuntimeError(
                f"動画を開けません:\n"
                f"{video_path}"
            )

        # ========================================================
        # 動画情報
        # ========================================================
        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:

            fps = 30.0

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        print()
        print(
            "========================================"
        )

        print(
            "動画処理開始"
        )

        print(
            "Input:",
            video_path
        )

        print(
            "FPS:",
            fps
        )

        print(
            "Resolution:",
            width,
            "x",
            height
        )

        print(
            "========================================"
        )

        # ========================================================
        # 出力動画
        # ========================================================
        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (
                width,
                height
            )
        )

        if not writer.isOpened():

            cap.release()

            raise RuntimeError(
                f"出力動画を作成できません:\n"
                f"{output_path}"
            )

        # ========================================================
        # フレームごとの結果
        # ========================================================
        results = []

        # ========================================================
        # フレーム
        # ========================================================
        frame_index = 0

        while True:

            ret, frame = cap.read()

            if not ret:

                break

            # ====================================================
            # timestamp
            # ====================================================
            timestamp_ms = (
                frame_index
                / fps
                * 1000.0
            )

            # ====================================================
            # ★ 1フレーム処理
            # ====================================================
            result = self.process_frame(
                frame,
                timestamp_ms
            )

            # ====================================================
            # ★ 結果保存
            # ====================================================
            results.append(
                result
            )

            # ====================================================
            # 動画保存用表示
            #
            # 画面には表示しない
            # ====================================================
            if result == 1:

                text = "push_up"

            elif result == 2:

                text = "squat"

            else:

                text = "unknown"

            cv2.putText(
                frame,
                text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            # ====================================================
            # Pose骨格
            # ====================================================
            frame = (
                self.pose_estimator.draw_landmarks(
                    frame
                )
            )

            # ====================================================
            # 結果動画保存
            # ====================================================
            writer.write(
                frame
            )

            frame_index += 1

        # ========================================================
        # 終了
        # ========================================================
        cap.release()

        writer.release()

        # ========================================================
        # numpy配列
        # ========================================================
        results = np.array(
            results,
            dtype=np.int64
        )

        print()
        print(
            "========================================"
        )

        print(
            "動画処理終了"
        )

        print(
            "Output:",
            output_path
        )

        print(
            "Number of frames:",
            len(results)
        )

        print(
            "Result shape:",
            results.shape
        )

        print(
            "========================================"
        )

        return results