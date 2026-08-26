import sys
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch


# ============================================================
# パス設定
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(BASE_DIR))


# ============================================================
# 他ファイルからクラスを読み込み
# ============================================================

from pose_estimator import PoseEstimator
from lstm_model import LSTMClassifier


# ============================================================
# 入力動画
# ============================================================

VIDEO_PATH = Path(
    r"C:\Users\池田一颯\internship\video\YTDown.com_YouTube_Media_i-2RWGuSyN0_002_720p.mp4"
)


# ============================================================
# MediaPipe Poseモデル
# ============================================================

POSE_MODEL_PATH = (
    BASE_DIR
    / "pose_landmarker_lite.task"
)


# ============================================================
# 学習済みLSTMモデル
# ============================================================

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "lstm_binary_pushup_squat.pth"
)


# ============================================================
# 出力動画
# ============================================================

OUTPUT_DIR = (
    BASE_DIR
    / "output"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_VIDEO_PATH = (
    OUTPUT_DIR
    / "result.mp4"
)


# ============================================================
# LSTM設定
# ============================================================

INPUT_SIZE = 7
SEQUENCE_LENGTH = 20
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
NUM_CLASSES = 2

CLASS_NAMES = [
    "push_up",
    "squat"
]


# ============================================================
# 検出設定
# ============================================================

CONFIDENCE_THRESHOLD = 0.70

REQUIRED_CONSECUTIVE = 3


# ============================================================
# 使用する特徴量
# ============================================================

FEATURE_COLUMNS = [
    "right_elbow_right_shoulder_right_hip",
    "left_elbow_left_shoulder_left_hip",
    "right_knee_mid_hip_left_knee",
    "right_hip_right_knee_right_ankle",
    "left_hip_left_knee_left_ankle",
    "right_wrist_right_elbow_right_shoulder",
    "left_wrist_left_elbow_left_shoulder"
]


# ============================================================
# デバイス
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 学習済みモデル読み込み
# ============================================================

def load_model():

    print()
    print("========================================")
    print("学習済みLSTMモデル読み込み")
    print("========================================")

    # --------------------------------------------------------
    # モデルファイル確認
    # --------------------------------------------------------

    if not MODEL_PATH.is_file():

        raise FileNotFoundError(
            f"学習済みモデルが見つかりません:\n"
            f"{MODEL_PATH}"
        )

    print("Model path:")
    print(MODEL_PATH)

    print(
        "Model size:",
        MODEL_PATH.stat().st_size,
        "bytes"
    )

    # --------------------------------------------------------
    # checkpoint読み込み
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # 学習時の設定
    # --------------------------------------------------------

    input_size = checkpoint.get(
        "input_size",
        INPUT_SIZE
    )

    hidden_size = checkpoint.get(
        "hidden_size",
        HIDDEN_SIZE
    )

    num_layers = checkpoint.get(
        "num_layers",
        NUM_LAYERS
    )

    num_classes = checkpoint.get(
        "num_classes",
        NUM_CLASSES
    )

    dropout = checkpoint.get(
        "dropout",
        DROPOUT
    )

    class_names = checkpoint.get(
        "class_names",
        CLASS_NAMES
    )

    # --------------------------------------------------------
    # モデル構造を作成
    # --------------------------------------------------------

    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    )

    # --------------------------------------------------------
    # 学習済み重みを読み込み
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # GPU / CPU
    # --------------------------------------------------------

    model.to(DEVICE)

    # ========================================================
    # 重要
    # ========================================================
    # eval()なので学習は行わない
    # ========================================================

    model.eval()

    print()
    print("LSTMモデル読み込み完了")
    print("Classes:", class_names)
    print("Input size:", input_size)
    print("Hidden size:", hidden_size)
    print("Layers:", num_layers)

    return model, class_names


# ============================================================
# 角度データ → numpy
# ============================================================

def angles_to_array(angles):

    values = []

    for column in FEATURE_COLUMNS:

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
    model,
    sequence
):

    # sequence
    #
    # (20, 7)
    #
    # ↓
    #
    # (1, 20, 7)

    x = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0)

    x = x.to(DEVICE)

    # ========================================================
    # 推論のみ
    # ========================================================

    with torch.no_grad():

        outputs = model(x)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = (
            probabilities.max(
                dim=1
            )
        )

    prediction = prediction.item()

    confidence = confidence.item()

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
# 動画処理
# ============================================================

def process_video():

    print()
    print("========================================")
    print("動画処理開始")
    print("========================================")

    # ========================================================
    # 動画確認
    # ========================================================

    if not VIDEO_PATH.is_file():

        raise FileNotFoundError(
            f"動画が見つかりません:\n"
            f"{VIDEO_PATH}"
        )

    print()
    print("入力動画:")
    print(VIDEO_PATH)

    # ========================================================
    # LSTMモデル読み込み
    # ========================================================

    model, class_names = load_model()

    # ========================================================
    # Poseモデル確認
    # ========================================================

    print()
    print("========================================")
    print("Poseモデル")
    print("========================================")

    print(
        "Path:",
        POSE_MODEL_PATH
    )

    # --------------------------------------------------------
    # 既存ファイルを確認
    # --------------------------------------------------------

    if POSE_MODEL_PATH.is_file():

        print(
            "既存のPoseモデルを使用します"
        )

        print(
            "Size:",
            POSE_MODEL_PATH.stat().st_size,
            "bytes"
        )

    else:

        print(
            "Poseモデルがありません。"
        )

        print(
            "PoseEstimator側でダウンロードします。"
        )

    # ========================================================
    # PoseEstimator
    # ========================================================
    #
    # pose_estimator.py は変更しない
    #
    # 既存のファイルをmodel_pathとして渡す
    #
    # ========================================================

    pose_estimator = PoseEstimator(
        model_path=str(
            POSE_MODEL_PATH
        )
    )

    print(
        "PoseEstimator初期化完了"
    )

    # ========================================================
    # 動画
    # ========================================================

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"動画を開けません:\n"
            f"{VIDEO_PATH}"
        )

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
    print("FPS:", fps)

    print(
        "Resolution:",
        width,
        "x",
        height
    )

    # ========================================================
    # 出力動画
    # ========================================================

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO_PATH),
        fourcc,
        fps,
        (
            width,
            height
        )
    )

    # ========================================================
    # 20フレームバッファ
    # ========================================================

    sequence_buffer = deque(
        maxlen=SEQUENCE_LENGTH
    )

    # ========================================================
    # 連続判定
    # ========================================================

    previous_prediction = None

    consecutive_count = 0

    current_exercise = "unknown"

    # ========================================================
    # フレーム処理
    # ========================================================

    frame_index = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ----------------------------------------------------
        # timestamp
        # ----------------------------------------------------

        timestamp_ms = (
            frame_index
            / fps
            * 1000.0
        )

        # ----------------------------------------------------
        # MediaPipe Pose
        # ----------------------------------------------------

        angles = pose_estimator.process_frame(
            frame,
            timestamp_ms
        )

        # ----------------------------------------------------
        # 7特徴量
        # ----------------------------------------------------

        feature = angles_to_array(
            angles
        )

        # ----------------------------------------------------
        # バッファ
        # ----------------------------------------------------

        sequence_buffer.append(
            feature
        )

        confidence = 0.0

        probabilities = None

        # ====================================================
        # 20フレーム貯まったら推論
        # ====================================================

        if len(sequence_buffer) >= SEQUENCE_LENGTH:

            sequence = np.array(
                sequence_buffer,
                dtype=np.float32
            )

            # ------------------------------------------------
            # LSTM推論
            # ------------------------------------------------

            (
                prediction,
                confidence,
                probabilities
            ) = predict_sequence(
                model,
                sequence
            )

            predicted_class = (
                class_names[prediction]
            )

            # ------------------------------------------------
            # 信頼度判定
            # ------------------------------------------------

            if confidence < CONFIDENCE_THRESHOLD:

                detected_class = "unknown"

                consecutive_count = 0

                previous_prediction = None

            else:

                detected_class = (
                    predicted_class
                )

                # --------------------------------------------
                # 連続判定
                # --------------------------------------------

                if (
                    detected_class
                    == previous_prediction
                ):

                    consecutive_count += 1

                else:

                    consecutive_count = 1

                previous_prediction = (
                    detected_class
                )

                # --------------------------------------------
                # 一定回数連続したら確定
                # --------------------------------------------

                if (
                    consecutive_count
                    >= REQUIRED_CONSECUTIVE
                ):

                    current_exercise = (
                        detected_class
                    )

        # ====================================================
        # 表示
        # ====================================================

        text = (
            f"{current_exercise}  "
            f"{confidence:.2f}"
        )

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
        # 確率表示
        # ====================================================

        if probabilities is not None:

            y_position = 90

            for i, class_name in enumerate(
                class_names
            ):

                probability = (
                    probabilities[i]
                )

                probability_text = (
                    f"{class_name}: "
                    f"{probability:.3f}"
                )

                cv2.putText(
                    frame,
                    probability_text,
                    (
                        30,
                        y_position
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                y_position += 30

        else:

            cv2.putText(
                frame,
                "Collecting...",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        # ====================================================
        # MediaPipe骨格描画
        # ====================================================

        frame = pose_estimator.draw_landmarks(
            frame
        )

        # ====================================================
        # 出力動画
        # ====================================================

        writer.write(
            frame
        )

        # ====================================================
        # 画面表示
        # ====================================================

        cv2.imshow(
            "Exercise Detection",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        frame_index += 1

    # ========================================================
    # 終了
    # ========================================================

    cap.release()

    writer.release()

    cv2.destroyAllWindows()

    print()
    print("========================================")
    print("動画処理終了")
    print("========================================")

    print(
        "入力:",
        VIDEO_PATH
    )

    print(
        "出力:",
        OUTPUT_VIDEO_PATH
    )


# ============================================================
# main
# ============================================================

if __name__ == "__main__":

    process_video()