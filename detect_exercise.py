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
from pose_estimator_cpu import PoseEstimator
from lstm_model import LSTMClassifier


# ============================================================
# 入力動画
# ============================================================
VIDEO_PATH = Path(
    r"input_video\IMG_6477.mp4"
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
# LSTM検出設定
# ============================================================

# ============================================================
# クラスごとの信頼度閾値
# ============================================================
#
# 腕立てとスクワットで別々に設定できます。
#
# 例:
#   腕立て  0.90
#   スクワット 0.80
#
# ============================================================

PUSHUP_CONFIDENCE_THRESHOLD = 0.90

SQUAT_CONFIDENCE_THRESHOLD = 0.1


# ============================================================
# 連続判定設定
# ============================================================
REQUIRED_CONSECUTIVE = 3


# ============================================================
# 立ち / 寝転び判定設定
# ============================================================

# 肩と腰のY座標差がこれ以下なら寝転んでいる
LYING_Y_THRESHOLD = 0.15


# 肩が腰より十分上なら立っている
STANDING_Y_THRESHOLD = 0.15


# ============================================================
# 表示設定
# ============================================================
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720


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

print(
    "Device:",
    DEVICE
)

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

    print(
        "Model path:"
    )

    print(
        MODEL_PATH
    )

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
    # モデル構造
    # --------------------------------------------------------
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    )

    # --------------------------------------------------------
    # 学習済み重み
    # --------------------------------------------------------
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # GPU / CPU
    # --------------------------------------------------------
    model.to(DEVICE)

    # --------------------------------------------------------
    # 推論モード
    # --------------------------------------------------------
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
        PUSHUP_CONFIDENCE_THRESHOLD
    )

    print(
        "Squat threshold:",
        SQUAT_CONFIDENCE_THRESHOLD
    )

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

    # --------------------------------------------------------
    # (20, 7)
    #
    # ↓
    #
    # (1, 20, 7)
    # --------------------------------------------------------
    x = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0)

    x = x.to(DEVICE)

    # --------------------------------------------------------
    # 推論
    # --------------------------------------------------------
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
# 身体姿勢判定
#
# current_landmarksから肩・腰の座標を取得
#
# 戻り値:
#
#     "standing"
#     "lying"
#     "unknown"
#
# ============================================================
def classify_body_position(
    pose_estimator
):

    landmarks = (
        pose_estimator.current_landmarks
    )

    # --------------------------------------------------------
    # 骨格が検出できない
    # --------------------------------------------------------
    if landmarks is None:
        return "unknown"

    # --------------------------------------------------------
    # MediaPipe Pose
    #
    # 左肩  = 11
    # 右肩  = 12
    #
    # 左腰  = 23
    # 右腰  = 24
    # --------------------------------------------------------
    try:

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        left_hip = landmarks[23]
        right_hip = landmarks[24]

    except IndexError:

        return "unknown"

    # --------------------------------------------------------
    # 左右の肩の中心
    # --------------------------------------------------------
    shoulder_y = (
        left_shoulder.y
        + right_shoulder.y
    ) / 2.0

    # --------------------------------------------------------
    # 左右の腰の中心
    # --------------------------------------------------------
    hip_y = (
        left_hip.y
        + right_hip.y
    ) / 2.0

    # --------------------------------------------------------
    # 肩と腰のY座標差
    # --------------------------------------------------------
    y_difference = abs(
        hip_y - shoulder_y
    )

    # ========================================================
    # 寝転んでいる
    #
    # 肩と腰のY座標が近い
    # ========================================================
    if (
        y_difference
        <
        LYING_Y_THRESHOLD
    ):

        return "lying"

    # ========================================================
    # 立っている
    #
    # 肩が腰より上
    # ========================================================
    if (
        shoulder_y < hip_y
        and
        y_difference >= STANDING_Y_THRESHOLD
    ):

        return "standing"

    # ========================================================
    # 中間
    # ========================================================
    return "unknown"


# ============================================================
# 姿勢によるLSTM結果フィルタ
#
# standing:
#     push_up → unknown
#
# lying:
#     squat → unknown
#
# ============================================================
def filter_prediction_by_body_position(
    predicted_class,
    body_position
):

    # --------------------------------------------------------
    # 立っている場合
    #
    # 腕立ては無効
    # --------------------------------------------------------
    if body_position == "standing":

        if predicted_class == "push_up":
            return "unknown"

        return predicted_class

    # --------------------------------------------------------
    # 寝転んでいる場合
    #
    # スクワットは無効
    # --------------------------------------------------------
    if body_position == "lying":

        if predicted_class == "squat":
            return "unknown"

        return predicted_class

    # --------------------------------------------------------
    # unknownの場合
    #
    # 安全のため両方とも無効
    # --------------------------------------------------------
    return "unknown"


# ============================================================
# 表示サイズ計算
#
# 入力動画と同じアスペクト比
# ============================================================
def calculate_display_size(
    width,
    height
):

    aspect_ratio = (
        width / height
    )

    if (
        width <= MAX_DISPLAY_WIDTH
        and
        height <= MAX_DISPLAY_HEIGHT
    ):

        return (
            width,
            height
        )

    # --------------------------------------------------------
    # 横長
    # --------------------------------------------------------
    if aspect_ratio >= 1:

        display_width = (
            MAX_DISPLAY_WIDTH
        )

        display_height = int(
            display_width
            / aspect_ratio
        )

    # --------------------------------------------------------
    # 縦長
    # --------------------------------------------------------
    else:

        display_height = (
            MAX_DISPLAY_HEIGHT
        )

        display_width = int(
            display_height
            * aspect_ratio
        )

    return (
        display_width,
        display_height
    )


# ============================================================
# 表示用リサイズ
# ============================================================
def resize_for_display(
    frame,
    display_width,
    display_height
):

    return cv2.resize(
        frame,
        (
            display_width,
            display_height
        ),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# 身体姿勢の表示
# ============================================================
def draw_body_position(
    frame,
    body_position
):

    if body_position == "standing":

        text = "BODY: STANDING"

    elif body_position == "lying":

        text = "BODY: LYING"

    else:

        text = "BODY: UNKNOWN"

    cv2.putText(
        frame,
        text,
        (30, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
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
    print(
        "入力動画:"
    )

    print(
        VIDEO_PATH
    )

    # ========================================================
    # LSTMモデル
    # ========================================================
    model, class_names = load_model()

    # ========================================================
    # Poseモデル
    # ========================================================
    print()
    print("========================================")
    print("Poseモデル")
    print("========================================")

    print(
        "Path:",
        POSE_MODEL_PATH
    )

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
    #
    # CPU版
    #
    # クラス構造は変更しない
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
        "FPS:",
        fps
    )

    print(
        "Input resolution:",
        width,
        "x",
        height
    )

    print(
        "Input aspect ratio:",
        f"{width / height:.4f}"
    )

    # ========================================================
    # 表示サイズ
    # ========================================================
    (
        display_width,
        display_height
    ) = calculate_display_size(
        width,
        height
    )

    print(
        "Display resolution:",
        display_width,
        "x",
        display_height
    )

    print(
        "Display aspect ratio:",
        f"{display_width / display_height:.4f}"
    )

    # ========================================================
    # 出力動画
    #
    # 入力と同じ解像度
    # ========================================================
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(
            OUTPUT_VIDEO_PATH
        ),
        fourcc,
        fps,
        (
            width,
            height
        )
    )

    if not writer.isOpened():

        raise RuntimeError(
            f"出力動画を作成できません:\n"
            f"{OUTPUT_VIDEO_PATH}"
        )

    # ========================================================
    # 20フレームバッファ
    # ========================================================
    sequence_buffer = deque(
        maxlen=SEQUENCE_LENGTH
    )

    # ========================================================
    # LSTM連続判定
    # ========================================================
    previous_prediction = None

    consecutive_count = 0

    current_exercise = "unknown"

    # ========================================================
    # OpenCVウィンドウ
    # ========================================================
    WINDOW_NAME = (
        "Exercise Detection"
    )

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        display_width,
        display_height
    )

    # ========================================================
    # フレーム処理
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
        # MediaPipe Pose
        # ====================================================
        angles = pose_estimator.process_frame(
            frame,
            timestamp_ms
        )

        # ====================================================
        # 身体姿勢判定
        # ====================================================
        body_position = classify_body_position(
            pose_estimator
        )

        # ====================================================
        # 7特徴量
        # ====================================================
        feature = angles_to_array(
            angles
        )

        # ====================================================
        # バッファ
        # ====================================================
        sequence_buffer.append(
            feature
        )

        confidence = 0.0
        probabilities = None

        # ====================================================
        # 20フレーム貯まったら推論
        # ====================================================
        if (
            len(sequence_buffer)
            >= SEQUENCE_LENGTH
        ):

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

            # =================================================
            # ★ クラスごとの信頼度閾値
            # =================================================
            if predicted_class == "push_up":

                confidence_threshold = (
                    PUSHUP_CONFIDENCE_THRESHOLD
                )

            elif predicted_class == "squat":

                confidence_threshold = (
                    SQUAT_CONFIDENCE_THRESHOLD
                )

            else:

                confidence_threshold = 1.0

            # =================================================
            # 信頼度判定
            # =================================================
            if confidence < confidence_threshold:

                detected_class = "unknown"

                consecutive_count = 0

                previous_prediction = None

            else:

                # =================================================
                # 身体姿勢によるフィルタ
                # =================================================
                detected_class = (
                    filter_prediction_by_body_position(
                        predicted_class,
                        body_position
                    )
                )

                # =================================================
                # unknownなら連続判定をリセット
                # =================================================
                if detected_class == "unknown":

                    consecutive_count = 0

                    previous_prediction = None

                else:

                    # =========================================
                    # 連続判定
                    # =========================================
                    if (
                        detected_class
                        ==
                        previous_prediction
                    ):

                        consecutive_count += 1

                    else:

                        consecutive_count = 1

                    previous_prediction = (
                        detected_class
                    )

                    # =========================================
                    # 一定回数連続したら確定
                    # =========================================
                    if (
                        consecutive_count
                        >= REQUIRED_CONSECUTIVE
                    ):

                        current_exercise = (
                            detected_class
                        )

        # ====================================================
        # 身体姿勢がunknownになったら
        #
        # 現在の運動判定も解除
        # ====================================================
        if body_position == "unknown":

            current_exercise = "unknown"

            consecutive_count = 0

            previous_prediction = None

        # ====================================================
        # 立っている場合
        #
        # push_upを絶対に表示しない
        # ====================================================
        if (
            body_position == "standing"
            and
            current_exercise == "push_up"
        ):

            current_exercise = "unknown"

        # ====================================================
        # 寝転んでいる場合
        #
        # squatを絶対に表示しない
        # ====================================================
        if (
            body_position == "lying"
            and
            current_exercise == "squat"
        ):

            current_exercise = "unknown"

        # ====================================================
        # 判定結果表示
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
        # 現在使用している閾値を表示
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

            # ------------------------------------------------
            # 現在の予測クラスに対応する閾値を表示
            # ------------------------------------------------
            if predicted_class == "push_up":

                threshold_text = (
                    f"Threshold: "
                    f"{PUSHUP_CONFIDENCE_THRESHOLD:.2f}"
                )

            elif predicted_class == "squat":

                threshold_text = (
                    f"Threshold: "
                    f"{SQUAT_CONFIDENCE_THRESHOLD:.2f}"
                )

            else:

                threshold_text = (
                    "Threshold: 1.00"
                )

            cv2.putText(
                frame,
                threshold_text,
                (30, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2
            )

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
        # 身体姿勢表示
        # ====================================================
        draw_body_position(
            frame,
            body_position
        )

        # ====================================================
        # MediaPipe骨格描画
        # ====================================================
        frame = pose_estimator.draw_landmarks(
            frame
        )

        # ====================================================
        # 出力動画保存
        #
        # 入力と同じ解像度
        # ====================================================
        writer.write(
            frame
        )

        # ====================================================
        # 表示用フレーム
        #
        # 入力と同じアスペクト比
        # ====================================================
        display_frame = resize_for_display(
            frame,
            display_width,
            display_height
        )

        # ====================================================
        # 画面表示
        # ====================================================
        cv2.imshow(
            WINDOW_NAME,
            display_frame
        )

        # ====================================================
        # キー入力
        # ====================================================
        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):
            break

        frame_index += 1

    # ========================================================
    # 終了処理
    # ========================================================
    cap.release()

    writer.release()

    cv2.destroyAllWindows()

    # ========================================================
    # 結果
    # ========================================================
    print()
    print("========================================")
    print("動画処理終了")
    print("========================================")

    print(
        "Input:",
        VIDEO_PATH
    )

    print(
        "Output:",
        OUTPUT_VIDEO_PATH
    )

    print(
        "Output resolution:",
        width,
        "x",
        height
    )


# ============================================================
# main
# ============================================================
if __name__ == "__main__":

    process_video()