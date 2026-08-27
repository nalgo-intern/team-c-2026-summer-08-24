from pathlib import Path

from exercise_detector import ExerciseDetector


BASE_DIR = Path(__file__).resolve().parent

detector = ExerciseDetector(
    model_path=(
        BASE_DIR
        / "models"
        / "lstm_binary_pushup_squat.pth"
    ),

    pose_model_path=(
        BASE_DIR
        / "pose_landmarker_lite.task"
    ),

    # 腕立て
    pushup_threshold=0.8,

    # スクワット
    squat_threshold=0.2,

    required_consecutive=3,

    sequence_length=20,
)


results = detector.process_video(
    video_path=(r"C:\dev\git\test_nalgo_intern_teamc.mp4"
        
    ),

    output_path=(
        BASE_DIR
        / "output"
        / "result.mp4"
    )
)


print(results)

import cv2
from pathlib import Path

from exercise_detector import ExerciseDetector


BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# ExerciseDetector
# ============================================================

detector = ExerciseDetector(
    model_path=(
        BASE_DIR
        / "models"
        / "lstm_binary_pushup_squat.pth"
    ),

    pose_model_path=(
        BASE_DIR
        / "pose_landmarker_lite.task"
    ),

    pushup_threshold=0.90,
    squat_threshold=0.80,

    required_consecutive=3,
    sequence_length=20,
)


# ============================================================
# カメラ
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("カメラを開けません")


# ============================================================
# Detectorの状態をリセット
# ============================================================

detector.reset()


# ============================================================
# FPS
# ============================================================

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0


frame_index = 0


# ============================================================
# リアルタイム処理
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # --------------------------------------------------------
    # timestamp
    # --------------------------------------------------------

    timestamp_ms = (
        frame_index
        / fps
        * 1000.0
    )


    # --------------------------------------------------------
    # ★ 1フレーム処理
    # --------------------------------------------------------

    result = detector.process_frame(
        frame,
        timestamp_ms
    )


    # --------------------------------------------------------
    # result
    #
    # 0 = unknown
    # 1 = push_up
    # 2 = squat
    # --------------------------------------------------------

    print(
        f"frame={frame_index}, "
        f"result={result}"
    )


    # --------------------------------------------------------
    # 結果を利用
    # --------------------------------------------------------

    if result == 1:

        print("腕立て")


    elif result == 2:

        print("スクワット")


    else:

        print("unknown")


    # --------------------------------------------------------
    # 画面表示
    # --------------------------------------------------------

    if result == 1:

        text = "PUSH UP"

    elif result == 2:

        text = "SQUAT"

    else:

        text = "UNKNOWN"


    cv2.putText(
        frame,
        text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )


    cv2.imshow(
        "Realtime Exercise Detection",
        frame
    )


    # --------------------------------------------------------
    # フレーム番号
    # --------------------------------------------------------

    frame_index += 1


    # --------------------------------------------------------
    # qで終了
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# 終了
# ============================================================

cap.release()

cv2.destroyAllWindows()