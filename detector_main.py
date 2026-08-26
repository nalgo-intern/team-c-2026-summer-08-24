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
    pushup_threshold=0.90,

    # スクワット
    squat_threshold=0.80,

    required_consecutive=3,

    sequence_length=20,
)


results = detector.process_video(
    video_path=(
        BASE_DIR
        / "input_video"
        / "IMG_6477.mp4"
    ),

    output_path=(
        BASE_DIR
        / "output"
        / "result.mp4"
    )
)


print(results)