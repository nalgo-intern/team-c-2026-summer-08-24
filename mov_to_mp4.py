from pathlib import Path
import subprocess
import imageio_ffmpeg

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("input_videos")

OUTPUT_DIR.mkdir(exist_ok=True)

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

mov_files = list(INPUT_DIR.glob("*.MOV")) + list(INPUT_DIR.glob("*.mov"))

if not mov_files:
    print("MOVファイルが見つかりません")
    exit()

for mov_file in mov_files:

    output_file = OUTPUT_DIR / f"{mov_file.stem}.mp4"

    print(f"変換中: {mov_file.name}")

    command = [
        ffmpeg,
        "-i", str(mov_file),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "aac",
        "-y",
        str(output_file)
    ]

    subprocess.run(command, check=True)

    print(f"完了: {output_file}")

print("すべての変換が完了しました。")