import os
import numpy as np
import pandas as pd


# ============================================================
# 設定
# ============================================================

# フレームごとの角度データ
FRAME_CSV = r"C:\Users\池田一颯\internship\Physical Exercise Recognition Time Series Dataset\angles.csv"

# 動画ごとのラベル
LABEL_CSV = r"C:\Users\池田一颯\internship\Physical Exercise Recognition Time Series Dataset\labels.csv"

# 保存先
SAVE_DIR = r"C:\Users\池田一颯\internship\data"

# 1系列あたりのフレーム数
SEQUENCE_LENGTH = 20

# push_up / squat
# 1動画から取得する最大サンプル数
NUM_SAMPLES_PER_VIDEO = 5

# push_up / squat の候補フレーム同士の最低間隔
MIN_FRAME_GAP = 10

# 最小角度から何%以内を候補とするか
NEAR_MIN_RATIO = 0.10

# 乱数シード
RANDOM_SEED = 42


# ============================================================
# クラス設定
# ============================================================

TARGET_CLASSES = [
    "push_up",
    "squat"
]

OTHER_CLASSES = [
    "jumping_jack",
    "pull_up",
    "situp"
]

CLASS_MAP = {
    "push_up": 0,
    "squat": 1,
    "other": 2
}


# ============================================================
# 保存先作成
# ============================================================

os.makedirs(SAVE_DIR, exist_ok=True)

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# CSV読み込み
# ============================================================

df = pd.read_csv(FRAME_CSV)
labels_df = pd.read_csv(LABEL_CSV)

print("Angles data:", df.shape)
print("Labels data:", labels_df.shape)


# ============================================================
# 特徴量
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
# 必要な列を確認
# ============================================================

required_angle_columns = [
    "vid_id",
    "frame_order"
] + FEATURE_COLUMNS

missing_angle = [
    col
    for col in required_angle_columns
    if col not in df.columns
]

if missing_angle:
    raise ValueError(
        "angles.csv に以下の列がありません:\n"
        + "\n".join(missing_angle)
    )


required_label_columns = [
    "vid_id",
    "class"
]

missing_label = [
    col
    for col in required_label_columns
    if col not in labels_df.columns
]

if missing_label:
    raise ValueError(
        "labels.csv に以下の列がありません:\n"
        + "\n".join(missing_label)
    )


# ============================================================
# vid_idの型を統一
# ============================================================

df["vid_id"] = df["vid_id"].astype(int)
labels_df["vid_id"] = labels_df["vid_id"].astype(int)


# ============================================================
# labels.csvを結合
# ============================================================

df = df.merge(
    labels_df[["vid_id", "class"]],
    on="vid_id",
    how="left"
)


# ============================================================
# ラベルがない動画を確認
# ============================================================

missing_class = df[
    df["class"].isna()
]["vid_id"].unique()

if len(missing_class) > 0:
    raise ValueError(
        "labels.csv に対応するラベルがない vid_idがあります:\n"
        + str(missing_class)
    )


# ============================================================
# 各クラスの動画数を表示
# ============================================================

print("\n元データの動画数")
print(
    df.groupby("class")["vid_id"]
    .nunique()
)


# ============================================================
# 曲がり具合
# ============================================================

def calculate_bending_score(video_df, exercise):

    """
    小さいほど曲がっていると判断する。

    push_up:
        左右の肘角度の平均

    squat:
        左右の膝角度の平均
    """

    if exercise == "push_up":

        right = video_df[
            "right_wrist_right_elbow_right_shoulder"
        ]

        left = video_df[
            "left_wrist_left_elbow_left_shoulder"
        ]

    elif exercise == "squat":

        right = video_df[
            "right_hip_right_knee_right_ankle"
        ]

        left = video_df[
            "left_hip_left_knee_left_ankle"
        ]

    else:

        raise ValueError(
            f"Unknown exercise: {exercise}"
        )

    return (right + left) / 2.0


# ============================================================
# push_up / squat
# 最小角度付近のフレームを選択
# ============================================================

def select_target_frames(video_df, exercise):

    video_df = video_df.sort_values(
        "frame_order"
    ).reset_index(drop=True)

    score = calculate_bending_score(
        video_df,
        exercise
    )

    min_score = score.min()

    threshold = min_score * (
        1.0 + NEAR_MIN_RATIO
    )

    candidate_indices = np.where(
        score.values <= threshold
    )[0]

    if len(candidate_indices) == 0:

        candidate_indices = [
            int(score.values.argmin())
        ]

    # 角度が小さい順
    candidate_indices = sorted(
        candidate_indices,
        key=lambda i: score.iloc[i]
    )

    selected = []

    for idx in candidate_indices:

        if all(
            abs(idx - selected_idx)
            >= MIN_FRAME_GAP
            for selected_idx in selected
        ):

            selected.append(idx)

        if len(selected) >= NUM_SAMPLES_PER_VIDEO:
            break

    return sorted(selected), score


# ============================================================
# other用
# ランダムに20フレームを選択
# ============================================================

def select_random_sequences(
    video_df,
    num_samples
):

    """
    otherクラス用。

    動画内から20フレーム連続した
    区間をランダムに選択する。
    """

    video_df = video_df.sort_values(
        "frame_order"
    ).reset_index(drop=True)

    video_length = len(video_df)

    # 20フレーム未満の動画は使用できない
    if video_length < SEQUENCE_LENGTH:
        return []

    # 20フレーム系列の開始可能位置
    possible_starts = np.arange(
        0,
        video_length - SEQUENCE_LENGTH + 1
    )

    # 必要数より候補が少ない場合
    sample_count = min(
        num_samples,
        len(possible_starts)
    )

    # ランダムに開始位置を選択
    selected_starts = rng.choice(
        possible_starts,
        size=sample_count,
        replace=False
    )

    return sorted(
        selected_starts.tolist()
    )


# ============================================================
# まず push_up / squat の系列を作成
# ============================================================

X = []
y = []
metadata = []

class_sequence_count = {
    "push_up": 0,
    "squat": 0,
    "other": 0
}


print("\n========================================")
print("push_up / squat の系列作成")
print("========================================")


for vid_id, video_df in df[
    df["class"].isin(TARGET_CLASSES)
].groupby("vid_id"):

    video_df = video_df.sort_values(
        "frame_order"
    ).reset_index(drop=True)

    exercise = video_df[
        "class"
    ].iloc[0]

    target_indices, score = select_target_frames(
        video_df,
        exercise
    )

    print(
        f"vid_id={vid_id}, "
        f"class={exercise}, "
        f"target="
        f"{video_df.loc[target_indices, 'frame_order'].tolist()}"
    )

    for target_idx in target_indices:

        start_idx = (
            target_idx
            - SEQUENCE_LENGTH
            + 1
        )

        end_idx = target_idx + 1

        if start_idx < 0:
            continue

        sequence = video_df.loc[
            start_idx:end_idx - 1,
            FEATURE_COLUMNS
        ].values.astype(
            np.float32
        )

        if sequence.shape != (
            SEQUENCE_LENGTH,
            len(FEATURE_COLUMNS)
        ):
            continue

        X.append(sequence)

        y.append(
            CLASS_MAP[exercise]
        )

        metadata.append({

            "vid_id":
                int(vid_id),

            "class":
                exercise,

            "source_class":
                exercise,

            "target_frame":
                int(
                    video_df.loc[
                        target_idx,
                        "frame_order"
                    ]
                ),

            "sequence_start_frame":
                int(
                    video_df.loc[
                        start_idx,
                        "frame_order"
                    ]
                ),

            "sequence_end_frame":
                int(
                    video_df.loc[
                        target_idx,
                        "frame_order"
                    ]
                ),

            "target_angle":
                float(
                    score.iloc[target_idx]
                )
        })

        class_sequence_count[
            exercise
        ] += 1


# ============================================================
# push_up / squat の最大系列数を取得
# ============================================================

target_other_count = max(
    class_sequence_count["push_up"],
    class_sequence_count["squat"]
)

print("\n========================================")
print("other系列数")
print("========================================")

print(
    "push_up:",
    class_sequence_count["push_up"]
)

print(
    "squat:",
    class_sequence_count["squat"]
)

print(
    "other目標:",
    target_other_count
)


# ============================================================
# other用動画
# jumping_jack / pull_up / situp
# ============================================================

other_video_df = df[
    df["class"].isin(OTHER_CLASSES)
].copy()


# 動画単位で取得
other_videos = list(
    other_video_df.groupby("vid_id")
)

print(
    "\nother用動画数:",
    len(other_videos)
)


# ============================================================
# other系列をランダム生成
# ============================================================

other_candidates = []


for vid_id, video_df in other_videos:

    video_df = video_df.sort_values(
        "frame_order"
    ).reset_index(drop=True)

    starts = select_random_sequences(
        video_df,
        NUM_SAMPLES_PER_VIDEO
    )

    for start_idx in starts:

        end_idx = (
            start_idx
            + SEQUENCE_LENGTH
        )

        sequence = video_df.loc[
            start_idx:end_idx - 1,
            FEATURE_COLUMNS
        ].values.astype(
            np.float32
        )

        if sequence.shape != (
            SEQUENCE_LENGTH,
            len(FEATURE_COLUMNS)
        ):
            continue

        other_candidates.append({

            "vid_id":
                int(vid_id),

            "source_class":
                video_df["class"].iloc[0],

            "start_idx":
                start_idx,

            "end_idx":
                end_idx,

            "sequence":
                sequence
        })


# ============================================================
# otherからランダムに必要数だけ選択
# ============================================================

print(
    "other候補系列数:",
    len(other_candidates)
)


if len(other_candidates) < target_other_count:

    raise ValueError(
        "\nother系列が不足しています。\n"
        f"必要: {target_other_count}\n"
        f"作成可能: {len(other_candidates)}\n"
        "\n"
        "NUM_SAMPLES_PER_VIDEOを増やすか、"
        "other用動画を追加してください。"
    )


selected_other = rng.choice(
    len(other_candidates),
    size=target_other_count,
    replace=False
)


# ============================================================
# otherをX/y/metadataへ追加
# ============================================================

for candidate_index in selected_other:

    candidate = other_candidates[
        candidate_index
    ]

    sequence = candidate["sequence"]

    X.append(sequence)

    y.append(
        CLASS_MAP["other"]
    )

    video_df = df[
        df["vid_id"]
        == candidate["vid_id"]
    ].sort_values(
        "frame_order"
    ).reset_index(drop=True)

    start_idx = candidate["start_idx"]
    end_idx = candidate["end_idx"] - 1

    metadata.append({

        "vid_id":
            candidate["vid_id"],

        "class":
            "other",

        "source_class":
            candidate["source_class"],

        "target_frame":
            -1,

        "sequence_start_frame":
            int(
                video_df.loc[
                    start_idx,
                    "frame_order"
                ]
            ),

        "sequence_end_frame":
            int(
                video_df.loc[
                    end_idx,
                    "frame_order"
                ]
            ),

        "target_angle":
            np.nan
    })

    class_sequence_count[
        "other"
    ] += 1


# ============================================================
# NumPy配列へ変換
# ============================================================

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y,
    dtype=np.int64
)

metadata_df = pd.DataFrame(
    metadata
)


# ============================================================
# シャッフル
# ============================================================

indices = rng.permutation(
    len(X)
)

X = X[indices]

y = y[indices]

metadata_df = metadata_df.iloc[
    indices
].reset_index(drop=True)


# ============================================================
# 保存
# ============================================================

np.save(
    os.path.join(
        SAVE_DIR,
        "X.npy"
    ),
    X
)

np.save(
    os.path.join(
        SAVE_DIR,
        "y.npy"
    ),
    y
)

metadata_df.to_csv(
    os.path.join(
        SAVE_DIR,
        "metadata.csv"
    ),
    index=False
)


# ============================================================
# 結果表示
# ============================================================

print("\n========================================")
print("データ作成完了")
print("========================================")

print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)

print("\nクラス数")

print(
    metadata_df["class"].value_counts()
)

print("\notherの内訳")

print(
    metadata_df[
        metadata_df["class"] == "other"
    ]["source_class"].value_counts()
)

print("\n保存先:")
print(SAVE_DIR)

print("\n保存ファイル:")
print("  X.npy")
print("  y.npy")
print("  metadata.csv")