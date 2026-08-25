# train_random_forest.py
#
# Random Forestによるポーズ分類
#
# 1行 = 1ポーズ
#
# 入力:
#   7個の角度特徴量
#
# 出力:
#   push_up
#   squat
#
# LSTMは使用しない


import os
import random

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.preprocessing import LabelEncoder
import joblib


# ============================================================
# 設定
# ============================================================

# フレームごとの角度ではなく、
# ポーズ単位の角度データ
ANGLE_CSV = (
    r"C:\Users\池田一颯\internship"
    r"\Physical Exercise Recognition Dataset"
    r"\angles.csv"
)

# ポーズのラベル
POSE_CSV = (
    r"C:\Users\池田一颯\internship"
    r"\Physical Exercise Recognition Dataset"
    r"\labels.csv"
)

# モデル保存先
MODEL_DIR = r"models"

MODEL_NAME = "random_forest_pose.pkl"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    MODEL_NAME
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 学習設定
# ============================================================

SEED = 42

VALIDATION_RATIO = 0.2

# Random Forest
N_ESTIMATORS = 300

MAX_DEPTH = None

MIN_SAMPLES_SPLIT = 2

MIN_SAMPLES_LEAF = 1

MAX_FEATURES = "sqrt"

N_JOBS = -1


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
# 対象クラス
# ============================================================

CLASS_NAMES = [
    "push_up",
    "squat"
]


# ============================================================
# 乱数固定
# ============================================================

random.seed(SEED)

np.random.seed(SEED)


# ============================================================
# データ読み込み
# ============================================================

print()
print("========================================")
print("データ読み込み")
print("========================================")

angles_df = pd.read_csv(
    ANGLE_CSV
)

poses_df = pd.read_csv(
    POSE_CSV
)

print(
    "angles.csv:",
    angles_df.shape
)

print(
    "poses.csv:",
    poses_df.shape
)


# ============================================================
# 列確認
# ============================================================

required_angle_columns = [
    "pose_id"
] + FEATURE_COLUMNS

for column in required_angle_columns:

    if column not in angles_df.columns:

        raise ValueError(
            f"angles.csv に "
            f"'{column}' がありません。"
        )


required_pose_columns = [
    "pose_id",
    "pose"
]

for column in required_pose_columns:

    if column not in poses_df.columns:

        raise ValueError(
            f"poses.csv に "
            f"'{column}' がありません。"
        )


# ============================================================
# pose_idの型を統一
# ============================================================

angles_df["pose_id"] = (
    angles_df["pose_id"]
    .astype(int)
)

poses_df["pose_id"] = (
    poses_df["pose_id"]
    .astype(int)
)


# ============================================================
# ラベル結合
# ============================================================

df = angles_df.merge(
    poses_df[
        [
            "pose_id",
            "pose"
        ]
    ],
    on="pose_id",
    how="inner"
)


print()
print(
    "結合後:",
    df.shape
)


# ============================================================
# 元ラベル確認
# ============================================================

print()
print("========================================")
print("元ラベル")
print("========================================")

print(
    df["pose"]
    .value_counts()
)


# ============================================================
# squat / push_upだけ抽出
# ============================================================

def convert_label(label):

    label = str(label).lower()

    if "push_up" in label:

        return "push_up"

    if "squat" in label:

        return "squat"

    return None


df["class"] = (
    df["pose"]
    .apply(convert_label)
)


# ============================================================
# 対象外を削除
# ============================================================

df = df[
    df["class"].notna()
].copy()


print()
print("========================================")
print("使用するクラス")
print("========================================")

print(
    df["class"]
    .value_counts()
)


# ============================================================
# クラス不足チェック
# ============================================================

for class_name in CLASS_NAMES:

    count = (
        df["class"] == class_name
    ).sum()

    if count == 0:

        raise ValueError(
            f"'{class_name}' のデータがありません。"
        )


# ============================================================
# 欠損値確認
# ============================================================

print()
print("========================================")
print("欠損値")
print("========================================")

print(
    df[
        FEATURE_COLUMNS
    ].isnull().sum()
)


# ============================================================
# 欠損値を持つ行を削除
# ============================================================

before_count = len(df)

df = df.dropna(
    subset=FEATURE_COLUMNS
).reset_index(
    drop=True
)

after_count = len(df)

print(
    f"欠損値削除: "
    f"{before_count - after_count} 件"
)


# ============================================================
# X / y
# ============================================================

X = df[
    FEATURE_COLUMNS
].values.astype(
    np.float32
)

y = df[
    "class"
].values


print()
print("========================================")
print("データ")
print("========================================")

print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)


# ============================================================
# ラベルを数値化
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(
    y
)


print()
print("クラス番号")

for index, class_name in enumerate(
    label_encoder.classes_
):

    print(
        f"{index}: {class_name}"
    )


# ============================================================
# Train / Validation分割
# ============================================================

print()
print("========================================")
print("Train / Validation 分割")
print("========================================")


# ------------------------------------------------------------
# 重要
#
# 今回のデータには動画IDがない可能性があるため、
# pose_idをそのままGroupには使用しない。
#
# pose_idが単なるポーズ番号なら、
# 通常のランダム分割を使用する。
# ------------------------------------------------------------

from sklearn.model_selection import train_test_split


X_train, X_val, y_train, y_val = (
    train_test_split(
        X,
        y_encoded,
        test_size=VALIDATION_RATIO,
        random_state=SEED,
        stratify=y_encoded
    )
)


print(
    "Train:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)


# ============================================================
# Random Forest
# ============================================================

print()
print("========================================")
print("Random Forest")
print("========================================")


model = RandomForestClassifier(

    n_estimators=N_ESTIMATORS,

    max_depth=MAX_DEPTH,

    min_samples_split=MIN_SAMPLES_SPLIT,

    min_samples_leaf=MIN_SAMPLES_LEAF,

    max_features=MAX_FEATURES,

    random_state=SEED,

    n_jobs=N_JOBS,

    class_weight="balanced"

)


print(model)


# ============================================================
# 学習
# ============================================================

print()
print("========================================")
print("学習開始")
print("========================================")


model.fit(
    X_train,
    y_train
)


print(
    "学習完了"
)


# ============================================================
# Train評価
# ============================================================

train_predictions = model.predict(
    X_train
)

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)


# ============================================================
# Validation評価
# ============================================================

val_predictions = model.predict(
    X_val
)

val_accuracy = accuracy_score(
    y_val,
    val_predictions
)


print()
print("========================================")
print("Accuracy")
print("========================================")

print(
    f"Train Accuracy: "
    f"{train_accuracy:.4f}"
)

print(
    f"Validation Accuracy: "
    f"{val_accuracy:.4f}"
)


# ============================================================
# Classification Report
# ============================================================

print()
print("========================================")
print("Classification Report")
print("========================================")

print(
    classification_report(

        y_val,

        val_predictions,

        labels=np.arange(
            len(label_encoder.classes_)
        ),

        target_names=(
            label_encoder.classes_
        ),

        zero_division=0

    )
)


# ============================================================
# Confusion Matrix
# ============================================================

print()
print("========================================")
print("Confusion Matrix")
print("========================================")


cm = confusion_matrix(
    y_val,
    val_predictions
)


print(
    pd.DataFrame(
        cm,

        index=label_encoder.classes_,

        columns=label_encoder.classes_
    )
)


# ============================================================
# 特徴量重要度
# ============================================================

print()
print("========================================")
print("Feature Importance")
print("========================================")


feature_importance = pd.DataFrame({

    "feature":
        FEATURE_COLUMNS,

    "importance":
        model.feature_importances_

})


feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
)


print(
    feature_importance
    .to_string(
        index=False
    )
)


# ============================================================
# モデル保存
# ============================================================

print()
print("========================================")
print("モデル保存")
print("========================================")


joblib.dump(

    {
        "model":
            model,

        "feature_columns":
            FEATURE_COLUMNS,

        "class_names":
            list(
                label_encoder.classes_
            ),

        "label_encoder":
            label_encoder

    },

    MODEL_PATH

)


print(
    "Model:",
    MODEL_PATH
)


# ============================================================
# 完了
# ============================================================

print()
print("========================================")
print("学習完了")
print("========================================")

print(
    f"Train Accuracy: "
    f"{train_accuracy:.4f}"
)

print(
    f"Validation Accuracy: "
    f"{val_accuracy:.4f}"
)