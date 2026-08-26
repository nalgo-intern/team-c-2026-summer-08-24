# train_4class.py
#
# Random Forestによる4クラス姿勢分類
#
# 1行 = 1ポーズ
#
# 入力:
#   7個の角度特徴量
#
# 出力:
#   squat_down
#   squat_up
#   push_up_down
#   push_up_up
#
# LSTMは使用しない

import os
import random

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.preprocessing import LabelEncoder


# ============================================================
# 設定
# ============================================================

ANGLE_CSV = (
    r"C:\Users\池田一颯\internship"
    r"\Physical Exercise Recognition Dataset"
    r"\angles.csv"
)

POSE_CSV = (
    r"C:\Users\池田一颯\internship"
    r"\Physical Exercise Recognition Dataset"
    r"\labels.csv"
)

# モデル保存先
MODEL_DIR = r"models"

MODEL_NAME = "random_forest_4class.pkl"

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

N_ESTIMATORS = 300

MAX_DEPTH = None

MIN_SAMPLES_SPLIT = 2

MIN_SAMPLES_LEAF = 1

MAX_FEATURES = "sqrt"

N_JOBS = -1


# ============================================================
# 使用する7個の特徴量
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
# 元データの4クラス
# ============================================================

SOURCE_CLASSES = [

    "squats_down",

    "squats_up",

    "pushups_down",

    "pushups_up"

]


# ============================================================
# 学習に使用する4クラス
# ============================================================

CLASS_NAMES = [

    "squat_down",

    "squat_up",

    "push_up_down",

    "push_up_up"

]


# ============================================================
# ラベル変換
# ============================================================

LABEL_MAPPING = {

    "squats_down":
        "squat_down",

    "squats_up":
        "squat_up",

    "pushups_down":
        "push_up_down",

    "pushups_up":
        "push_up_up"

}


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
    "labels.csv:",
    poses_df.shape
)


# ============================================================
# 列確認
# ============================================================

print()
print("========================================")
print("列確認")
print("========================================")

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
            f"labels.csv に "
            f"'{column}' がありません。"
        )


print("必要な列はすべて存在します。")


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

print()
print("========================================")
print("ラベル結合")
print("========================================")

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
# 4クラスだけ抽出
# ============================================================

print()
print("========================================")
print("4クラス抽出")
print("========================================")

df = df[
    df["pose"].isin(
        SOURCE_CLASSES
    )
].copy()


if len(df) == 0:

    raise ValueError(
        "4クラスのデータが見つかりません。"
        "labels.csv の pose 列を確認してください。"
    )


# ============================================================
# ラベル変換
# ============================================================

df["class"] = (

    df["pose"]
    .map(LABEL_MAPPING)

)


# ============================================================
# 使用クラス確認
# ============================================================

print()

print(
    df["class"]
    .value_counts()
)


# ============================================================
# 4クラス存在確認
# ============================================================

missing_classes = [

    class_name

    for class_name in CLASS_NAMES

    if class_name not in
    df["class"].unique()

]


if missing_classes:

    raise ValueError(

        "以下のクラスがありません:\n"

        + "\n".join(
            missing_classes
        )

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
# 欠損値削除
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
print("学習データ")
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

y_encoded = (

    label_encoder
    .fit_transform(y)

)


print()
print("========================================")
print("クラス番号")
print("========================================")

for index, class_name in enumerate(

    label_encoder.classes_

):

    print(
        f"{index}: {class_name}"
    )


# ============================================================
# クラス順確認
# ============================================================

if set(
    label_encoder.classes_
) != set(
    CLASS_NAMES
):

    raise ValueError(

        "想定している4クラスと"
        "実際のクラスが一致していません。\n"

        f"想定: {CLASS_NAMES}\n"

        f"実際: "
        f"{list(label_encoder.classes_)}"

    )


# ============================================================
# Train / Validation分割
# ============================================================

print()
print("========================================")
print("Train / Validation 分割")
print("========================================")


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

train_predictions = (

    model.predict(
        X_train
    )

)


train_accuracy = (

    accuracy_score(

        y_train,

        train_predictions

    )

)


# ============================================================
# Validation評価
# ============================================================

val_predictions = (

    model.predict(
        X_val
    )

)


val_accuracy = (

    accuracy_score(

        y_val,

        val_predictions

    )

)


# ============================================================
# Accuracy
# ============================================================

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

            len(
                label_encoder.classes_
            )

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


cm_df = pd.DataFrame(

    cm,

    index=label_encoder.classes_,

    columns=label_encoder.classes_

)


print(
    cm_df
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
            label_encoder,

        "confidence_threshold":
            0.70

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

print()
print("4クラス:")

for class_name in CLASS_NAMES:

    print(
        f"  - {class_name}"
    )