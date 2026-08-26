# train_mlp.py
#
# LSTMなし
# ピークフレームの7個の角度だけを使った3クラス分類
#
# 0 = push_up
# 1 = squat
# 2 = other

import os
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


# ============================================================
# 設定
# ============================================================

DATA_DIR = r"C:\Users\池田一颯\internship\data"

X_PATH = os.path.join(
    DATA_DIR,
    "X.npy"
)

Y_PATH = os.path.join(
    DATA_DIR,
    "y.npy"
)

METADATA_PATH = os.path.join(
    DATA_DIR,
    "metadata.csv"
)


# ============================================================
# モデル保存先
# ============================================================

MODEL_DIR = "models"

MODEL_NAME = "mlp_peak_angle.pth"

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

BATCH_SIZE = 32

EPOCHS = 100

LEARNING_RATE = 1e-3

VALIDATION_RATIO = 0.2

NUM_WORKERS = 0


# ============================================================
# MLP設定
# ============================================================

INPUT_SIZE = 7

HIDDEN_SIZE = 64

NUM_CLASSES = 3

DROPOUT = 0.3


CLASS_NAMES = [
    "push_up",
    "squat",
    "other"
]


# ============================================================
# 再現性
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# デバイス
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# Dataset
# ============================================================

class ExerciseDataset(Dataset):

    def __init__(
        self,
        X,
        y
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(
        self,
        index
    ):

        return (
            self.X[index],
            self.y[index]
        )


# ============================================================
# MLPモデル
# ============================================================

class MLPClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes,
        dropout
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                hidden_size
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_size,
                hidden_size
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_size,
                num_classes
            )
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# データ読み込み
# ============================================================

print("\n========================================")
print("データ読み込み")
print("========================================")


X = np.load(
    X_PATH
)

y = np.load(
    Y_PATH
)

metadata = pd.read_csv(
    METADATA_PATH
)


print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)

print(
    "metadata shape:",
    metadata.shape
)


# ============================================================
# データ確認
# ============================================================

if X.ndim != 3:

    raise ValueError(
        f"Xは3次元配列である必要があります。"
        f"現在: {X.shape}"
    )


if X.shape[1] != 20:

    raise ValueError(
        f"sequence_lengthが20ではありません。"
        f"現在: {X.shape[1]}"
    )


if X.shape[2] != INPUT_SIZE:

    raise ValueError(
        f"特徴量数が7ではありません。"
        f"現在: {X.shape[2]}"
    )


if len(X) != len(y):

    raise ValueError(
        "Xとyのサンプル数が一致していません。"
    )


if len(X) != len(metadata):

    raise ValueError(
        "Xとmetadataのサンプル数が一致していません。"
    )


# ============================================================
# クラス確認
# ============================================================

print("\nクラス分布")

print(
    metadata["class"].value_counts()
)


# ============================================================
# 1フレームだけ取り出す
# ============================================================

print("\n========================================")
print("ピークフレーム抽出")
print("========================================")


# ------------------------------------------------------------
# X:
#
# (N, 20, 7)
#
# ↓
#
# (N, 7)
#
# push_up / squat:
# target_frameに対応するピークフレーム
#
# other:
# 現在の20フレーム系列の先頭フレーム
# ------------------------------------------------------------

X_peak = np.zeros(
    (len(X), INPUT_SIZE),
    dtype=np.float32
)


for i in range(len(X)):

    label = metadata.loc[
        i,
        "class"
    ]

    if label in [
        "push_up",
        "squat"
    ]:

        # ----------------------------------------------------
        # 現在のXは
        #
        # target_frameを最後のフレームとして
        # 過去20フレームを保存している
        #
        # したがってピークフレームは[-1]
        # ----------------------------------------------------

        X_peak[i] = X[
            i,
            -1,
            :
        ]

    else:

        # ----------------------------------------------------
        # other
        #
        # 現在は20フレーム系列の先頭を使用
        # ----------------------------------------------------

        X_peak[i] = X[
            i,
            0,
            :
        ]


print(
    "X_peak shape:",
    X_peak.shape
)


# ============================================================
# NaN / Inf確認
# ============================================================

if np.isnan(X_peak).any():

    raise ValueError(
        "X_peakにNaNが含まれています。"
    )


if np.isinf(X_peak).any():

    raise ValueError(
        "X_peakにInfが含まれています。"
    )


# ============================================================
# Train / Validation 分割
# ============================================================

print("\n========================================")
print("Train / Validation 分割")
print("========================================")


groups = metadata[
    "vid_id"
].values


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=VALIDATION_RATIO,
    random_state=SEED
)


train_indices, val_indices = next(
    splitter.split(
        X_peak,
        y,
        groups=groups
    )
)


X_train = X_peak[
    train_indices
]

y_train = y[
    train_indices
]


X_val = X_peak[
    val_indices
]

y_val = y[
    val_indices
]


print(
    "Train:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)


# ============================================================
# Train / Validation クラス分布
# ============================================================

print("\nTrain class distribution")

print(
    pd.Series(y_train).value_counts().sort_index()
)


print("\nValidation class distribution")

print(
    pd.Series(y_val).value_counts().sort_index()
)


# ============================================================
# 特徴量の標準化
# ============================================================

print("\n========================================")
print("特徴量標準化")
print("========================================")


# ------------------------------------------------------------
# Trainデータだけから平均・標準偏差を計算
# ------------------------------------------------------------

mean = X_train.mean(
    axis=0
)

std = X_train.std(
    axis=0
)


# 0除算防止

std[std < 1e-6] = 1.0


X_train = (
    X_train - mean
) / std


X_val = (
    X_val - mean
) / std


print(
    "Mean:",
    mean
)

print(
    "Std:",
    std
)


# ============================================================
# Dataset
# ============================================================

train_dataset = ExerciseDataset(
    X_train,
    y_train
)


val_dataset = ExerciseDataset(
    X_val,
    y_val
)


# ============================================================
# DataLoader
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# クラス重み
# ============================================================

print("\n========================================")
print("クラス重み")
print("========================================")


class_counts = np.bincount(
    y_train,
    minlength=NUM_CLASSES
)


print(
    "Train class counts:",
    class_counts
)


class_weights = (
    len(y_train)
    / (
        NUM_CLASSES
        * class_counts
    )
)


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)


for i, name in enumerate(
    CLASS_NAMES
):

    print(
        f"{name}: "
        f"count={class_counts[i]}, "
        f"weight="
        f"{class_weights[i].item():.4f}"
    )


# ============================================================
# モデル
# ============================================================

model = MLPClassifier(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
).to(device)


print("\n========================================")
print("モデル")
print("========================================")

print(model)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# 学習
# ============================================================

best_val_accuracy = 0.0


print("\n========================================")
print("学習開始")
print("========================================")


for epoch in range(EPOCHS):

    # ========================================================
    # Train
    # ========================================================

    model.train()

    train_loss = 0.0

    train_correct = 0

    train_total = 0


    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(
            device
        )

        batch_y = batch_y.to(
            device
        )


        optimizer.zero_grad()


        outputs = model(
            batch_X
        )


        loss = criterion(
            outputs,
            batch_y
        )


        loss.backward()


        optimizer.step()


        train_loss += (
            loss.item()
            * batch_X.size(0)
        )


        predictions = outputs.argmax(
            dim=1
        )


        train_correct += (
            predictions == batch_y
        ).sum().item()


        train_total += (
            batch_y.size(0)
        )


    train_loss /= train_total


    train_accuracy = (
        train_correct
        / train_total
    )


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    val_loss = 0.0

    val_correct = 0

    val_total = 0


    all_val_predictions = []

    all_val_labels = []


    with torch.no_grad():

        for batch_X, batch_y in val_loader:

            batch_X = batch_X.to(
                device
            )

            batch_y = batch_y.to(
                device
            )


            outputs = model(
                batch_X
            )


            loss = criterion(
                outputs,
                batch_y
            )


            val_loss += (
                loss.item()
                * batch_X.size(0)
            )


            predictions = outputs.argmax(
                dim=1
            )


            val_correct += (
                predictions == batch_y
            ).sum().item()


            val_total += (
                batch_y.size(0)
            )


            all_val_predictions.extend(
                predictions.cpu().numpy()
            )


            all_val_labels.extend(
                batch_y.cpu().numpy()
            )


    val_loss /= val_total


    val_accuracy = (
        val_correct
        / val_total
    )


    # ========================================================
    # 表示
    # ========================================================

    print(
        f"Epoch "
        f"{epoch + 1:03d}/{EPOCHS} | "
        f"Train Loss: "
        f"{train_loss:.4f} | "
        f"Train Acc: "
        f"{train_accuracy:.4f} | "
        f"Val Loss: "
        f"{val_loss:.4f} | "
        f"Val Acc: "
        f"{val_accuracy:.4f}"
    )


    # ========================================================
    # ベストモデル保存
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "input_size":
                    INPUT_SIZE,

                "hidden_size":
                    HIDDEN_SIZE,

                "num_classes":
                    NUM_CLASSES,

                "dropout":
                    DROPOUT,

                "class_names":
                    CLASS_NAMES,

                # 標準化に必要
                "mean":
                    mean,

                "std":
                    std
            },
            MODEL_PATH
        )


        print(
            "  → Best model saved "
            f"(Val Acc: "
            f"{val_accuracy:.4f})"
        )


# ============================================================
# 最終評価
# ============================================================

print("\n========================================")
print("最終評価")
print("========================================")


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


model.eval()


all_predictions = []

all_labels = []


with torch.no_grad():

    for batch_X, batch_y in val_loader:

        batch_X = batch_X.to(
            device
        )


        outputs = model(
            batch_X
        )


        predictions = outputs.argmax(
            dim=1
        )


        all_predictions.extend(
            predictions.cpu().numpy()
        )


        all_labels.extend(
            batch_y.numpy()
        )


# ============================================================
# Classification Report
# ============================================================

print("\nClassification Report")


print(
    classification_report(
        all_labels,
        all_predictions,
        labels=[
            0,
            1,
            2
        ],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[
        0,
        1,
        2
    ]
)


print("\nConfusion Matrix")


print(
    "             "
    "push_up "
    "squat "
    "other"
)


for name, row in zip(
    CLASS_NAMES,
    cm
):

    print(
        f"{name:10s}",
        row
    )


# ============================================================
# 完了
# ============================================================

print("\n========================================")
print("学習完了")
print("========================================")


print(
    "Best Validation Accuracy:",
    f"{best_val_accuracy:.4f}"
)


print(
    "Model:",
    MODEL_PATH
)