# ============================================================
# train.py
#
# LSTMによる2クラス運動分類
#
# 0 = push_up
# 1 = squat
#
# otherクラスは使用しない
#
# 入力:
#   X.npy
#   shape = (N, 20, 7)
#
# 出力:
#   push_up
#   squat
# ============================================================

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

MODEL_DIR = r"models"

MODEL_NAME = "lstm_binary_pushup_squat.pth"

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

LEARNING_RATE = 1e-4

VALIDATION_RATIO = 0.2

NUM_WORKERS = 0


# ============================================================
# データ拡張設定
# ============================================================

# 速度倍率
#
# 0.7 → 遅い
# 1.0 → 元の速度
# 1.3 → 速い
#
# 現在は1～1なので速度拡張なし

SPEED_MIN = 1.0
SPEED_MAX = 1.0


# ============================================================
# LSTM設定
# ============================================================

INPUT_SIZE = 7

HIDDEN_SIZE = 64

NUM_LAYERS = 2

DROPOUT = 0.3

# push_up / squat
NUM_CLASSES = 2

CLASS_NAMES = [
    "push_up",
    "squat"
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
# 速度データ拡張
# ============================================================

def speed_augmentation(sequence):

    seq_len = sequence.shape[0]

    speed = np.random.uniform(
        SPEED_MIN,
        SPEED_MAX
    )

    original_time = np.arange(
        seq_len,
        dtype=np.float32
    )

    center = (
        seq_len - 1
    ) / 2.0

    new_time = (
        center
        + (
            original_time - center
        ) * speed
    )

    new_time = np.clip(
        new_time,
        0,
        seq_len - 1
    )

    augmented = np.zeros_like(
        sequence,
        dtype=np.float32
    )

    for feature_index in range(
        sequence.shape[1]
    ):

        augmented[
            :,
            feature_index
        ] = np.interp(
            new_time,
            original_time,
            sequence[
                :,
                feature_index
            ]
        )

    return augmented


# ============================================================
# Dataset
# ============================================================

class ExerciseDataset(Dataset):

    def __init__(
        self,
        X,
        y,
        training=False
    ):

        self.X = X

        self.y = y

        self.training = training


    def __len__(self):

        return len(self.X)


    def __getitem__(self, index):

        sequence = self.X[
            index
        ].copy()

        label = self.y[
            index
        ]

        # ----------------------------------------------------
        # 学習時のみデータ拡張
        # ----------------------------------------------------

        if self.training:

            sequence = speed_augmentation(
                sequence
            )

        # ----------------------------------------------------
        # Tensor化
        # ----------------------------------------------------

        sequence = torch.tensor(
            sequence,
            dtype=torch.float32
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return (
            sequence,
            label
        )


# ============================================================
# LSTMモデル
# ============================================================

class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            )
        )

        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )


    def forward(self, x):

        output, (
            hidden,
            cell
        ) = self.lstm(x)

        last_hidden = hidden[-1]

        logits = self.fc(
            last_hidden
        )

        return logits


# ============================================================
# データ読み込み
# ============================================================

print()
print("========================================")
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
# 元のクラス分布
# ============================================================

print()
print("========================================")
print("元のクラス分布")
print("========================================")

print(
    metadata["class"].value_counts()
)


# ============================================================
# push_up / squat だけを抽出
# ============================================================

print()
print("========================================")
print("2クラス分類用データ抽出")
print("========================================")


# ------------------------------------------------------------
# metadataのclassを使用
#
# push_up
# squat
#
# だけを残す
# ------------------------------------------------------------

valid_mask = metadata[
    "class"
].isin(
    CLASS_NAMES
).values


X = X[
    valid_mask
]

y = y[
    valid_mask
]

metadata = metadata.loc[
    valid_mask
].reset_index(
    drop=True
)


print(
    "抽出後 X:",
    X.shape
)

print(
    "抽出後 y:",
    y.shape
)

print(
    "抽出後 metadata:",
    metadata.shape
)


# ============================================================
# ラベルを2値化
# ============================================================

print()
print("========================================")
print("ラベル変換")
print("========================================")


# ------------------------------------------------------------
# metadataのclassから確実に作る
#
# push_up = 0
# squat   = 1
# ------------------------------------------------------------

y = metadata[
    "class"
].map(
    {
        "push_up": 0,
        "squat": 1
    }
).values.astype(
    np.int64
)


# ------------------------------------------------------------
# 念のため確認
# ------------------------------------------------------------

if np.any(
    (y < 0) |
    (y >= NUM_CLASSES)
):

    raise ValueError(
        "ラベル変換に失敗しました。"
    )


print(
    "push_up:",
    np.sum(y == 0)
)

print(
    "squat:",
    np.sum(y == 1)
)


# ============================================================
# クラス不足チェック
# ============================================================

for class_id, class_name in enumerate(
    CLASS_NAMES
):

    count = np.sum(
        y == class_id
    )

    if count == 0:

        raise ValueError(
            f"'{class_name}' のデータがありません。"
        )


# ============================================================
# Train / Validation 分割
#
# vid_id単位
# ============================================================

print()
print("========================================")
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
        X,
        y,
        groups=groups
    )
)


X_train = X[
    train_indices
]

y_train = y[
    train_indices
]


X_val = X[
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


print()
print("Train class distribution")

print(
    pd.Series(
        y_train
    ).value_counts().sort_index()
)


print()
print("Validation class distribution")

print(
    pd.Series(
        y_val
    ).value_counts().sort_index()
)


# ============================================================
# Dataset
# ============================================================

train_dataset = ExerciseDataset(
    X_train,
    y_train,
    training=True
)


val_dataset = ExerciseDataset(
    X_val,
    y_val,
    training=False
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

print()
print("========================================")
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

model = LSTMClassifier(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
).to(device)


print()
print("========================================")
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
    lr=LEARNING_RATE
)


# ============================================================
# 学習
# ============================================================

best_val_accuracy = 0.0


print()
print("========================================")
print("学習開始")
print("========================================")

print(
    f"Speed augmentation: "
    f"{SPEED_MIN} ～ {SPEED_MAX}"
)


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
    # 結果
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

                "num_layers":
                    NUM_LAYERS,

                "num_classes":
                    NUM_CLASSES,

                "dropout":
                    DROPOUT,

                "class_names":
                    CLASS_NAMES
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

print()
print("========================================")
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


# ============================================================
# Validation予測
# ============================================================

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

print()
print("Classification Report")


print(
    classification_report(
        all_labels,
        all_predictions,
        labels=[
            0,
            1
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
        1
    ]
)


print()
print("Confusion Matrix")


print(
    "              "
    "push_up "
    "squat"
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

print()
print("========================================")
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