import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from dataset import get_loaders

from models.efficientnetv2 import get_model as get_efficientnet
from models.densenet121 import get_model as get_densenet
from models.resnet101 import get_model as get_resnet
from models.convnext import get_model as get_convnext
from models.swin import get_model as get_swin


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ============================================================
# ARGUMENTS
# ============================================================

if len(sys.argv) < 3:

    print("\nUsage:")
    print("python src/train_optimized.py <dataset> <model>")

    print("\nDatasets:")
    print("  ham10000")
    print("  isic2018")
    print("  ph2")

    print("\nModels:")
    print("  efficientnetv2")
    print("  densenet121")
    print("  resnet101")
    print("  convnext")
    print("  swin")

    sys.exit()


DATASET = sys.argv[1].lower()
MODEL_NAME = sys.argv[2].lower()


# ============================================================
# NUMBER OF CLASSES
# ============================================================

if DATASET in ["ham10000", "isic2018"]:

    NUM_CLASSES = 7

elif DATASET == "ph2":

    NUM_CLASSES = 3

else:

    raise ValueError(
        "Unknown dataset. Choose: "
        "ham10000, isic2018, ph2"
    )


print("Dataset:", DATASET)
print("Model:", MODEL_NAME)
print("Classes:", NUM_CLASSES)


# ============================================================
# HYPERPARAMETERS
# ============================================================

BATCH_SIZE = 4

EPOCHS = 10

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 3


# ============================================================
# DATA
# ============================================================

train_loader, val_loader, test_loader = get_loaders(
    dataset_name=DATASET,
    batch_size=BATCH_SIZE
)

print("\nTrain samples:", len(train_loader.dataset))
print("Validation samples:", len(val_loader.dataset))
print("Test samples:", len(test_loader.dataset))


# ============================================================
# MODEL
# ============================================================

if MODEL_NAME == "efficientnetv2":

    model = get_efficientnet(
        num_classes=NUM_CLASSES
    )

elif MODEL_NAME == "densenet121":

    model = get_densenet(
        num_classes=NUM_CLASSES
    )

elif MODEL_NAME == "resnet101":

    model = get_resnet(
        num_classes=NUM_CLASSES
    )

elif MODEL_NAME == "convnext":

    model = get_convnext(
        num_classes=NUM_CLASSES
    )

elif MODEL_NAME == "swin":

    model = get_swin(
        num_classes=NUM_CLASSES
    )

else:

    raise ValueError(
        "Unknown model: "
        + MODEL_NAME
    )


model = model.to(DEVICE)


# ============================================================
# CLASS WEIGHTS
# ============================================================

# Calculate class frequencies from training labels.
train_labels = []

for _, labels in train_loader:

    train_labels.extend(
        labels.tolist()
    )


class_counts = torch.bincount(
    torch.tensor(train_labels),
    minlength=NUM_CLASSES
).float()


print("\nTraining class counts:")
print(class_counts.tolist())


# Inverse-frequency weighting
class_weights = 1.0 / class_counts

class_weights = (
    class_weights /
    class_weights.sum()
    * NUM_CLASSES
)


class_weights = class_weights.to(DEVICE)


print("Class weights:")
print(class_weights.detach().cpu().tolist())


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

model_dir = Path("models") / "optimized"

model_dir.mkdir(
    parents=True,
    exist_ok=True
)


model_path = (
    model_dir /
    f"{MODEL_NAME}_{DATASET}_optimized.pth"
)


# ============================================================
# TRAINING HISTORY
# ============================================================

best_val_f1 = 0.0

epochs_without_improvement = 0

history = []


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()


    avg_train_loss = (
        running_loss /
        len(train_loader)
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_true = []
    val_pred = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_true.extend(
                labels.cpu().numpy()
            )

            val_pred.extend(
                predictions.cpu().numpy()
            )


    val_accuracy = (
        sum(
            p == t
            for p, t in zip(
                val_pred,
                val_true
            )
        )
        / len(val_true)
    )


    val_f1 = f1_score(
        val_true,
        val_pred,
        average="macro",
        zero_division=0
    )


    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler.step(val_f1)

    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history.append({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_accuracy": val_accuracy,
        "val_f1": val_f1,
        "learning_rate": current_lr
    })


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {avg_train_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.4f} "
        f"Val F1: {val_f1:.4f} "
        f"LR: {current_lr:.6f}"
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_name": MODEL_NAME,
                "dataset": DATASET,
                "num_classes": NUM_CLASSES,
                "best_val_f1": best_val_f1,
                "epoch": epoch + 1
            },
            model_path
        )

        print(
            "  -> Best model saved"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"  -> No improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

import pandas as pd

history_df = pd.DataFrame(history)

history_path = (
    model_dir /
    f"{MODEL_NAME}_{DATASET}_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("OPTIMIZED TRAINING COMPLETE")
print("=" * 60)

print("Dataset:", DATASET)
print("Model:", MODEL_NAME)

print(
    "Best validation F1:",
    f"{best_val_f1:.4f}"
)

print(
    "Model saved:",
    model_path
)

print(
    "Training history:",
    history_path
)