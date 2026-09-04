# import sys
# import torch
# import torch.nn as nn
# import torch.optim as optim

# from dataset import get_loaders
# from models.efficientnetv2 import get_model as efficientnet_model
# from models.densenet121 import get_model as densenet_model
# from models.resnet101 import get_model as resnet_model
# from models.convnext import get_model as convnext_model
# from models.swin import get_model as swin_model


# # Choose model from command line
# MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "efficientnetv2"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)
# print("Model:", MODEL_NAME)


# # Data
# train_loader, val_loader, test_loader = get_loaders(batch_size=16)


# #Select model
# if MODEL_NAME == "efficientnetv2":
#     model = efficientnet_model(num_classes=7)

# elif MODEL_NAME == "densenet121":
#     model = densenet_model(num_classes=7)

# elif MODEL_NAME == "resnet101":
#     model = resnet_model(num_classes=7)

# elif MODEL_NAME == "convnext":
#     model = convnext_model(num_classes=7)

# elif MODEL_NAME == "swin":
#     model = swin_model(num_classes=7)

# else:
#     raise ValueError(
#         "Unknown model. Choose: efficientnetv2, densenet121, "
#         "resnet101, convnext, swin"
#     )


# model = model.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# EPOCHS = 2


# for epoch in range(EPOCHS):

#     # Training
#     model.train()
#     train_loss = 0

#     for images, labels in train_loader:
#         images = images.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()

#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()

#     # Validation
#     model.eval()
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for images, labels in val_loader:
#             images = images.to(device)
#             labels = labels.to(device)

#             outputs = model(images)
#             predictions = outputs.argmax(1)

#             total += labels.size(0)
#             correct += (predictions == labels).sum().item()

#     accuracy = correct / total

#     print(
#         f"Epoch [{epoch+1}/{EPOCHS}] "
#         f"Loss: {train_loss/len(train_loader):.4f} "
#         f"Val Accuracy: {accuracy:.4f}"
#     )


# # Save model for HAM10000
# # torch.save(
# #     model.state_dict(),
# #     f"{MODEL_NAME}_ham_small.pth"
# # )


# # save model for ISIC2018
# torch.save(
#     model.state_dict(),
#     f"{MODEL_NAME}_isic2018_small.pth"
# )

    


# print("Model saved!")
# print("Training completed!")







import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

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
    print("python src/train.py <dataset> <model>")

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
# DATA
# ============================================================

train_loader, val_loader, test_loader = get_loaders(
    dataset_name=DATASET,
    batch_size=4
)


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
        "Unknown model. Choose: "
        "efficientnetv2, densenet121, resnet101, "
        "convnext, swin"
    )


model = model.to(DEVICE)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)


# ============================================================
# TRAINING
# ============================================================

EPOCHS = 2

for epoch in range(EPOCHS):

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


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()


    val_accuracy = correct / total

    avg_loss = (
        running_loss /
        len(train_loader)
    )


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {avg_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.4f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

model_dir = Path("models")

model_dir.mkdir(
    parents=True,
    exist_ok=True
)

model_path = (
    model_dir /
    f"{MODEL_NAME}_{DATASET}.pth"
)


torch.save(
    model.state_dict(),
    model_path
)


print("\nModel saved:", model_path)
print("Training completed!")