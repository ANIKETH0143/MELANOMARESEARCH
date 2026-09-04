import os
import torch
import torch.optim as optim

from models.unet import UNet
from segmentation_dataset import get_segmentation_loaders
from segmentation_loss import DiceBCELoss


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-4

MODEL_DIR = "models/segmentation"
os.makedirs(MODEL_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(
    MODEL_DIR,
    "unet_isic2018.pth"
)


def dice_score(logits, targets, threshold=0.5):
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    predictions = predictions.view(
        predictions.size(0), -1
    )
    targets = targets.view(
        targets.size(0), -1
    )

    intersection = (predictions * targets).sum(dim=1)

    dice = (
        (2.0 * intersection + 1.0)
        /
        (
            predictions.sum(dim=1)
            + targets.sum(dim=1)
            + 1.0
        )
    )

    return dice.mean().item()


def main():
    print("Device:", DEVICE)

    train_loader, val_loader = get_segmentation_loaders(
        batch_size=BATCH_SIZE
    )

    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))

    model = UNet(
        in_channels=3,
        out_channels=1
    ).to(DEVICE)

    criterion = DiceBCELoss(
        dice_weight=1.0,
        bce_weight=1.0
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_dice = 0.0

    for epoch in range(1, EPOCHS + 1):

        model.train()

        train_loss = 0.0
        train_dice = 0.0

        for images, masks in train_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice_score(outputs, masks)

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        model.eval()

        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(images)

                loss = criterion(outputs, masks)

                val_loss += loss.item()
                val_dice += dice_score(outputs, masks)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f}"
        )

        if val_dice > best_val_dice:

            best_val_dice = val_dice

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice
                },
                CHECKPOINT_PATH
            )

            print(
                f"  Best model saved: "
                f"Val Dice = {val_dice:.4f}"
            )

    print()
    print("Training complete.")
    print("Best validation Dice:", f"{best_val_dice:.4f}")
    print("Checkpoint:", CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
