import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ISICSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=224):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(".jpg")
        ])

        self.mask_lookup = {
            os.path.splitext(f)[0].replace("_segmentation", ""): f
            for f in os.listdir(mask_dir)
            if f.lower().endswith(".png")
            and f.endswith("_segmentation.png")
        }

        self.images = [
            f for f in self.images
            if os.path.splitext(f)[0] in self.mask_lookup
        ]

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        image_id = os.path.splitext(image_name)[0]
        mask_name = self.mask_lookup[image_id]

        image = Image.open(
            os.path.join(self.image_dir, image_name)
        ).convert("RGB")

        mask = Image.open(
            os.path.join(self.mask_dir, mask_name)
        ).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        mask = (mask > 0.5).float()

        return image, mask


def get_segmentation_loaders(batch_size=4):
    train_image_dir = (
        "data/RAW/ISIC2018/"
        "ISIC2018_Task1-2_Training_Input"
    )

    train_mask_dir = (
        "data/RAW/ISIC2018/"
        "ISIC2018_Task1_Training_GroundTruth"
    )

    val_image_dir = (
        "data/RAW/ISIC2018/"
        "ISIC2018_Task1-2_Validation_Input"
    )

    val_mask_dir = (
        "data/RAW/ISIC2018/"
        "ISIC2018_Task1_Validation_GroundTruth"
    )

    train_dataset = ISICSegmentationDataset(
        train_image_dir,
        train_mask_dir
    )

    val_dataset = ISICSegmentationDataset(
        val_image_dir,
        val_mask_dir
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader
