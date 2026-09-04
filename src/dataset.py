# import pandas as pd
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms


# # HAM10000 class mapping
# CLASS_MAP = {
#     "akiec": 0,
#     "bcc": 1,
#     "bkl": 2,
#     "df": 3,
#     "mel": 4,
#     "nv": 5,
#     "vasc": 6
# }


# class HAM10000Dataset(Dataset):

#     def __init__(self, csv_file, train=False):
#         self.data = pd.read_csv(csv_file)

#         if train:
#             self.transform = transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomRotation(10),
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]
#                 )
#             ])
#         else:
#             self.transform = transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]
#                 )
#             ])

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         row = self.data.iloc[index]

#         image = Image.open(row["image_path"]).convert("RGB")
#         image = self.transform(image)

#         label = CLASS_MAP[row["dx"]]

#         return image, label


# def get_loaders(batch_size=32):

#     train_dataset = HAM10000Dataset(
#         "data/splits/HAM10000/train_small.csv",
#         train=True
#     )

#     val_dataset = HAM10000Dataset(
#         "data/splits/HAM10000/val_small.csv",
#         train=False
#     )

#     test_dataset = HAM10000Dataset(
#         "data/splits/HAM10000/test_small.csv",
#         train=False
#     )

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=0
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0
#     )

#     return train_loader, val_loader, test_loader



import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ISIC2018 class mapping
CLASS_MAP = {
    "AKIEC": 0,
    "BCC": 1,
    "BKL": 2,
    "DF": 3,
    "MEL": 4,
    "NV": 5,
    "VASC": 6
}


# class ISIC2018Dataset(Dataset):

#     def __init__(self, csv_file, train=False):
#         self.data = pd.read_csv(csv_file)

#         if train:
#             self.transform = transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomRotation(10),
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]
#                 )
#             ])
#         else:
#             self.transform = transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]
#                 )
#             ])

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         row = self.data.iloc[index]

#         image = Image.open(row["image_path"]).convert("RGB")
#         image = self.transform(image)

#         label = CLASS_MAP[row["label"]]

#         return image, label


# def get_loaders(batch_size=32):

#     train_dataset = ISIC2018Dataset(
#         "data/splits/ISIC2018/train_small.csv",
#         train=True
#     )

#     val_dataset = ISIC2018Dataset(
#         "data/splits/ISIC2018/val_small.csv",
#         train=False
#     )

#     test_dataset = ISIC2018Dataset(
#         "data/splits/ISIC2018/test_small.csv",
#         train=False
#     )

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=0
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0
#     )

#     return train_loader, val_loader, test_loader





import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# CLASS MAPPINGS
# ============================================================

HAM_CLASSES = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6
}

ISIC_CLASSES = {
    "AKIEC": 0,
    "BCC": 1,
    "BKL": 2,
    "DF": 3,
    "MEL": 4,
    "NV": 5,
    "VASC": 6
}

PH2_CLASSES = {
    "common_nevus": 0,
    "atypical_nevus": 1,
    "melanoma": 2
}


# ============================================================
# DATASET
# ============================================================

class SkinDataset(Dataset):

    def __init__(self, csv_file, dataset_name, train=False):

        self.data = pd.read_csv(csv_file)
        self.dataset_name = dataset_name.lower()

        if train:

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        else:

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])


    def __len__(self):

        return len(self.data)


    def __getitem__(self, index):

        row = self.data.iloc[index]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        image = self.transform(image)

        label_name = str(row["label"]).strip()


        # ----------------------------------------------------
        # HAM10000
        # ----------------------------------------------------

        if self.dataset_name == "ham10000":

            label = HAM_CLASSES[label_name.lower()]


        # ----------------------------------------------------
        # ISIC2018
        # ----------------------------------------------------

        elif self.dataset_name == "isic2018":

            label = ISIC_CLASSES[label_name.upper()]


        # ----------------------------------------------------
        # PH2
        # ----------------------------------------------------

        elif self.dataset_name == "ph2":

            label = PH2_CLASSES[label_name.lower()]


        else:

            raise ValueError(
                f"Unknown dataset: {dataset_name}"
            )


        return image, label


# ============================================================
# GET LOADERS
# ============================================================

def get_loaders(
    dataset_name="HAM10000",
    batch_size=32
):

    dataset_name = dataset_name.lower()


    if dataset_name == "ham10000":

        base_path = "data/splits/HAM10000"

    elif dataset_name == "isic2018":

        base_path = "data/splits/ISIC2018"

    elif dataset_name == "ph2":

        base_path = "data/splits/PH2"

    else:

        raise ValueError(
            "Unknown dataset. Choose: "
            "HAM10000, ISIC2018, PH2"
        )


    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = SkinDataset(
        f"{base_path}/train.csv",
        dataset_name,
        train=True
    )

    val_dataset = SkinDataset(
        f"{base_path}/val.csv",
        dataset_name,
        train=False
    )

    test_dataset = SkinDataset(
        f"{base_path}/test.csv",
        dataset_name,
        train=False
    )


    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )


    return train_loader, val_loader, test_loader