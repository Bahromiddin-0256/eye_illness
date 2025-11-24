import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from typing import Tuple, Optional


class EyeDiseaseDataset(Dataset):
    """
    Ko'z kasalliklari dataseti uchun custom Dataset class
    """

    def __init__(self, data_dir: str, transform=None, mode: str = 'train'):
        """
        Args:
            data_dir: Dataset papkasi yo'li
            transform: Image transformations
            mode: 'train', 'val', yoki 'test'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode

        # Class names va labels
        self.classes = ['cataract', 'glaucoma', 'diabetic_retinopathy', 'normal']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Rasmlar ro'yxatini to'plash
        self.images = []
        self.labels = []

        self._load_dataset()

    def _load_dataset(self):
        """Datasetni yuklash"""
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, self.mode, class_name)

            if not os.path.exists(class_dir):
                print(f"Ogohlantirish: {class_dir} topilmadi")
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

        print(f"{self.mode} dataseti: {len(self.images)} ta rasm yuklandi")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Rasmni yuklash
        image = Image.open(img_path).convert('RGB')

        # Transform qo'llash
        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Training va validation uchun transformations
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Train, validation va test data loaderlarni yaratish
    """
    train_transform, val_transform = get_data_transforms()

    # Datasetlarni yaratish
    train_dataset = EyeDiseaseDataset(
        data_dir=data_dir,
        transform=train_transform,
        mode='train'
    )

    val_dataset = EyeDiseaseDataset(
        data_dir=data_dir,
        transform=val_transform,
        mode='val'
    )

    test_dataset = EyeDiseaseDataset(
        data_dir=data_dir,
        transform=val_transform,
        mode='test'
    )

    # Data loaderlarni yaratish
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def download_sample_dataset(output_dir: str = 'data/raw'):
    """
    Namuna dataset yuklab olish (Kaggle yoki GitHub dan)
    """
    print("Dataset yuklab olish boshlandi...")
    print("Quyidagi manbalardan dataset yuklab olishingiz mumkin:")
    print("1. Kaggle: https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification")
    print("2. Kaggle: https://www.kaggle.com/datasets/jr2ngb/cataractdataset")
    print("3. GitHub: Ochiq manbali ko'z kasalliklari datasetlari")
    print("\nDataset strukturasi:")
    print("data/raw/")
    print("├── train/")
    print("│   ├── cataract/")
    print("│   ├── glaucoma/")
    print("│   ├── diabetic_retinopathy/")
    print("│   └── normal/")
    print("├── val/")
    print("└── test/")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

    for split in ['train', 'val', 'test']:
        for class_name in ['cataract', 'glaucoma', 'diabetic_retinopathy', 'normal']:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

    print(f"\n✓ Dataset papkalari yaratildi: {output_dir}")
    print("Iltimos, rasmlarni tegishli papkalarga joylashtiring.")
