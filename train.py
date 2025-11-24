"""
Ko'z kasalliklarini aniqlash modelini o'rgatish scripti
"""

import os
import sys
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from app.models.eye_disease_model import EyeDiseaseClassifier, EyeDiseaseTrainer
from app.utils.dataset_loader import create_data_loaders


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='training_history.png'):
    """Training history ni grafikda ko'rsatish"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Validation Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(train_accs, label='Train Accuracy', marker='o')
    ax2.plot(val_accs, label='Validation Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history grafigi saqlandi: {save_path}")


def train_model(
    data_dir: str,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    save_dir: str = 'models'
):
    """
    Modelni o'rgatish
    """
    print("=" * 60)
    print("KO'Z KASALLIKLARINI ANIQLASH MODELINI O'RGATISH")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Save directory yaratish
    os.makedirs(save_dir, exist_ok=True)

    # Data loaderlarni yaratish
    print("\nData loaderlar yaratilmoqda...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=2
        )
    except Exception as e:
        print(f"\n❌ Xatolik: {e}")
        print("\nDataset papkasi strukturasini tekshiring:")
        print("data/raw/")
        print("├── train/")
        print("│   ├── cataract/")
        print("│   ├── glaucoma/")
        print("│   ├── diabetic_retinopathy/")
        print("│   └── normal/")
        print("├── val/")
        print("└── test/")
        sys.exit(1)

    # Model va trainer yaratish
    print("\nModel yaratilmoqda...")
    model = EyeDiseaseClassifier(num_classes=4)
    trainer = EyeDiseaseTrainer(model, learning_rate=learning_rate)

    print(f"\nTrain samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")

    # Training
    print("\n" + "=" * 60)
    print("TRAINING BOSHLANDI")
    print("=" * 60 + "\n")

    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        val_loss, val_acc = trainer.validate(val_loader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Scheduler
        trainer.scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Eng yaxshi modelni saqlash
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, 'eye_disease_model.pth')
            model.save_model(save_path, epoch, trainer.optimizer, val_loss)
            print(f"✓ Eng yaxshi model saqlandi! Val Acc: {val_acc:.2f}%")

    # Final test
    print("\n" + "=" * 60)
    print("TEST QILISH")
    print("=" * 60)

    test_loss, test_acc = trainer.validate(test_loader)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Training history grafigi
    plot_training_history(train_losses, val_losses, train_accs, val_accs,
                         save_path=os.path.join(save_dir, 'training_history.png'))

    # Natijalarni saqlash
    results = {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }

    print("\n" + "=" * 60)
    print("TRAINING TUGADI!")
    print("=" * 60)
    print(f"\nEng yaxshi validation accuracy: {best_val_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"\nModel saqlandi: {save_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ko'z kasalliklari modelini o'rgatish")

    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Dataset papkasi yo\'li')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Epochlar soni')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Model saqlanadigan papka')

    args = parser.parse_args()

    # Training
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir
    )
