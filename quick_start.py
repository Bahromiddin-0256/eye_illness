"""
Tez boshlash uchun script - Dataset strukturasini yaratish va kerakli fayllarni tekshirish
"""

import os
import sys


def check_dependencies():
    """Kerakli kutubxonalar o'rnatilganligini tekshirish"""
    print("\n" + "=" * 60)
    print("DEPENDENCIES TEKSHIRILMOQDA")
    print("=" * 60 + "\n")

    required_packages = [
        'fastapi',
        'uvicorn',
        'torch',
        'torchvision',
        'PIL',
        'cv2',
        'numpy',
        'pandas'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'cv2':
                __import__('cv2')
            else:
                __import__(package)
            print(f"✓ {package} o'rnatilgan")
        except ImportError:
            print(f"✗ {package} topilmadi")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n❌ {len(missing_packages)} ta paket topilmadi")
        print("\nO'rnatish uchun:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\n✓ Barcha kerakli paketlar o'rnatilgan")
        return True


def create_directory_structure():
    """Dataset va boshqa papkalarni yaratish"""
    print("\n" + "=" * 60)
    print("PAPKALAR STRUKTURASI YARATILMOQDA")
    print("=" * 60 + "\n")

    directories = [
        'data/raw/train/cataract',
        'data/raw/train/glaucoma',
        'data/raw/train/diabetic_retinopathy',
        'data/raw/train/normal',
        'data/raw/val/cataract',
        'data/raw/val/glaucoma',
        'data/raw/val/diabetic_retinopathy',
        'data/raw/val/normal',
        'data/raw/test/cataract',
        'data/raw/test/glaucoma',
        'data/raw/test/diabetic_retinopathy',
        'data/raw/test/normal',
        'data/processed',
        'models',
        'uploads',
        'app/static',
        'app/templates'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Yaratildi: {directory}")

    print("\n✓ Barcha papkalar yaratildi")


def check_model():
    """Model faylini tekshirish"""
    print("\n" + "=" * 60)
    print("MODEL TEKSHIRILMOQDA")
    print("=" * 60 + "\n")

    model_path = "models/eye_disease_model.pth"

    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ Model topildi: {model_path}")
        print(f"  Hajmi: {size_mb:.2f} MB")
        return True
    else:
        print(f"✗ Model topilmadi: {model_path}")
        print("\nModelni o'rgatish uchun:")
        print("  1. Dataset tayyorlang (data/raw/)")
        print("  2. python train.py --epochs 30")
        return False


def check_dataset():
    """Dataset mavjudligini tekshirish"""
    print("\n" + "=" * 60)
    print("DATASET TEKSHIRILMOQDA")
    print("=" * 60 + "\n")

    splits = ['train', 'val', 'test']
    classes = ['cataract', 'glaucoma', 'diabetic_retinopathy', 'normal']

    total_images = 0
    dataset_stats = {}

    for split in splits:
        dataset_stats[split] = {}
        for cls in classes:
            path = f"data/raw/{split}/{cls}"
            if os.path.exists(path):
                images = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                count = len(images)
                dataset_stats[split][cls] = count
                total_images += count
            else:
                dataset_stats[split][cls] = 0

    # Natijalarni ko'rsatish
    for split in splits:
        print(f"\n{split.upper()}:")
        split_total = sum(dataset_stats[split].values())
        for cls in classes:
            count = dataset_stats[split][cls]
            print(f"  {cls:25s}: {count:4d} rasmlar")
        print(f"  {'JAMI':25s}: {split_total:4d} rasmlar")

    print(f"\n{'='*60}")
    print(f"UMUMIY RASMLAR: {total_images}")
    print(f"{'='*60}")

    if total_images == 0:
        print("\n⚠️  Dataset bo'sh!")
        print("\nDataset yuklab olish uchun:")
        print("  1. Kaggle: https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification")
        print("  2. Rasmlarni data/raw/ papkasiga joylashtiring")
        return False
    else:
        print(f"\n✓ Dataset topildi: {total_images} ta rasm")
        return True


def print_next_steps(deps_ok, model_ok, dataset_ok):
    """Keyingi qadamlarni ko'rsatish"""
    print("\n" + "=" * 60)
    print("KEYINGI QADAMLAR")
    print("=" * 60 + "\n")

    if not deps_ok:
        print("1. Dependencies o'rnatish:")
        print("   pip install -r requirements.txt\n")

    if not dataset_ok:
        print("2. Dataset tayyorlash:")
        print("   - Kaggle dan dataset yuklab oling")
        print("   - Rasmlarni data/raw/ papkasiga joylashtiring")
        print("   - Har bir kasallik uchun alohida papka yarating\n")

    if not model_ok and dataset_ok:
        print("3. Modelni o'rgatish:")
        print("   python train.py --epochs 30 --batch_size 32\n")

    if model_ok and dataset_ok and deps_ok:
        print("✓ Hammasi tayyor! API ni ishga tushirish:")
        print("   python -m uvicorn app.main:app --reload\n")
        print("Brauzerda oching: http://localhost:8000\n")
        print("Yoki test qilish:")
        print("   python test_model.py --image path/to/image.jpg")

    print("=" * 60 + "\n")


def main():
    """Asosiy funksiya"""
    print("\n" + "=" * 60)
    print("KO'Z KASALLIKLARINI ANIQLASH - QUICK START")
    print("=" * 60)

    # 1. Dependencies tekshirish
    deps_ok = check_dependencies()

    # 2. Papkalar yaratish
    create_directory_structure()

    # 3. Model tekshirish
    model_ok = check_model()

    # 4. Dataset tekshirish
    dataset_ok = check_dataset()

    # 5. Keyingi qadamlar
    print_next_steps(deps_ok, model_ok, dataset_ok)


if __name__ == "__main__":
    main()
