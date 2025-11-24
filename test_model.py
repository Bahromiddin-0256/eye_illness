"""
Modelni test qilish va demo scripti
"""

import sys
import os

# Model mavjudligini tekshirish
MODEL_PATH = "models/eye_disease_model.pth"

if not os.path.exists(MODEL_PATH):
    print("❌ Model topilmadi!")
    print(f"Model fayli: {MODEL_PATH}")
    print("\nModelni o'rgatish uchun:")
    print("  python train.py")
    sys.exit(1)

from app.models.eye_disease_model import EyeDiseaseClassifier
import argparse


def test_single_image(image_path: str):
    """Bitta rasmni test qilish"""
    print("\n" + "=" * 60)
    print("KO'Z KASALLIGINI ANIQLASH")
    print("=" * 60)

    # Rasm mavjudligini tekshirish
    if not os.path.exists(image_path):
        print(f"❌ Rasm topilmadi: {image_path}")
        return

    print(f"\nRasm: {image_path}")
    print("Model yuklanmoqda...")

    # Modelni yuklash
    model = EyeDiseaseClassifier(model_path=MODEL_PATH)

    print("Bashorat qilinmoqda...\n")

    # Bashorat
    result = model.predict(image_path)

    # Natijalarni ko'rsatish
    print("-" * 60)
    print(f"Aniqlangan kasallik: {result['predicted_class']}")
    print(f"Ishonch darajasi: {result['confidence'] * 100:.2f}%")
    print("-" * 60)

    print("\nBarcha ehtimolliklar:")
    for class_name, probability in result['all_probabilities'].items():
        bar_length = int(probability * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"  {class_name:25s} [{bar}] {probability * 100:.2f}%")

    print("\n" + "=" * 60)

    # Tavsiyalar
    if result['confidence'] < 0.5:
        print("⚠️  Ogohlantirish: Ishonch darajasi past. Rasmni tekshiring.")
    elif result['predicted_class'] != 'Normal':
        print("⚠️  Shifokorga murojaat qiling!")
    else:
        print("✓  Ko'z holati normal ko'rinadi.")

    print("=" * 60 + "\n")


def test_batch_images(image_dir: str):
    """Ko'p rasmlarni test qilish"""
    print("\n" + "=" * 60)
    print("BATCH TEST")
    print("=" * 60)

    # Rasmlarni topish
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []

    for file in os.listdir(image_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(image_dir, file))

    if not image_files:
        print(f"❌ {image_dir} papkasida rasmlar topilmadi")
        return

    print(f"\nTopilgan rasmlar: {len(image_files)}")
    print("Model yuklanmoqda...")

    # Modelni yuklash
    model = EyeDiseaseClassifier(model_path=MODEL_PATH)

    print("Bashorat qilinmoqda...\n")

    # Batch bashorat
    results = model.predict_batch(image_files)

    # Natijalarni ko'rsatish
    print("-" * 60)
    for i, result in enumerate(results, 1):
        filename = os.path.basename(result['image_path'])
        print(f"\n{i}. {filename}")
        print(f"   Kasallik: {result['predicted_class']}")
        print(f"   Ishonch: {result['confidence'] * 100:.2f}%")

    print("\n" + "=" * 60)

    # Statistika
    disease_counts = {}
    for result in results:
        disease = result['predicted_class']
        disease_counts[disease] = disease_counts.get(disease, 0) + 1

    print("\nStatistika:")
    for disease, count in disease_counts.items():
        percentage = (count / len(results)) * 100
        print(f"  {disease}: {count} ({percentage:.1f}%)")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modelni test qilish")

    parser.add_argument('--image', type=str, help='Bitta rasmni test qilish')
    parser.add_argument('--batch', type=str, help='Papkadagi barcha rasmlarni test qilish')

    args = parser.parse_args()

    if args.image:
        test_single_image(args.image)
    elif args.batch:
        test_batch_images(args.batch)
    else:
        print("Foydalanish:")
        print("  Bitta rasm:  python test_model.py --image path/to/image.jpg")
        print("  Batch test:  python test_model.py --batch path/to/images/")
        print("\nMisol:")
        print("  python test_model.py --image uploads/eye_sample.jpg")
        print("  python test_model.py --batch uploads/")
