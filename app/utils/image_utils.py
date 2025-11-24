import cv2
import numpy as np
from PIL import Image
import os
from typing import Tuple


def enhance_image(image_path: str) -> np.ndarray:
    """
    Ko'z rasmini yaxshilash (contrast, brightness)
    """
    img = cv2.imread(image_path)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return enhanced


def preprocess_eye_image(image_path: str, output_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """
    Ko'z rasmini preprocessing qilish
    """
    # Rasmni yuklash
    img = Image.open(image_path).convert('RGB')

    # O'lchamini o'zgartirish
    img = img.resize(output_size, Image.Resampling.LANCZOS)

    return img


def validate_image(image_path: str) -> bool:
    """
    Rasm formatini tekshirish
    """
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    ext = os.path.splitext(image_path)[1].lower()

    if ext not in valid_extensions:
        return False

    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except:
        return False


def detect_eye_region(image_path: str) -> np.ndarray:
    """
    Ko'z regionini aniqlash (optional - advanced feature)
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Yumshoq blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Circular detection (ko'z doira shaklida)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=30,
        maxRadius=200
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]

            # Ko'z regionini kesish
            mask = np.zeros_like(gray)
            cv2.circle(mask, center, radius, 255, -1)
            result = cv2.bitwise_and(img, img, mask=mask)

            return result

    return img


def augment_image(image: np.ndarray) -> list:
    """
    Data augmentation - training uchun
    """
    augmented_images = []

    # Original
    augmented_images.append(image)

    # Horizontal flip
    augmented_images.append(cv2.flip(image, 1))

    # Rotation
    rows, cols = image.shape[:2]
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        augmented_images.append(rotated)

    # Brightness adjustment
    bright = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
    dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-10)
    augmented_images.append(bright)
    augmented_images.append(dark)

    return augmented_images
