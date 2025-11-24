import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, Tuple


class EyeDiseaseClassifier:
    """
    Ko'z kasalliklarini aniqlash uchun transfer learning asosidagi model
    Diseases: Cataract, Glaucoma, Diabetic Retinopathy, Normal
    """

    def __init__(self, num_classes: int = 4, model_path: str = None):
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ['Katarakta', 'Glaukoma', 'Diabetik Retinopatiya', 'Normal']

        # EfficientNet-B0 modelini yuklash
        self.model = models.efficientnet_b0(pretrained=True)

        # Classifier qismini o'zgartirish
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

        self.model = self.model.to(self.device)

        # Agar model mavjud bo'lsa, yuklash
        if model_path:
            self.load_model(model_path)

        # Image preprocessing transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path: str):
        """Saqlangan modelni yuklash"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model muvaffaqiyatli yuklandi: {model_path}")
        except Exception as e:
            print(f"Model yuklashda xatolik: {e}")

    def save_model(self, save_path: str, epoch: int, optimizer, loss: float):
        """Modelni saqlash"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'class_names': self.class_names
        }, save_path)
        print(f"Model saqlandi: {save_path}")

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Rasmni preprocessing qilish"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)

    def predict(self, image_path: str) -> Dict:
        """Bitta rasmni bashorat qilish"""
        self.model.eval()

        with torch.no_grad():
            image_tensor = self.preprocess_image(image_path)
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            # Barcha sinflar uchun ehtimolliklar
            all_probs = probabilities[0].cpu().numpy()

            result = {
                'predicted_class': self.class_names[predicted.item()],
                'confidence': float(confidence.item()),
                'all_probabilities': {
                    self.class_names[i]: float(all_probs[i])
                    for i in range(self.num_classes)
                }
            }

        return result

    def predict_batch(self, image_paths: list) -> list:
        """Ko'p rasmlarni bashorat qilish"""
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            result['image_path'] = image_path
            results.append(result)
        return results


class EyeDiseaseTrainer:
    """Model o'rgatish uchun class"""

    def __init__(self, model: EyeDiseaseClassifier, learning_rate: float = 0.001):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )

    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """Bir epoch uchun training"""
        self.model.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.model.device), labels.to(self.model.device)

            self.optimizer.zero_grad()
            outputs = self.model.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc

    def validate(self, val_loader) -> Tuple[float, float]:
        """Validation"""
        self.model.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.model.device), labels.to(self.model.device)
                outputs = self.model.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = 100 * correct / total

        return val_loss, val_acc
