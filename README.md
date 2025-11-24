# Ko'z Kasalliklarini Aniqlash Tizimi

Sun'iy intellekt (AI) va Deep Learning yordamida ko'z kasalliklarini avtomatik aniqlash tizimi.

## Xususiyatlar

- **Transfer Learning**: EfficientNet-B0 pre-trained model asosida
- **Ko'p kasallikni aniqlash**: Katarakta, Glaukoma, Diabetik Retinopatiya, Normal
- **Web Interface**: Foydalanuvchilarga qulay web interfeys
- **REST API**: Boshqa dasturlarga integratsiya uchun API
- **Batch Processing**: Ko'p rasmlarni bir vaqtda tekshirish
- **Real-time Prediction**: Tezkor natija

## Aniqlanadigan Kasalliklar

1. **Katarakta** - Ko'z linzasining xiralashishi
2. **Glaukoma** - Ko'z ichki bosimining oshishi
3. **Diabetik Retinopatiya** - Diabet tufayli ko'z shikastlanishi
4. **Normal** - Sog'lom ko'z holati

## Loyiha Strukturasi

```
eye_illness/
├── app/
│   ├── models/
│   │   ├── __init__.py
│   │   └── eye_disease_model.py    # ML model
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_utils.py          # Rasm preprocessing
│   │   └── dataset_loader.py       # Dataset yuklash
│   ├── static/
│   │   ├── style.css               # CSS stillar
│   │   └── script.js               # Frontend JS
│   ├── templates/
│   │   └── index.html              # Web interfeys
│   ├── __init__.py
│   └── main.py                     # FastAPI application
├── data/
│   ├── raw/                        # Xom dataset
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── processed/                  # Qayta ishlangan ma'lumotlar
├── models/                         # Saqlangan modellar
├── uploads/                        # Yuklangan rasmlar
├── train.py                        # Training script
├── requirements.txt                # Dependencies
└── README.md
```

## O'rnatish

### 1. Repository ni klonlash

```bash
cd /home/bahromiddin/PycharmProjects/eye_illness
```

### 2. Virtual environment yaratish

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# yoki
.venv\Scripts\activate  # Windows
```

### 3. Dependencies o'rnatish

```bash
pip install -r requirements.txt
```

## Dataset Tayyorlash

### Dataset Strukturasi

Dataset quyidagi strukturada bo'lishi kerak:

```
data/raw/
├── train/
│   ├── cataract/           # Katarakta rasmlari
│   ├── glaucoma/           # Glaukoma rasmlari
│   ├── diabetic_retinopathy/  # Diabetik retinopatiya rasmlari
│   └── normal/             # Normal ko'z rasmlari
├── val/
│   ├── cataract/
│   ├── glaucoma/
│   ├── diabetic_retinopathy/
│   └── normal/
└── test/
    ├── cataract/
    ├── glaucoma/
    ├── diabetic_retinopathy/
    └── normal/
```

### Dataset Manbalar

1. **Kaggle Datasets**:
   - [Eye Diseases Classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
   - [Cataract Dataset](https://www.kaggle.com/datasets/jr2ngb/cataractdataset)
   - [Ocular Disease Recognition](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)

2. **Kaggle API orqali yuklash**:

```bash
# Kaggle API o'rnatish
pip install kaggle

# API key sozlash (~/.kaggle/kaggle.json)
# Dataset yuklash
kaggle datasets download -d gunavenkatdoddi/eye-diseases-classification
unzip eye-diseases-classification.zip -d data/raw/
```

3. **Dataset papkalarini yaratish**:

```bash
python -c "from app.utils.dataset_loader import download_sample_dataset; download_sample_dataset()"
```

## Modelni O'rgatish

### Asosiy Training

```bash
python train.py --data_dir data/raw --epochs 30 --batch_size 32 --lr 0.001
```

### Training Parametrlari

- `--data_dir`: Dataset papkasi yo'li (default: data/raw)
- `--epochs`: Epochlar soni (default: 30)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--save_dir`: Model saqlanadigan papka (default: models)

### Training Jarayoni

Training davomida quyidagilar ko'rsatiladi:
- Train va Validation Loss
- Train va Validation Accuracy
- Eng yaxshi model avtomatik saqlanadi
- Training history grafigi yaratiladi

## API ni Ishga Tushirish

### Development Mode

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
python app/main.py
```

Brauzerda oching: `http://localhost:8000`

## API Endpoints

### 1. Asosiy Sahifa
- **URL**: `GET /`
- **Tavsif**: Web interfeys

### 2. Health Check
- **URL**: `GET /health`
- **Javob**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-11-24T10:30:00"
}
```

### 3. Bitta Rasmni Bashorat Qilish
- **URL**: `POST /predict`
- **Body**: `multipart/form-data` (file)
- **Javob**:
```json
{
  "success": true,
  "data": {
    "predicted_class": "Katarakta",
    "confidence": 0.95,
    "all_probabilities": {
      "Katarakta": 0.95,
      "Glaukoma": 0.02,
      "Diabetik Retinopatiya": 0.01,
      "Normal": 0.02
    },
    "filename": "eye_image.jpg"
  }
}
```

### 4. Ko'p Rasmlarni Bashorat Qilish
- **URL**: `POST /predict-batch`
- **Body**: `multipart/form-data` (files)
- **Limit**: Maksimum 10 ta rasm

### 5. Model Ma'lumotlari
- **URL**: `GET /model-info`
- **Javob**:
```json
{
  "model_type": "EfficientNet-B0 (Transfer Learning)",
  "num_classes": 4,
  "class_names": ["Katarakta", "Glaukoma", "Diabetik Retinopatiya", "Normal"],
  "device": "cuda",
  "input_size": "224x224"
}
```

### 6. Yuklangan Rasmlarni O'chirish
- **URL**: `DELETE /clear-uploads`

## cURL bilan Foydalanish

```bash
# Bitta rasm yuklash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/eye_image.jpg"

# Ko'p rasmlar yuklash
curl -X POST "http://localhost:8000/predict-batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

## Python bilan Foydalanish

```python
import requests

# Bitta rasm
url = "http://localhost:8000/predict"
files = {'file': open('eye_image.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())

# Ko'p rasmlar
url = "http://localhost:8000/predict-batch"
files = [
    ('files', open('image1.jpg', 'rb')),
    ('files', open('image2.jpg', 'rb')),
]
response = requests.post(url, files=files)
print(response.json())
```

## Model Arxitekturasi

- **Base Model**: EfficientNet-B0 (ImageNet pre-trained)
- **Input Size**: 224x224x3
- **Custom Classifier**:
  - Dropout(0.3)
  - Linear(1280 → 512)
  - ReLU
  - Dropout(0.2)
  - Linear(512 → 4)
- **Output**: 4 class (Softmax)

## Training Strategiyasi

1. **Transfer Learning**: Pre-trained EfficientNet-B0
2. **Data Augmentation**:
   - Random horizontal flip
   - Random rotation (±10°)
   - Color jitter
3. **Optimization**:
   - Optimizer: Adam
   - Learning rate: 0.001
   - Scheduler: ReduceLROnPlateau
4. **Loss**: CrossEntropyLoss

## Performance

Model performansini oshirish uchun:
- Ko'proq dataset qo'shing
- Epochlar sonini oshiring
- Data augmentation qo'shing
- Ensemble modellar ishlatish

## Muammolarni Hal Qilish

### Model yuklanmayapti
```bash
# Model fayli mavjudligini tekshiring
ls models/eye_disease_model.pth

# Agar yo'q bo'lsa, modelni o'rgating
python train.py
```

### CUDA xatosi
```bash
# CPU da ishlash uchun
# model.py faylidagi device sozlamalarini tekshiring
```

### Dataset topilmadi
```bash
# Dataset strukturasini tekshiring
tree data/raw/

# Papkalarni yarating
python -c "from app.utils.dataset_loader import download_sample_dataset; download_sample_dataset()"
```

## Texnologiyalar

- **Backend**: FastAPI, Uvicorn
- **ML Framework**: PyTorch, TorchVision
- **Model**: EfficientNet-B0
- **Image Processing**: OpenCV, PIL
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: NumPy, Pandas

## Litsenziya

Bu loyiha ta'lim maqsadlari uchun yaratilgan.

## Muhim Ogohlantirish

⚠️ **Diqqat**: Bu tizim faqat ko'makchi diagnostika vositasi hisoblanadi. Aniq tashxis va davolash uchun malakali shifokorga murojaat qiling. Tizim tibbiy ko'rik o'rnini bosa olmaydi!

## Muallif

Bahromiddin

## Yordam

Savollar yoki muammolar bo'lsa, GitHub Issues orqali murojaat qiling.

---

**Made with ❤️ for better healthcare**
