from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import os
import shutil
from datetime import datetime
import uvicorn

from app.models.eye_disease_model import EyeDiseaseClassifier
from app.utils.image_utils import validate_image, enhance_image

# FastAPI app yaratish
app = FastAPI(
    title="Ko'z Kasalliklarini Aniqlash API",
    description="Sun'iy intellekt yordamida ko'z kasalliklarini aniqlash tizimi",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files va templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Upload papkasi
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ML modelni yuklash
MODEL_PATH = "models/eye_disease_model.pth"
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = EyeDiseaseClassifier(model_path=MODEL_PATH)
        print("✓ Model muvaffaqiyatli yuklandi")
    else:
        model = EyeDiseaseClassifier()
        print("⚠ Ogohlantirilgan model topilmadi. Bo'sh model yaratildi.")
        print("  Model o'rgatish uchun: python train.py")
except Exception as e:
    print(f"✗ Model yuklashda xatolik: {e}")
    model = EyeDiseaseClassifier()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Asosiy sahifa"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Bitta rasmni bashorat qilish
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model yuklanmagan")

    # Fayl formatini tekshirish
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(
            status_code=400,
            detail="Faqat PNG, JPG, JPEG formatdagi rasmlar qabul qilinadi"
        )

    try:
        # Faylni saqlash
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = os.path.splitext(file.filename)[1]
        save_filename = f"{timestamp}_{file.filename}"
        save_path = os.path.join(UPLOAD_DIR, save_filename)

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Rasm validatsiyasi
        if not validate_image(save_path):
            os.remove(save_path)
            raise HTTPException(status_code=400, detail="Noto'g'ri rasm formati")

        # Bashorat qilish
        prediction = model.predict(save_path)
        prediction['image_path'] = save_path
        prediction['filename'] = save_filename
        prediction['upload_time'] = timestamp

        return JSONResponse(content={
            "success": True,
            "data": prediction
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Xatolik yuz berdi: {str(e)}")


@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Ko'p rasmlarni bashorat qilish
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model yuklanmagan")

    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Bir vaqtning o'zida maksimum 10 ta rasm yuklash mumkin"
        )

    results = []
    errors = []

    for file in files:
        try:
            # Faylni saqlash
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            save_filename = f"{timestamp}_{file.filename}"
            save_path = os.path.join(UPLOAD_DIR, save_filename)

            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Bashorat qilish
            if validate_image(save_path):
                prediction = model.predict(save_path)
                prediction['filename'] = file.filename
                results.append(prediction)
            else:
                errors.append({
                    "filename": file.filename,
                    "error": "Noto'g'ri rasm formati"
                })
                os.remove(save_path)

        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })

    return JSONResponse(content={
        "success": True,
        "total_files": len(files),
        "successful_predictions": len(results),
        "failed_predictions": len(errors),
        "results": results,
        "errors": errors
    })


@app.get("/model-info")
async def get_model_info():
    """Model haqida ma'lumot"""
    if not model:
        raise HTTPException(status_code=503, detail="Model yuklanmagan")

    return {
        "model_type": "EfficientNet-B0 (Transfer Learning)",
        "num_classes": model.num_classes,
        "class_names": model.class_names,
        "device": str(model.device),
        "input_size": "224x224",
        "supported_formats": ["PNG", "JPG", "JPEG"]
    }


@app.delete("/clear-uploads")
async def clear_uploads():
    """Yuklangan rasmlarni o'chirish"""
    try:
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        return {
            "success": True,
            "message": "Barcha yuklangan rasmlar o'chirildi"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
