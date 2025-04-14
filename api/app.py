from fastapi import FastAPI, UploadFile, File, HTTPException
from model import FaceDetector
from typing import Dict, Any
import numpy as np
from PIL import Image
import io

app = FastAPI(
    title="Fake Face Detector API",
    description="API для определения фейковых лиц с помощью YOLO модели"
)

MODEL_PATH = "./weights/weights_v1.pt"
detector = FaceDetector(MODEL_PATH)


@app.get("/")
def health_check():
    return {"status": "OK"}


@app.post("/detect/")
async def detect_fake_faces(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Эндпоинт для анализа изображения на наличие фейковых лиц

    :arg:
    - file: изображение в формате JPEG/PNG

    :returns:
    - original_image: исходное изображение в байтах
    - detections: список обнаруженных лиц с метками и уверенностью
    """
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        detections = detector.img_predict(image_np)

        return {
            "detections": detections,
            "message": "Анализ завершен успешно"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")
