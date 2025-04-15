from fastapi import UploadFile, File, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import io
import numpy as np
from typing import Dict
from .model import FaceDetector

router = APIRouter(tags=['Model'])

MODEL_PATH = "model/weights/weights_v1.pt"
detector = FaceDetector(MODEL_PATH)


@router.post("/predict/")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
        Анализирует изображение на наличие фейковых лиц с помощью YOLO модели

        Args:
            file (UploadFile): Загружаемое изображение в формате JPEG/PNG (макс. 10MB)

        Returns:
            Dict[str, Any]: JSON-ответ с результатами анализа:
            {
                "status": "success"|"error",
                "result": {
                    "class": "real"|"fake",  # Класс изображения
                    "prob": "XX.XX%"          # Уверенность модели в процентах
                },
                "error": str                  # Описание ошибки (при status="error")
            }

        Raises:
            HTTPException: 400 если файл не изображение/большой размер
            HTTPException: 500 при внутренних ошибках обработки

        """
    try:
        # Чтение и проверка размера файла
        image_bytes = await file.read()
        if len(image_bytes) > 10_000_000:  # 10MB лимит
            raise HTTPException(
                status_code=400,
                detail="Размер файла превышает 10MB"
            )

        # Проверка сигнатуры файла
        if not (image_bytes.startswith(b'\xff\xd8') or  # JPEG
                image_bytes.startswith(b'\x89PNG')):  # PNG
            raise HTTPException(
                status_code=400,
                detail="Некорректный формат изображения"
            )

        try:
            # Загрузка изображения
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)

            # Проверка на 3 канала (RGB)
            if len(image_np.shape) != 3 or image_np.shape[2] != 3:
                raise HTTPException(
                    status_code=400,
                    detail="Изображение должно быть в цветном формате (RGB)"
                )
        except UnidentifiedImageError:
            raise HTTPException(
                status_code=400,
                detail="Невозможно прочитать изображение"
            )

        # Обработка изображения моделью
        detections = detector.img_predict(image_np)
        return JSONResponse(
            content={
                "status": "success",
                "result": detections,
            }
        )

    except HTTPException:
        raise  # Пробрасываем уже обработанные ошибки
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Внутренняя ошибка сервера при обработке изображения"
        )
