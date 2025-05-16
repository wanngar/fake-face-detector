import os
import tempfile
from fastapi import UploadFile, File, APIRouter
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import io
import numpy as np
import cv2
from app.schemas.responses import ModelSuccessResponse
from app.services.detector import FaceDetector
from app.services.classificator import FaceClassificator
from app.utils.error_handler import make_error_response
from app.utils.image_decoder import cv2_to_base64

router = APIRouter(tags=['Model'])
MODEL_PATH = "app/weights/weights_v1.pt"
classificator = FaceClassificator(MODEL_PATH)
detector = FaceDetector()


@router.post("/predict/image", response_model=ModelSuccessResponse)
async def predict_image(file: UploadFile = File(...)) -> ModelSuccessResponse | JSONResponse:
    try:
        image_bytes = await file.read()
        if len(image_bytes) > 10_000_000:
            return make_error_response(status_code=400, content="File size exceeded")

        if not (image_bytes.startswith(b'\xff\xd8') or  # JPEG
                image_bytes.startswith(b'\x89PNG')):  # PNG
            return make_error_response(status_code=400, content="Invalid image format")

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)

            if len(image_np.shape) != 3 or image_np.shape[2] != 3:
                return make_error_response(status_code=400, content="Image must be in RGB format")

        except UnidentifiedImageError:
            return make_error_response(status_code=400, content="Cannot read image file")

        try:
            detected_face_np = detector.detect(image_np)
        except ValueError as e:
            return make_error_response(status_code=400, content=f"{e}")

        try:
            predictions = classificator.classify_image(detected_face_np)
            img_base64 = cv2_to_base64(detected_face_np)
            return ModelSuccessResponse(result=predictions, face_image=img_base64)

        except Exception as e:
            return make_error_response(status_code=500, content=f"Image processing failed : {e}")

    except Exception:
        return make_error_response(status_code=500, content="Internal server error")


@router.post("/predict/video", response_model=ModelSuccessResponse)
async def prediction_video(video_file: UploadFile = File(...)):
    '''метод для обработки видео'''

    class_counts = {'fake': 0, 'real': 0}

    # Создаем временный файл для видео
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        # Читаем загруженный файл и записываем во временный файл
        content = await video_file.read()
        temp_video.write(content)
        temp_video_path = temp_video.name
        img_base64 = ''
    try:
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise ValueError("Не удалось открыть видеофайл")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            try:
                detected_face = detector.detect(frame)
                result = classificator.classify_image(detected_face)

            except ValueError as e:
                return make_error_response(status_code=400, content=f"{e}")

            class_name = result['class']
            class_counts[class_name if class_name == 'fake' else 'real'] += 1
            img_base64 = cv2_to_base64(detected_face)

        total_frames = class_counts['fake'] + class_counts['real']
        fake_percentage = (class_counts['fake'] / total_frames) * 100 if total_frames > 0 else 0.0
        predictions = {"class": f"{'fake' if fake_percentage >= 65 else 'real'}", "prob": f"{float(fake_percentage):.2f}%"}
        return ModelSuccessResponse(result=predictions, face_image=img_base64)

    except Exception as e:
        return make_error_response(status_code=500, content=f"Internal server error {e}")

    finally:
        # Удаляем временный файл после обработки
        cap.release()
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
