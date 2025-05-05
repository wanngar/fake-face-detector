from fastapi import UploadFile, File, APIRouter
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import io
import numpy as np
import cv2
import base64
from app.schemas.responses import ModelErrorResponse, ModelSuccessResponse
from app.services.detector import FaceDetector
from app.services.classificator import FaceClassificator
from app.utils.error_handler import make_error_response

router = APIRouter(tags=['Model'])
MODEL_PATH = "app/weights/weights_v1.pt"
classificator = FaceClassificator(MODEL_PATH)
detector = FaceDetector()


@router.post("/predict/image", response_model=ModelSuccessResponse)
async def predict(file: UploadFile = File(...)) -> ModelSuccessResponse | JSONResponse:
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
            image_rgb = cv2.cvtColor(detected_face_np, cv2.COLOR_BGR2RGB)
            _, img_encoded = cv2.imencode('.jpg', image_rgb)
            img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
            return ModelSuccessResponse(result=predictions, face_image=img_base64)

        except Exception as e:
            return make_error_response(status_code=500, content=f"Image processing failed : {e}")

    except Exception:
        return make_error_response(status_code=500, content="Internal server error")
