from ultralytics import YOLO
import numpy as np
from typing import Any


class FaceClassificator:
    """Класс для  классификации лиц на изображениях с использованием YOLO."""

    def __init__(self, weights_path: str):
        self._model = YOLO(weights_path)

    def classify_image(self, frame: np.ndarray[Any, np.dtype]) -> dict[str:str]:
        try:
            result = self._model.predict(frame, verbose=False)
            class_index = result[0].probs.top1
            class_name = result[0].names[class_index]
            confidence = result[0].probs.top1conf.item()
            return {"class": f"{class_name}", "prob": f"{confidence:.2%}"}

        except Exception:
            raise RuntimeError("Model runtime error")
