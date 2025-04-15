import os
from typing import Any
import numpy as np
from ultralytics import YOLO


class FaceDetector:
    """Класс для детекции и классификации лиц на изображениях с использованием YOLO."""

    def __init__(self, weights_path: str):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Файл с весами '{weights_path}' не найден.")

        try:
            self._model = YOLO(weights_path)
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели: {str(e)}")

    def img_predict(self, img: np.ndarray[Any, np.dtype]) -> dict[str:str]:
        """Предсказывает класс лица на изображении (real/fake) и возвращает результат.

        Args:
            img : Путь к изображению для классификации.

        Returns:
            str: Строка формата "Класс: {class_name}, Вероятность: {confidence:.2%}",
                 где class_name - "real" или "fake", а confidence - вероятность в процентах.

        Raises:
            FileNotFoundError: Если файл изображения не существует.
            ValueError: Если изображение невалидно или модель не может его обработать.
            RuntimeError: Если во время предсказания произошла ошибка.

        """
        try:
            result = self._model.predict(img, verbose=False)

            if not hasattr(result[0], 'probs'):
                raise ValueError("Модель не вернула вероятности классов (возможно, некорректные веса).")

            class_index = result[0].probs.top1
            class_name = result[0].names[class_index]
            confidence = result[0].probs.top1conf.item()

            return {"class": f"{class_name}", "prob": f"{confidence:.2%}"}

        except Exception as e:
            raise RuntimeError(f"Ошибка предсказания: {str(e)}")
