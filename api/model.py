import os
from ultralytics import YOLO



class FaceDetector:
    """Класс для детекции и классификации лиц на изображениях с использованием YOLO.

    Использует предобученную модель YOLO для определения, является ли лицо на изображении
    реальным (real) или поддельным (fake). Возвращает результат с вероятностью.

    Attributes:
        model (YOLO): Загруженная модель YOLO для предсказания.

    Example:
        >>> detector = FaceDetector(weights_path="weights_v1.pt")
        >>> result = detector.img_predict("face.jpg")
        >>> print(result)
        Класс: real, Вероятность: 99.50%
    """

    def __init__(self, weights_path: str):
        """Инициализирует детектор с указанными весами модели.

        Args:
            weights_path (str): Путь к файлу с весами модели (.pt).

        Raises:
            FileNotFoundError: Если файл с весами не существует.
            RuntimeError: Если не удалось загрузить модель.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Файл с весами '{weights_path}' не найден.")

        try:
            self.model = YOLO(weights_path)
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели: {str(e)}")

    def img_predict(self, img_path: str) -> str:
        """Предсказывает класс лица на изображении (real/fake) и возвращает результат.

        Args:
            img_path (str): Путь к изображению для классификации.

        Returns:
            str: Строка формата "Класс: {class_name}, Вероятность: {confidence:.2%}",
                 где class_name - "real" или "fake", а confidence - вероятность в процентах.

        Raises:
            FileNotFoundError: Если файл изображения не существует.
            ValueError: Если изображение невалидно или модель не может его обработать.
            RuntimeError: Если во время предсказания произошла ошибка.

        Example:
            >>> detector = FaceDetector("model.pt")
            >>> detector.img_predict("person.jpg")
            'Класс: fake, Вероятность: 87.33%'
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Изображение '{img_path}' не найдено.")

        try:
            result = self.model.predict(img_path, verbose=False)

            # Проверка, что модель вернула валидный результат
            if not hasattr(result[0], 'probs'):
                raise ValueError("Модель не вернула вероятности классов (возможно, некорректные веса).")

            class_index = result[0].probs.top1
            class_name = result[0].names[class_index]
            confidence = result[0].probs.top1conf.item()

            return f"Класс: {class_name}, Вероятность: {confidence:.2%}"

        except Exception as e:
            raise RuntimeError(f"Ошибка предсказания: {str(e)}")
