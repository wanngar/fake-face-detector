from typing import Any
import mediapipe as mp
import numpy as np


class FaceDetector:
    """Класс для  детекции лиц на изображениях с использованием YOLO."""

    def __init__(self):
        self._model = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

    def detect(self, frame: np.ndarray) -> np.ndarray[Any, np.dtype]:
        try:
            results = self._model.process(frame)
            for i in results.detections:
                relative_bounding_box = i.location_data.relative_bounding_box
                w = int(relative_bounding_box.width * frame.shape[1])
                h = int(relative_bounding_box.height * frame.shape[0])
                x = max(0, int(relative_bounding_box.xmin * frame.shape[1]) - int(w * 0.125))
                y = max(0, int(relative_bounding_box.ymin * frame.shape[0]) - int(h * 0.125))
                cropped_frame = frame[y:y + int(h * 1.25), x:x + int(w * 1.25)]
            return cropped_frame

        except TypeError:
            raise ValueError("Face not found")

        except Exception:
            raise RuntimeError("Model runtime error")
