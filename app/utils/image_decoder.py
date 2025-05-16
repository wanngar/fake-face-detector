import base64
import cv2


def cv2_to_base64(detected_face_np):
    image_rgb = cv2.cvtColor(detected_face_np, cv2.COLOR_BGR2RGB)
    _, img_encoded = cv2.imencode('.jpg', image_rgb)
    return base64.b64encode(img_encoded.tobytes()).decode('utf-8')