# Cambiato: utilizzo ONNX invece di ultralytics YOLO
import onnxruntime as ort
import numpy as np
from PIL import Image
from config import YOLO_CAR_MODEL

def preprocess_image(image: Image.Image, img_size=640):
    image = image.convert("RGB").resize((img_size, img_size))
    img_np = np.array(image, dtype=np.float32)
    img_np = img_np / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

session_car = ort.InferenceSession(YOLO_CAR_MODEL)

def detect_car_make_model(image: Image.Image):
    input_tensor = preprocess_image(image)
    input_name = session_car.get_inputs()[0].name
    outputs = session_car.run(None, {input_name: input_tensor})
    # Post-processing YOLOv8 ONNX: estrai le box
    # Assumiamo che outputs[0] sia (N, 6): x1, y1, x2, y2, conf, class
    boxes = outputs[0]
    if boxes.shape[0] == 0:
        return None, None, None
    # Scegli la box con confidenza massima
    box = max(boxes, key=lambda b: float(b[4]))
    cls_id = int(box[5].item()) if np.isscalar(box[5]) or box[5].shape == () else int(box[5][0].item())
    conf = float(box[4].item()) if np.isscalar(box[4]) or box[4].shape == () else float(box[4][0].item())
    # label = ... (da implementare se hai la lista delle classi)
    return cls_id, conf, box[:4]
