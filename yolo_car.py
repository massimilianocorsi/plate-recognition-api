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
    # Post-processing da adattare in base all'output del tuo modello ONNX
    # Qui va implementata la logica per estrarre le box/classi come faceva YOLO
    # ...
    return outputs
