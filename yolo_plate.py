# Cambiato: utilizzo ONNX invece di ultralytics YOLO
import onnxruntime as ort
import numpy as np
from PIL import Image
from config import YOLO_PLATE_MODEL

def preprocess_image(image: Image.Image, img_size=640):
    # Ridimensiona e normalizza l'immagine per YOLOv8 ONNX
    image = image.convert("RGB").resize((img_size, img_size))
    img_np = np.array(image, dtype=np.float32)
    img_np = img_np / 255.0  # Normalizza
    img_np = np.transpose(img_np, (2, 0, 1))  # Canali prima
    img_np = np.expand_dims(img_np, axis=0)  # Batch dim
    return img_np

# Carica il modello ONNX
session_plate = ort.InferenceSession(YOLO_PLATE_MODEL)

# Funzione di inferenza

def detect_plate(image: Image.Image):
    input_tensor = preprocess_image(image)
    input_name = session_plate.get_inputs()[0].name
    outputs = session_plate.run(None, {input_name: input_tensor})
    # Post-processing da adattare in base all'output del tuo modello ONNX
    # Qui va implementata la logica per estrarre le box come faceva YOLO
    # ...
    return outputs
