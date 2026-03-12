import onnxruntime as ort
import numpy as np
from PIL import Image
from config import YOLO_PLATE_MODEL

def preprocess_image(image: Image.Image, img_size=640):
    image = image.convert("RGB").resize((img_size, img_size))
    img_np = np.array(image, dtype=np.float32)
    img_np = img_np / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

session_plate = ort.InferenceSession(YOLO_PLATE_MODEL)

def detect_plate(image: Image.Image):
    orig_w, orig_h = image.size
    # Temporaneo: usa tutta l'immagine come area di detection
    return 0, 0, orig_w, orig_h