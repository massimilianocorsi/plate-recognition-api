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
    
    # 1. Rimuovi la dimensione batch e trasponi: da [1, 84, 8400] a [8400, 84]
    output = np.squeeze(outputs[0]).T
    
    # 2. Trova la confidenza massima per ogni riga (dopo l'indice 4 iniziano i punteggi delle classi)
    scores = output[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    
    # 3. Filtra per una soglia minima (es. 0.25)
    mask = confidences > 0.25
    if not np.any(mask):
        return None, None, None
        
    # 4. Prendi il migliore
    best_idx = np.argmax(confidences[mask])
    
    # Estrai i valori usando indici scalari sicuri
    cls_id = int(class_ids[mask][best_idx])
    conf = float(confidences[mask][best_idx])
    box = output[mask][best_idx][:4] # [x_center, y_center, width, height]
    
    return cls_id, conf, box
