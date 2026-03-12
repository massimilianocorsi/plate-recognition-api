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

    input_tensor = preprocess_image(image)
    input_name = session_plate.get_inputs()[0].name
    outputs = session_plate.run(None, {input_name: input_tensor})

    raw = outputs[0]

    try:
        from flask import current_app
        current_app.logger.debug("yolo_plate raw shape: %s", raw.shape)
    except:
        pass

    # Gestisci shape diverse
    if raw.ndim == 3:
        squeezed = np.squeeze(raw)
        boxes = squeezed.T if squeezed.shape[0] < squeezed.shape[1] else squeezed
    elif raw.ndim == 2:
        boxes = raw
    else:
        return None

    if boxes.shape[0] == 0:
        return None

    # Filtra per confidenza > 0.25
    if boxes.shape[1] >= 5:
        mask = boxes[:, 4] > 0.25
        if not np.any(mask):
            return None
        boxes = boxes[mask]

    # Box con area maggiore
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best = boxes[int(np.argmax(areas))]

    # Coordinate: potrebbero essere normalizzate [0,1] o in pixel su 640
    # Rileva automaticamente: se tutti i valori < 2.0 → normalizzate
    coords = best[:4]
    if np.all(coords <= 2.0):
        # Normalizzate → moltiplica per dimensione originale
        x1 = int(float(coords[0]) * orig_w)
        y1 = int(float(coords[1]) * orig_h)
        x2 = int(float(coords[2]) * orig_w)
        y2 = int(float(coords[3]) * orig_h)
    else:
        # Pixel su 640 → scala alla dimensione originale
        x1 = int(float(coords[0]) * orig_w / 640)
        y1 = int(float(coords[1]) * orig_h / 640)
        x2 = int(float(coords[2]) * orig_w / 640)
        y2 = int(float(coords[3]) * orig_h / 640)

    try:
        from flask import current_app
        current_app.logger.debug("box finale: x1=%d y1=%d x2=%d y2=%d", x1, y1, x2, y2)
    except:
        pass

    # Sanity check
    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2