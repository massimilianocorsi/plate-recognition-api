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
    input_tensor = preprocess_image(image)
    input_name = session_plate.get_inputs()[0].name
    outputs = session_plate.run(None, {input_name: input_tensor})

    raw = outputs[0]
    app_logger_available = False
    try:
        from flask import current_app
        current_app.logger.debug("yolo_plate raw shape: %s", raw.shape)
        app_logger_available = True
    except:
        pass

    # Gestisci shape diverse
    # Caso A: [1, 6, N] → trasponi a [N, 6]
    # Caso B: [1, N, 6] → squeeze a [N, 6]
    # Caso C: [N, 6]    → già ok
    if raw.ndim == 3:
        squeezed = np.squeeze(raw)  # → [6, N] o [N, 6]
        if squeezed.shape[0] < squeezed.shape[1]:
            boxes = squeezed.T  # → [N, 6]
        else:
            boxes = squeezed    # → [N, 6]
    elif raw.ndim == 2:
        boxes = raw
    else:
        return None

    if boxes.shape[0] == 0:
        return None

    # Filtra per confidenza > 0.25 (colonna 4)
    if boxes.shape[1] >= 5:
        mask = boxes[:, 4] > 0.25
        if np.any(mask):
            boxes = boxes[mask]

    # Scegli la box con area maggiore
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best = boxes[int(np.argmax(areas))]

    x1 = int(float(best[0]))
    y1 = int(float(best[1]))
    x2 = int(float(best[2]))
    y2 = int(float(best[3]))

    return x1, y1, x2, y2