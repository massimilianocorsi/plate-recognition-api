import onnxruntime as ort
import numpy as np
from PIL import Image
from config import YOLO_PLATE_MODEL
import logging

logger = logging.getLogger(__name__)

def preprocess_image(image: Image.Image, img_size=640):
    image = image.convert("RGB").resize((img_size, img_size))
    img_np = np.array(image, dtype=np.float32) / 255.0
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

    if raw.ndim == 3:
        squeezed = np.squeeze(raw)
        boxes = squeezed.T if squeezed.shape[0] < squeezed.shape[1] else squeezed
    elif raw.ndim == 2:
        boxes = raw
    else:
        return None

    # Colonna 4 = confidenza
    confidences = boxes[:, 4]
    logger.debug("plate max conf: %.4f", float(np.max(confidences)))

    mask = confidences > 0.25
    if not np.any(mask):
        return None

    boxes = boxes[mask]
    confidences = confidences[mask]

    best = boxes[int(np.argmax(confidences))]
    logger.debug("best box raw: %s", best[:5].tolist())

    # Coordinate in pixel su 640 → scala a dimensione originale
    # Formato: x_center, y_center, width, height
    cx = float(best[0]) / 640 * orig_w
    cy = float(best[1]) / 640 * orig_h
    bw = float(best[2]) / 640 * orig_w
    bh = float(best[3]) / 640 * orig_h

    x1 = int(cx - bw / 2)
    y1 = int(cy - bh / 2)
    x2 = int(cx + bw / 2)
    y2 = int(cy + bh / 2)

    # Clamp ai bordi
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(orig_w, x2)
    y2 = min(orig_h, y2)

    logger.debug("plate box finale: x1=%d y1=%d x2=%d y2=%d", x1, y1, x2, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2