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
    squeezed = np.squeeze(raw)
    boxes = squeezed.T if squeezed.shape[0] < squeezed.shape[1] else squeezed

    confidences = np.max(boxes[:, 4:], axis=1)

    try:
        from flask import current_app
        current_app.logger.debug("plate model - max conf: %.4f", float(np.max(confidences)))
        current_app.logger.debug("plate model - top5 conf: %s", np.sort(confidences)[::-1][:5].tolist())
        current_app.logger.debug("plate model - box con max conf: %s", boxes[int(np.argmax(confidences))][:6].tolist())
    except:
        pass

    # Temporaneo: tutta l'immagine
    return 0, 0, orig_w, orig_h