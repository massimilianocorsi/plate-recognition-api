# detect_car.py
import logging
import onnxruntime as ort
import numpy as np
from PIL import Image
from config import YOLO_CAR_MODEL

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def preprocess_image(image: Image.Image, img_size=640):
    image = image.convert("RGB").resize((img_size, img_size))
    img_np = np.array(image, dtype=np.float32)
    img_np = img_np / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

session_car = ort.InferenceSession(YOLO_CAR_MODEL)

def detect_car_make_model(image: Image.Image):
    try:
        input_tensor = preprocess_image(image)
        input_name = session_car.get_inputs()[0].name
        outputs = session_car.run(None, {input_name: input_tensor})

        raw = outputs[0]
        logger.debug("raw shape: %s", raw.shape)

        if raw.ndim == 3:
            squeezed = np.squeeze(raw)
            logger.debug("squeezed shape: %s", squeezed.shape)
            output = squeezed.T if squeezed.shape[0] < squeezed.shape[1] else squeezed
        elif raw.ndim == 2:
            output = raw
        else:
            logger.error("shape inattesa: %s", raw.shape)
            return None, None, None

        logger.debug("output shape: %s", output.shape)

        if output.shape[1] < 5:
            logger.error("troppe poche colonne: %d", output.shape[1])
            return None, None, None

        scores = output[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        mask = confidences > 0.25
        logger.debug("rilevazioni sopra soglia: %d", np.sum(mask))

        if not np.any(mask):
            return None, None, None

        filtered_output      = output[mask]
        filtered_confidences = confidences[mask]
        filtered_class_ids   = class_ids[mask]

        best_idx = int(np.argmax(filtered_confidences))
        logger.debug("best_idx: %d", best_idx)

        cls_id = int(filtered_class_ids[best_idx])
        conf   = float(filtered_confidences[best_idx])
        box    = filtered_output[best_idx][:4].tolist()

        logger.debug("risultato → cls_id=%d, conf=%.3f, box=%s", cls_id, conf, box)
        return cls_id, conf, box

    except Exception as e:
        logger.exception("Errore in detect_car_make_model: %s", e)
        raise