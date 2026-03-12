# detect_car.py
import sys
import traceback
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
    try:
        input_tensor = preprocess_image(image)
        input_name = session_car.get_inputs()[0].name
        outputs = session_car.run(None, {input_name: input_tensor})

        raw = outputs[0]
        print(f"DEBUG raw shape: {raw.shape}", file=sys.stderr, flush=True)

        # Gestisci shape diversi in uscita dal modello ONNX
        # Caso A: [1, 84, 8400] → trasponi a [8400, 84]
        # Caso B: [1, 8400, 84] → già nel formato giusto
        # Caso C: [8400, 84]    → già nel formato giusto

        if raw.ndim == 3:
            # [batch, a, b]
            squeezed = np.squeeze(raw)  # → [a, b]
            print(f"DEBUG squeezed shape: {squeezed.shape}", file=sys.stderr, flush=True)

            if squeezed.shape[0] < squeezed.shape[1]:
                # es. [84, 8400] → serve trasporre
                output = squeezed.T
            else:
                # es. [8400, 84] → già ok
                output = squeezed
        elif raw.ndim == 2:
            output = raw
        else:
            print(f"DEBUG shape inattesa: {raw.shape}", file=sys.stderr, flush=True)
            return None, None, None

        print(f"DEBUG output finale shape: {output.shape}", file=sys.stderr, flush=True)

        # Verifica che ci siano almeno 5 colonne (4 box + almeno 1 classe)
        if output.shape[1] < 5:
            print(f"DEBUG troppe poche colonne: {output.shape[1]}", file=sys.stderr, flush=True)
            return None, None, None

        scores = output[:, 4:]
        print(f"DEBUG scores shape: {scores.shape}", file=sys.stderr, flush=True)

        class_ids = np.argmax(scores, axis=1)       # [N]
        confidences = np.max(scores, axis=1)         # [N]

        mask = confidences > 0.25
        print(f"DEBUG rilevazioni sopra soglia: {np.sum(mask)}", file=sys.stderr, flush=True)

        if not np.any(mask):
            return None, None, None

        filtered_output       = output[mask]          # [M, 84]
        filtered_confidences  = confidences[mask]     # [M]
        filtered_class_ids    = class_ids[mask]       # [M]

        best_idx = int(np.argmax(filtered_confidences))
        print(f"DEBUG best_idx: {best_idx}", file=sys.stderr, flush=True)

        cls_id = int(filtered_class_ids[best_idx])
        conf   = float(filtered_confidences[best_idx])
        box    = filtered_output[best_idx][:4].tolist()  # [x_c, y_c, w, h]

        print(f"DEBUG risultato → cls_id={cls_id}, conf={conf:.3f}, box={box}", file=sys.stderr, flush=True)

        return cls_id, conf, box

    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise