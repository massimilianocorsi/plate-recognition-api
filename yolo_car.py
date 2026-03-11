from ultralytics import YOLO
from PIL import Image
from config import YOLO_CAR_MODEL

model_car = YOLO(YOLO_CAR_MODEL)

def detect_car_make_model(image: Image.Image):
    results = model_car(image)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None, None, None

    box = max(boxes, key=lambda b: float(b.conf[0]))
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    label = results[0].names[cls_id]

    parts = label.split(" ", 1)
    make = parts[0]
    model = parts[1] if len(parts) > 1 else None

    return make, model, conf
