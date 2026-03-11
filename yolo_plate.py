from ultralytics import YOLO
from PIL import Image
from config import YOLO_PLATE_MODEL

model_plate = YOLO(YOLO_PLATE_MODEL)

def detect_plate(image: Image.Image):
    results = model_plate(image)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None

    # choose largest box
    box = max(
        boxes,
        key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])
    )
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    return x1, y1, x2, y2
