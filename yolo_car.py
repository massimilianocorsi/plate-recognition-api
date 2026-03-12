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
    
    print("outputs[0].shape:", outputs[0].shape)
    
    output = np.squeeze(outputs[0]).T
    print("output.shape after squeeze+T:", output.shape)
    
    scores = output[:, 4:]
    print("scores.shape:", scores.shape)
    
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    print("class_ids.shape:", class_ids.shape)
    print("confidences.shape:", confidences.shape)
    
    mask = confidences > 0.25
    print("mask sum:", np.sum(mask))
    
    if not np.any(mask):
        return None, None, None
    
    filtered_output = output[mask]
    filtered_confidences = confidences[mask]
    filtered_class_ids = class_ids[mask]
    
    print("filtered shapes:", filtered_output.shape, filtered_confidences.shape, filtered_class_ids.shape)
    
    best_idx = int(np.argmax(filtered_confidences))
    print("best_idx:", best_idx, type(best_idx))
    
    cls_id = int(filtered_class_ids[best_idx])
    conf = float(filtered_confidences[best_idx])
    box = filtered_output[best_idx][:4]
    
    return cls_id, conf, box