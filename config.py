import os

PLATE_API_KEY = os.getenv("PLATE_API_KEY")
PLATE_API_URL = os.getenv("PLATE_API_URL", "https://api.platerecognizer.com/v1/plate-reader/")
PLATE_REGION = os.getenv("PLATE_REGION", "it")

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

CACHE_TTL = int(os.getenv("CACHE_TTL", "5"))

YOLO_PLATE_MODEL = "yolov8n_it_plate.onnx"
YOLO_CAR_MODEL = "yolov8n_car_model.onnx"
