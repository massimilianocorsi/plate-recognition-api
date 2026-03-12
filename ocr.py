import requests
import logging
from config import PLATE_API_KEY, PLATE_API_URL, PLATE_REGION

logger = logging.getLogger(__name__)

def call_ocr(image_bytes: bytes):
    logger.debug("Invio a PlateRecognizer: %d bytes, url=%s, region=%s", 
                 len(image_bytes), PLATE_API_URL, PLATE_REGION)
    
    resp = requests.post(
        PLATE_API_URL,
        files={"upload": ("plate.jpg", image_bytes, "image/jpeg")},
        data={"regions": PLATE_REGION},
        headers={"Authorization": f"Token {PLATE_API_KEY}"}
    )
    
    logger.debug("PlateRecognizer status: %d", resp.status_code)
    logger.debug("PlateRecognizer response: %s", resp.text)
    
    data = resp.json()
    plate = None
    if "results" in data and data["results"]:
        plate = data["results"][0].get("plate", "")
        if plate:
            plate = plate.upper()
            plate = "".join(c for c in plate if c.isalnum())
    
    logger.debug("Targa estratta: %s", plate)
    return plate, data