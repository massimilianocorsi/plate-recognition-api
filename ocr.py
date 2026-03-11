import requests
from config import PLATE_API_KEY, PLATE_API_URL, PLATE_REGION

def call_ocr(image_bytes: bytes):
    resp = requests.post(
        PLATE_API_URL,
        files={"upload": ("plate.jpg", image_bytes, "image/jpeg")},
        data={"regions": PLATE_REGION},
        headers={"Authorization": f"Token {PLATE_API_KEY}"}
    )
    data = resp.json()
    plate = None
    if "results" in data and data["results"]:
        plate = data["results"][0].get("plate", "")
        if plate:
            plate = plate.upper()
            plate = "".join(c for c in plate if c.isalnum())
    return plate, data
