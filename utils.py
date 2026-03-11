import io
from PIL import Image

MAX_SIZE = 3_500_000  # 3.5 MB

def compress_image(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    quality = 85
    while True:
        buffer.seek(0)
        image.save(buffer, format="JPEG", quality=quality)
        size = buffer.tell()
        if size <= MAX_SIZE or quality <= 30:
            break
        quality -= 5
    return buffer.getvalue()

def is_valid_italian_plate(plate: str) -> bool:
    if not plate or len(plate) != 7:
        return False
    return plate[:2].isalpha() and plate[2:5].isdigit() and plate[5:].isalpha()
