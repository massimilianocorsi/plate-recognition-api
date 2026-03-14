import anthropic
import base64
import logging
from PIL import Image
from config import ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def detect_car_make_model_anthropic(image: Image.Image):
    """
    Sends the full-frame image to Anthropic Claude and asks it to identify
    the car make and model. Returns (make, model, None) — no confidence score.
    """
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    image_b64 = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

    prompt = (
        "Look at this image. Identify plate number and the car brand (make) and model visible in the photo. "
        "Reply ONLY with a JSON object with two keys: \"plate\" and \"make\" and \"model\". "
        "If you cannot determine one or both, use null for that field. "
        "Example: {\"plate\": \"AZ705RT\",\"make\": \"Toyota\", \"model\": \"Corolla\"}"
    )

    logger.debug("Invio immagine ad Anthropic per riconoscimento auto")

    message = _client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    )

    raw_text = message.content[0].text.strip()
    logger.debug("Risposta Anthropic: %s", raw_text)

    import json
    try:
        result = json.loads(raw_text)
        car_make = result.get("make")
        car_model = result.get("model")
        car_plate = result.get("plate")
    except json.JSONDecodeError:
        logger.warning("Risposta Anthropic non parsabile come JSON: %s", raw_text)
        car_make = None
        car_model = None
        car_plate = None

    logger.debug("Anthropic → make=%s model=%s plate=%s", car_make, car_model, car_plate)
    return car_make, car_model, car_plate, None
