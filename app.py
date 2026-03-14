from flask import Flask, request, jsonify
from PIL import Image

from yolo_plate import detect_plate
# from yolo_car import detect_car_make_model  # disabilitato: sostituito da Anthropic
from anthropic_car import detect_car_make_model_anthropic
from direction import classify_plate_type, get_direction
from dedupe import make_fingerprint, is_duplicate
from utils import compress_image, is_valid_italian_plate
from ocr import call_ocr

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)

app = Flask(__name__)

@app.route("/recognize", methods=["POST"])
def recognize():
    if len(request.files) == 0:
        return jsonify({"error": "no_file"}), 400

    camera_id = request.args.get("camera", "unknown")
    file_key = next(iter(request.files))
    file = request.files[file_key]

    try:
        image = Image.open(file.stream).convert("RGB")

        # 1) Plate detection with YOLO
        box = detect_plate(image)
        if not box:
            return jsonify({"error": "no_plate_detected"}), 404

        x1, y1, x2, y2 = box
        app.logger.debug("box: x1=%d y1=%d x2=%d y2=%d", x1, y1, x2, y2)

        # 2) Duplicate suppression (Redis)
        fingerprint = make_fingerprint(x1, y1, x2, y2, camera_id)
        if is_duplicate(fingerprint):
            return jsonify({
                "status": "ignored_duplicate",
                "camera": camera_id
            }), 200

        # 3) Plate type and direction
        plate_type, ratio = classify_plate_type(x1, y1, x2, y2)
        direction = get_direction(plate_type)
        app.logger.debug("plate_type=%s ratio=%.2f direction=%s", plate_type, ratio, direction)

        # 4) Crop plate and compress
        cropped_plate = image.crop((x1, y1, x2, y2))
        compressed_bytes = compress_image(cropped_plate)
        app.logger.debug("compressed_bytes size: %d", len(compressed_bytes))

        # 5) Exit: no OCR, no car model — TEMPORANEAMENTE DISABILITATO
        # if direction == "egress":
        #     return jsonify({
        #         "direction": direction,
        #         "plate": None,
        #         "plate_valid": False,
        #         "plate_type": plate_type,
        #         "ratio": ratio,
        #         "ocr_called": False,
        #         "car_make": None,
        #         "car_model": None,
        #         "car_confidence": None,
        #         "camera": camera_id
        #     }), 200

        # 6) Entry: car make/model on full frame
        # car_make, car_model, car_conf = detect_car_make_model(image)  # YOLO disabilitato
        car_make, car_model, car_conf = detect_car_make_model_anthropic(image)

        app.logger.debug("car: make=%s model=%s conf=%s", car_make, car_model, car_conf)

        # 7) Entry: OCR on plate crop
        plate, raw_ocr = call_ocr(compressed_bytes)
        plate_valid = is_valid_italian_plate(plate) if plate else False
        app.logger.debug("plate=%s valid=%s", plate, plate_valid)

        return jsonify({
            "direction": direction,
            "plate": plate,
            "plate_valid": plate_valid,
            "plate_type": plate_type,
            "ratio": ratio,
            "ocr_called": True,
            "car_make": car_make,
            "car_model": car_model,
            "car_confidence": car_conf,
            "camera": camera_id,
            "ocr_raw": raw_ocr
        }), 200

    except Exception as e:
        import traceback
        app.logger.error("ERRORE RECOGNIZE:\n%s", traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000)