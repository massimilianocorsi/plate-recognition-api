from flask import Flask, request, jsonify
import requests
import os

API_KEY = os.getenv("PLATE_API_KEY")
API_URL = os.getenv("PLATE_API_URL")
REGIONS = os.getenv("REGIONS")

app = Flask(__name__)

@app.route("/recognize", methods=["POST"])
def recognize():
    # Accept ANY file field name
    if len(request.files) == 0:
        return jsonify({"error": "No file uploaded"}), 400

    # Take the first file in the request
    file_key = next(iter(request.files))
    file = request.files[file_key]

    try:
        resp = requests.post(
            API_URL,
            files={"upload": file},
            data={"regions": "it"},
            headers={"Authorization": f"Token {API_KEY}"}
        )
        data = resp.json()

        # Normalize plate
        plate = None
        if "results" in data and len(data["results"]) > 0:
            plate = data["results"][0].get("plate", "").upper()
            plate = "".join(c for c in plate if c.isalnum())

        return jsonify({
            "raw": data,
            "plate": plate,
            "field_used": file_key
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000)
