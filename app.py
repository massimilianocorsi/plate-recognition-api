from flask import Flask, request, jsonify
import requests
import os

API_KEY = os.getenv("PLATE_API_KEY")
API_URL = os.getenv("https://api.platerecognizer.com/v1/plate-reader/")

app = Flask(__name__)

@app.route("/recognize", methods=["POST"])
def recognize():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    try:
        resp = requests.post(
            API_URL,
            files={"upload": file},
            headers={"Authorization": f"Token {API_KEY}"}
        )
        data = resp.json()

        # Normalizzazione targa
        plate = None
        if "results" in data and len(data["results"]) > 0:
            plate = data["results"][0].get("plate", "").upper()
            plate = "".join(c for c in plate if c.isalnum())

        return jsonify({
            "raw": data,
            "plate": plate
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000)

