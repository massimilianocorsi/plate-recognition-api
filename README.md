# 📘 README.md — Plate Recognition API (Docker + ParkPow)

```markdown
# 🚗 Plate Recognition API  
A lightweight Dockerized microservice for license plate recognition using the ParkPow / PlateRecognizer API  
(2,500 free requests per month included)

## 📌 Overview

This project provides a **REST microservice** exposing a single endpoint:

```
POST /recognize
```

The service accepts an image containing a vehicle, forwards it to the  
**ParkPow / PlateRecognizer** cloud API, and returns:

- the recognized license plate  
- the raw API response  
- a normalized plate string (A–Z, 0–9 only)

This microservice is designed to be:

- lightweight  
- easy to deploy  
- fully containerized  
- ideal for n8n, Node‑RED, Telegram/WhatsApp bots  
- production‑ready for Kubernetes / OpenShift  

---

## 🧱 Architecture

```
Client → /recognize → Plate API → ParkPow Cloud → Normalized Plate
```

The container does **not** run any local ML model.  
All recognition is performed by the ParkPow cloud service.

---

## 🚀 Quick Start

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-user>/plate-api.git
cd plate-api
```

### 2️⃣ Set your API key

You can set it in the Dockerfile or pass it as an environment variable at runtime.

### 3️⃣ Build the container

```bash
docker build -t plate-api .
```

### 4️⃣ Run the container

```bash
docker run -d -p 7000:7000 \
  -e PLATE_API_KEY="YOUR_API_KEY" \
  plate-api
```

---

## 📡 API Endpoint

### `POST /recognize`

**Body (multipart/form-data):**

| Field | Type | Description |
|-------|------|-------------|
| image | file | Vehicle image |

**Example using curl:**

```bash
curl -X POST http://localhost:7000/recognize \
  -F "image=@/path/to/vehicle.jpg"
```

**Example response:**

```json
{
  "raw": {
    "results": [
      {
        "plate": "ab777c",
        "score": 0.92,
        "region": "it"
      }
    ]
  },
  "plate": "AB777C"
}
```

---

## 🧩 Integration with n8n

1. Add an **HTTP Request** node  
2. Method: `POST`  
3. URL: `http://plate-api:7000/recognize`  
4. Body: `Form-Data`  
5. Field: `image` → Binary File (e.g., `data`)  

Normalize the plate in the next node:

```javascript
return [{
  json: {
    plate: $json.plate
  }
}];
```

---

## 🛠 Project Files

### `app.py`
Flask microservice that receives the image and calls the ParkPow API.

### `requirements.txt`
Minimal dependencies:

```
Flask
requests
```

### `Dockerfile`
Lightweight Python 3.11‑slim container.

---

## 🧪 Local Testing (without Docker)

```bash
export PLATE_API_KEY="YOUR_API_KEY"
python app.py
```

Then:

```bash
curl -X POST http://localhost:7000/recognize \
  -F "image=@test.jpg"
```

---

## ☸️ Kubernetes Deployment (optional)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: plate-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: plate-api
  template:
    metadata:
      labels:
        app: plate-api
    spec:
      containers:
      - name: plate-api
        image: registry/plate-api:latest
        env:
        - name: PLATE_API_KEY
          value: "YOUR_API_KEY"
        ports:
        - containerPort: 7000
---
apiVersion: v1
kind: Service
metadata:
  name: plate-api
spec:
  selector:
    app: plate-api
  ports:
  - port: 7000
    targetPort: 7000
```

---

## 📄 License

MIT License — free for personal and commercial use.

---

## 👨‍💻 Author

Massimiliano Corsi  
Cloud Architect & Automation Engineer  
Rome, Italy
```

---

If you want, I can also generate:

- a **logo** for the project  
- a **docker-compose.yml**  
- a **Helm chart**  
- a **GitHub Actions CI/CD pipeline**  
- a **full n8n workflow** ready to import  
