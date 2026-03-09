FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PLATE_API_KEY="INSERISCI_LA_TUA_API_KEY"

EXPOSE 7000

CMD ["python", "app.py"]
