FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import whisper; whisper.load_model('small')"

COPY . .

EXPOSE 2345

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "2345"]
