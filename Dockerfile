# PLK AirCast — Hugging Face Spaces (Docker)
FROM python:3.9-slim

# เวลาไทย (โค้ดใช้นาฬิกาเครื่องเป็นเวลาไทย)
ENV TZ=Asia/Bangkok PYTHONUNBUFFERED=1 PORT=7860
RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face รันคอนเทนเนอร์ด้วย user id 1000 — ให้ user นี้เขียนไฟล์ในโฟลเดอร์แอปได้ (realtime_log.csv)
RUN useradd -m -u 1000 user
WORKDIR /app
COPY --chown=user requirements-space.txt .
RUN pip install --no-cache-dir -r requirements-space.txt
COPY --chown=user . .
USER user

EXPOSE 7860
CMD ["python", "app.py"]
