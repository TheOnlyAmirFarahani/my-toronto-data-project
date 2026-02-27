FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV SERVICE_NAME=training

WORKDIR /app

COPY training/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ /app/data
COPY training/ /app

CMD ["python", "training.py"]
