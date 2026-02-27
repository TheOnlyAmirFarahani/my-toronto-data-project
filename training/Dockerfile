FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV SERVICE_NAME=training

WORKDIR /app

COPY training/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install wget and download the assault data CSV
RUN apt-get update && \
    apt-get install -y wget && \
    mkdir -p /app/data && \
    wget -O /app/data/assault_data.csv \
    "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/Assault_Open_Data/FeatureServer/replicafilescache/Assault_Open_Data_8153726331132086508.csv" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY training/ /app

CMD ["python", "training.py"]
