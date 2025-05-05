FROM python:3.12

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install numpy==1.26.4 --no-cache-dir
RUN pip install torch torchvision  --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
RUN pip install ultralytics --no-deps --no-cache-dir
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
