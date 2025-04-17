# Fake Face Detector API
API for detecting fake faces using YOLOv11
The project is packaged in a Docker container for easy deployment.



## Tech Stack
- YOLOv11 model for face detection
- FastAPI 


## Requirements
- Docker Engine 20.10+
- 4GB+ RAM (recommended)
- NVIDIA GPU (optional for acceleration)
## Deployment
Clone the repository:
```bash
git clone https://github.com/wanngar/fake-face-detector-api.git
cd fake-face-detector-api
```
Build the Docker image:
```bash
docker build -t fake-face-detector-api .
```
Run the container:
```bash
docker run -p 8000:8000 --name detector-api fake-face-detector-api
```
The API will be available at:
```bash
http://localhost:8000/docs
```
