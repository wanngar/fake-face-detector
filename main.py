from fastapi import FastAPI
from model import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Fake Face Detector API",
    description="API for detecting fake faces with YOLO model"
)
app.include_router(router=router, prefix='/api')

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api",tags=['App'])
def health_check():
    return {"status": "success"}
