from fastapi import FastAPI
from app.api.routes.predict import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Fake Face Detector")
app.include_router(router=router, prefix='/api')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
