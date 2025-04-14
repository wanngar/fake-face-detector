from fastapi import APIRouter
from .predict import router as model_router

router = APIRouter()
router.include_router(router=model_router)
