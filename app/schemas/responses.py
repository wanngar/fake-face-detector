from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

class ModelErrorResponse(BaseModel):
    status: str = "error"
    error: ErrorDetail

class ModelSuccessResponse(BaseModel):
    status: str = "success"
    result: Dict[str, Any]
    face_image: str