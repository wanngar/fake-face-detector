from fastapi.responses import JSONResponse


def make_error_response(status_code: int, content: str) -> JSONResponse:
    """Унифицированный формат для всех ошибок"""
    error_data = {
        "status": "error",
        "detail": {
            "message": content,
        }
    }
    return JSONResponse(status_code=status_code, content=error_data)