from typing import Final

from fastapi import HTTPException

MODEL_ACCESS_DENIED_CLIENT_MESSAGE: Final = (
    "The requested model '{model}' is not available for this API key, or the model name is invalid. "
    "Check the models available to you and try again."
)


def model_access_denied_client_message(model: str | list[str]) -> str:
    return MODEL_ACCESS_DENIED_CLIENT_MESSAGE.format(model=model)


class ModelAccessDeniedHTTPException(HTTPException):
    def __init__(self, internal_message: str, status_code: int, detail: str | dict[str, str]) -> None:
        super().__init__(status_code=status_code, detail=detail)
        self.internal_message: Final = internal_message
