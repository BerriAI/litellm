from typing import Final

from fastapi import HTTPException

import litellm
from litellm.constants import MODEL_ACCESS_DENIED_MESSAGE_MODEL_PLACEHOLDER


def client_facing_model_access_denied_message(internal_message: str, model: str | list[str]) -> str:
    template: Final = litellm.model_access_denied_message
    if not template:
        return internal_message
    return template.replace(MODEL_ACCESS_DENIED_MESSAGE_MODEL_PLACEHOLDER, str(model))


class ModelAccessDeniedHTTPException(HTTPException):
    def __init__(self, internal_message: str, status_code: int, detail: str | dict[str, str]) -> None:
        super().__init__(status_code=status_code, detail=detail)
        self.internal_message: Final = internal_message
