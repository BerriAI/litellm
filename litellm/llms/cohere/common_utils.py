from typing import Optional

from litellm.llms.base_llm.transformation import BaseLLMException


class CohereError(BaseLLMException):
    def __init__(self, status_code, message):
        super().__init__(status_code=status_code, message=message)


def validate_environment(*, api_key: Optional[str], headers: dict) -> dict:
    headers.update(
        {
            "Request-Source": "unspecified:litellm",
            "accept": "application/json",
            "content-type": "application/json",
        }
    )
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers
