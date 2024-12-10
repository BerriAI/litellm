from typing import List, Optional

from litellm.llms.base_llm.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues


class CohereError(BaseLLMException):
    def __init__(self, status_code, message):
        super().__init__(status_code=status_code, message=message)


def validate_environment(
    headers: dict,
    model: str,
    messages: List[AllMessageValues],
    optional_params: dict,
    api_key: Optional[str] = None,
) -> dict:
    """
    Return headers to use for cohere chat completion request

    Cohere API Ref: https://docs.cohere.com/reference/chat
    Expected headers:
    {
        "Request-Source": "unspecified:litellm",
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": "bearer $CO_API_KEY"
    }
    """
    headers.update(
        {
            "Request-Source": "unspecified:litellm",
            "accept": "application/json",
            "content-type": "application/json",
        }
    )
    if api_key:
        headers["Authorization"] = f"bearer {api_key}"
    return headers
