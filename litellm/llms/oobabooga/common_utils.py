from typing import Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class OobaboogaError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[Union[dict, httpx.Headers]] = None,
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)


def validate_oobabooga_model_identifier(model: str) -> None:
    """Oobabooga endpoints must be configured with api_base, not model URLs."""
    if "://" not in model:
        return
    raise OobaboogaError(
        status_code=400,
        message=(
            "Invalid Oobabooga model identifier. Configure the endpoint with "
            "api_base instead of passing a URL as the model."
        ),
    )
