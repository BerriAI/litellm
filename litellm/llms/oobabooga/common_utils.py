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


def is_url_model_destination(model: str) -> bool:
    return model.startswith(("http://", "https://"))


def _should_reject_url_model_destinations() -> bool:
    import litellm

    return getattr(litellm, "reject_url_model_destinations", True) is True


def validate_oobabooga_model_identifier(model: str) -> None:
    """Oobabooga endpoints must be configured with api_base, not model URLs."""
    if "://" not in model:
        return
    if is_url_model_destination(model) and not _should_reject_url_model_destinations():
        return
    raise OobaboogaError(
        status_code=400,
        message=(
            "Invalid Oobabooga model identifier. Configure the endpoint with "
            "api_base instead of passing a URL as the model. To keep legacy "
            "URL-valued models for trusted inputs, set "
            "litellm.reject_url_model_destinations=False."
        ),
    )
