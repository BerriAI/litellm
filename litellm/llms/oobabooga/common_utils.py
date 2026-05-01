from typing import Optional, Union

import httpx

from litellm.litellm_core_utils.url_utils import is_url_destination_allowed_by_host
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


def _is_url_model_destination_allowed(model: str) -> bool:
    import litellm

    allowed_hosts = getattr(litellm, "provider_url_destination_allowed_hosts", []) or []
    return is_url_destination_allowed_by_host(model, allowed_hosts)


def validate_oobabooga_model_identifier(model: str) -> None:
    """Oobabooga endpoints must be configured with api_base, not model URLs."""
    if "://" not in model:
        return
    if is_url_model_destination(model) and _is_url_model_destination_allowed(model):
        return
    raise OobaboogaError(
        status_code=400,
        message=(
            "Invalid Oobabooga model identifier. Configure the endpoint with "
            "api_base instead of passing a URL as the model. To keep legacy "
            "URL-valued models for trusted endpoints, add the destination host "
            "or origin to `provider_url_destination_allowed_hosts` in "
            "litellm_settings."
        ),
    )
