from collections.abc import Mapping
from typing import Final

from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

# GigaChat API endpoint
GIGACHAT_BASE_URL: Final = "https://gigachat.devices.sberbank.ru/api/v1"


def convert_usage(usage_data: Mapping[str, int]) -> Usage:
    precached_prompt_tokens: Final = usage_data.get("precached_prompt_tokens", 0)
    prompt_tokens_details: Final = (
        PromptTokensDetailsWrapper(cached_tokens=precached_prompt_tokens) if precached_prompt_tokens > 0 else None
    )

    return Usage(
        prompt_tokens=usage_data.get("prompt_tokens", 0) + precached_prompt_tokens,
        completion_tokens=usage_data.get("completion_tokens", 0),
        prompt_tokens_details=prompt_tokens_details,
        total_tokens=usage_data.get("total_tokens", 0) + precached_prompt_tokens,
    )


def get_api_base(api_base: str | None = None) -> str | None:
    return api_base or get_secret_str("GIGACHAT_API_BASE") or GIGACHAT_BASE_URL
