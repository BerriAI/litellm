from typing import Optional

from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

# GigaChat API endpoint
GIGACHAT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"


def convert_usage(usage_data: dict[str, int]) -> Usage:
    prompt_tokens = usage_data.get("prompt_tokens", 0)
    completion_tokens = usage_data.get("completion_tokens", 0)
    precached_prompt_tokens = usage_data.get("precached_prompt_tokens", 0)
    total_tokens = usage_data.get("total_tokens", 0)

    prompt_tokens += precached_prompt_tokens
    total_tokens += precached_prompt_tokens

    prompt_tokens_details = None
    if precached_prompt_tokens > 0:
        prompt_tokens_details = PromptTokensDetailsWrapper(
            cached_tokens=precached_prompt_tokens
        )

    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tokens_details=prompt_tokens_details,
        total_tokens=total_tokens,
    )


def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
    return api_base or get_secret_str("GIGACHAT_API_BASE") or GIGACHAT_BASE_URL
