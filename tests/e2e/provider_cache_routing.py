from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import Final

from models import LiteLLMParamsBody, ModelMode

LIVE_PROVIDER_REQUIRED: Final[ContextVar[bool]] = ContextVar("live_provider_required", default=False)


def route_cache_model(
    params: LiteLLMParamsBody, base_for: Callable[[str], str | None], *, enabled: bool, mode: ModelMode | None = None,
) -> LiteLLMParamsBody:
    if not enabled or mode == "realtime" or LIVE_PROVIDER_REQUIRED.get() or params.api_base is not None or params.mock_response is not None:
        return params
    provider: Final = params.model.partition("/")[0]
    if provider not in {"openai", "anthropic"} or params.litellm_credential_name is not None:
        return params
    base: Final = base_for(provider)
    if base is None:
        return params
    return params.model_copy(update={"api_base": f"{base}/v1" if provider == "openai" else base})
