from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import litellm
from litellm.rust_bridge import failures
from litellm.rust_bridge.public_call import optional_str
from litellm.types.utils import ModelResponse


def connection_defaults(provider: str) -> tuple[str | None, str | None]:
    if provider == "anthropic":
        return litellm.anthropic_key or litellm.api_key, litellm.api_base
    return None, None


def response(value: Mapping[str, object]) -> ModelResponse:
    return ModelResponse(**value)


def map_failure(error: Exception, request: Mapping[str, object]) -> Exception:
    provider: Final = optional_str(request.get("custom_llm_provider")) or str(request["model"]).partition("/")[0]
    return failures.map_native_failure(
        error,
        str(request["model"]),
        provider,
        request,
        optional_str(request.get("api_base")) or optional_str(request.get("base_url")),
    )
