from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import litellm
from litellm.constants import OPENAI_CHAT_COMPLETION_PARAMS
from litellm.rust_bridge import failures
from litellm.rust_bridge.public_call import optional_str
from litellm.types.utils import ModelResponse

_TRANSPORT_PARAMETERS: Final = frozenset(
    {
        "api_base",
        "api_key",
        "api_version",
        "deployment_id",
        "organization",
        "base_url",
        "default_headers",
        "timeout",
        "request_timeout",
        "max_retries",
        "extra_headers",
    }
)
PARAMETERS: Final = tuple(name for name in OPENAI_CHAT_COMPLETION_PARAMS if name not in _TRANSPORT_PARAMETERS)


def connection_defaults(provider: str) -> tuple[str | None, str | None]:
    if provider == "anthropic":
        return litellm.anthropic_key or litellm.api_key, litellm.api_base
    return None, None


def response(value: Mapping[str, object]) -> ModelResponse:
    return ModelResponse(**value)


def arguments(request: Mapping[str, object]) -> Mapping[str, object]:
    return request


def map_failure(error: Exception, request: Mapping[str, object]) -> Exception:
    provider: Final = optional_str(request.get("custom_llm_provider")) or str(request["model"]).partition("/")[0]
    return failures.map_native_failure(
        error,
        str(request["model"]),
        provider,
        arguments(request),
        optional_str(request.get("api_base")) or optional_str(request.get("base_url")),
    )
