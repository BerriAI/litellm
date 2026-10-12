from __future__ import annotations

from types import MappingProxyType
from typing import Final

from litellm.constants import OPENAI_CHAT_COMPLETION_PARAMS
from litellm.llms.anthropic.pass_through.messages.utils import anthropic_messages_optional_param_keys
from litellm.rust_bridge.catalog import Route
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams

_CHAT_TRANSPORT: Final = frozenset(
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

_PARAMETERS: Final = MappingProxyType(
    {
        Route.CHAT_COMPLETIONS: tuple(
            dict.fromkeys(name for name in OPENAI_CHAT_COMPLETION_PARAMS if name not in _CHAT_TRANSPORT)
        ),
        Route.RESPONSES: (*ResponsesAPIOptionalRequestParams.__annotations__, "extra_body"),
        Route.MESSAGES: tuple(sorted(anthropic_messages_optional_param_keys())),
    }
)


def provider_parameters(route: str) -> tuple[str, ...]:
    return _PARAMETERS[Route(route)]
