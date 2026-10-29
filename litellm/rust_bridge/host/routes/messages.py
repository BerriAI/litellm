from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Final, cast  # noqa: TID251  # narrows the normalized native payload to the public TypedDict

import httpx
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.core_helpers import drops_unsupported_params, normalize_drop_params
from litellm.litellm_core_utils.request_timeout_resolver import get_configured_request_timeout
from litellm.llms.anthropic.messages.convertible_blocks import (
    drop_non_string_texts,
    emptied_messages_error,
    malformed_text_error,
)
from litellm.llms.anthropic.messages.utils import is_reasoning_auto_summary_enabled
from litellm.rust_bridge import failures
from litellm.rust_bridge.public_call import known_params, optional_sequence, optional_str
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse

_DROP_PATHS: Final = TypeAdapter(list[object])
_MESSAGE_LIST: Final = TypeAdapter(list[dict[str, object]])
_BODY_FIELDS: Final = frozenset(
    {
        "max_tokens",
        "metadata",
        "stop_sequences",
        "stream",
        "system",
        "temperature",
        "thinking",
        "tool_choice",
        "tools",
        "top_k",
        "inference_geo",
        "top_p",
        "mcp_servers",
        "context_management",
        "compaction",
        "container",
        "output_format",
        "speed",
        "output_config",
        "cache_control",
        "reasoning_effort",
        "safeguards",
    }
)


class _RejectedMessages(ValueError):
    messages_request_error: Final = True


@dataclass(frozen=True, slots=True)
class MessagesSettings:
    drop_params: bool
    reasoning_auto_summary: bool
    additional_drop_params: Sequence[str]
    request_timeout: float | None


def response(value: Mapping[str, object]) -> AnthropicMessagesResponse:
    return cast(  # cast-ok: AnthropicMessagesResponse is a TypedDict over the normalized native payload
        AnthropicMessagesResponse,
        dict(value),
    )


def body_params(request: Mapping[str, object]) -> Mapping[str, object]:
    return known_params(_BODY_FIELDS, request)


def messages(request: Mapping[str, object]) -> object:
    sent: Final = request.get("messages")
    error: Final = malformed_text_error(optional_sequence(sent) or ())
    if error is None:
        return sent
    if not drops_unsupported_params(request.get("drop_params")):
        raise _RejectedMessages(error)
    try:
        listed: Final = _MESSAGE_LIST.validate_python(sent)
    except ValidationError:
        return sent
    kept: Final = drop_non_string_texts(listed, request.get("drop_params"))
    emptied: Final = emptied_messages_error(listed, kept)
    if emptied is not None:
        raise _RejectedMessages(emptied)
    return kept


def stream_hidden_params(headers: Sequence[tuple[str, str]]) -> Mapping[str, object]:
    from litellm.llms.anthropic.messages.native.streaming_iterator import (
        anthropic_messages_stream_hidden_params,
    )

    return anthropic_messages_stream_hidden_params(httpx.Headers(list(headers)))


def map_failure(error: Exception, request: Mapping[str, object], request_provider: str) -> Exception:
    if getattr(error, "messages_request_error", False):
        return litellm.BadRequestError(
            message=str(error),
            model=str(request["model"]).removeprefix(f"{request_provider}/"),
            llm_provider=request_provider,
        )
    return failures.map_native_failure(
        error, str(request["model"]), request_provider, request, optional_str(request.get("api_base"))
    )


def _drop_params(kwargs: Mapping[str, object]) -> bool:
    return bool(litellm.drop_params) or normalize_drop_params(kwargs.get("drop_params")) is True


def _additional_drop_params(kwargs: Mapping[str, object]) -> tuple[str, ...]:
    try:
        configured: Final = _DROP_PATHS.validate_python(kwargs.get("additional_drop_params"))
    except ValidationError:
        return ()
    return tuple(path for path in configured if isinstance(path, str))


def settings(kwargs: Mapping[str, object]) -> dict[str, object]:
    return asdict(
        MessagesSettings(
            drop_params=_drop_params(kwargs),
            reasoning_auto_summary=is_reasoning_auto_summary_enabled(),
            additional_drop_params=_additional_drop_params(kwargs),
            request_timeout=get_configured_request_timeout(),
        )
    )
