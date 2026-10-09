from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Final, cast  # noqa: TID251  # narrows the normalized native payload to the public TypedDict

import httpx
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.core_helpers import normalize_drop_params
from litellm.llms.anthropic.pass_through.utils import is_reasoning_auto_summary_enabled
from litellm.rust_bridge import failures
from litellm.rust_bridge.public_call import optional_sequence, optional_str
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse

_DROP_PATHS: Final = TypeAdapter(list[object])


@dataclass(frozen=True, slots=True)
class MessagesSettings:
    drop_params: bool
    reasoning_auto_summary: bool
    additional_drop_params: Sequence[str]


def _listed(value: object) -> tuple[object, ...]:
    return tuple(optional_sequence(value) or ())


def decline_reason(request: Mapping[str, object]) -> str | None:
    """A streaming call whose callbacks run the agentic loop stays on the Python handler, which runs it at end of stream."""
    if not request.get("stream"):
        return None
    from litellm.llms.custom_httpx.llm_http_handler import overrides_agentic_loop_gate, resolve_custom_loggers

    registered: Final = (
        *litellm.callbacks,
        *_listed(request.get("callbacks")),
        *_listed(request.get("success_callback")),
    )
    if overrides_agentic_loop_gate(resolve_custom_loggers(registered)):
        return "native Messages streaming with an agentic loop hook"
    return None


def response(value: Mapping[str, object]) -> AnthropicMessagesResponse:
    return cast(  # cast-ok: AnthropicMessagesResponse is a TypedDict over the normalized native payload
        AnthropicMessagesResponse,
        dict(value),
    )


def stream_hidden_params(headers: Sequence[tuple[str, str]]) -> Mapping[str, object]:
    from litellm.llms.anthropic.pass_through.messages.streaming_iterator import (
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
        )
    )
