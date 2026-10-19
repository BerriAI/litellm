from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Final, cast  # noqa: TID251  # narrows the normalized native payload to the public TypedDict

import httpx
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.core_helpers import normalize_drop_params
from litellm.litellm_core_utils.request_timeout_resolver import get_configured_request_timeout
from litellm.llms.anthropic.pass_through.messages.utils import AnthropicMessagesRequestUtils
from litellm.rust_bridge import failures
from litellm.rust_bridge.host.provider import resolved_provider
from litellm.rust_bridge.public_call import optional_str
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse

_OBJECTS: Final = TypeAdapter(tuple[object, ...])
_STR_MAPPING: Final = TypeAdapter(Mapping[str, object])


@dataclass(frozen=True, slots=True)
class MessagesSettings:
    additional_drop_params: Sequence[str]
    request_timeout: float | None
    cache_control_injection_points: Sequence[Mapping[str, object]]
    enable_prompt_caching: bool
    prompt_caching_ttl: str | None
    user_agent: str | None


def response(value: Mapping[str, object]) -> AnthropicMessagesResponse:
    return cast(  # cast-ok: AnthropicMessagesResponse is a TypedDict over the normalized native payload
        AnthropicMessagesResponse,
        dict(value),
    )


def body_params(request: Mapping[str, object]) -> Mapping[str, object]:
    model, provider = resolved_provider(str(request["model"]), optional_str(request.get("custom_llm_provider")))
    return dict(
        AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
            request, model=model, drop_params=_drop_params(request), custom_llm_provider=provider
        )
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


def _objects(value: object) -> tuple[object, ...]:
    try:
        return _OBJECTS.validate_python(value)
    except ValidationError:
        return ()


def _additional_drop_params(kwargs: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(path for path in _objects(kwargs.get("additional_drop_params")) if isinstance(path, str))


def _str_mapping(value: object) -> Mapping[str, object] | None:
    try:
        return _STR_MAPPING.validate_python(value)
    except ValidationError:
        return None


def _cache_control_injection_points(kwargs: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    points: Final = (_str_mapping(item) for item in _objects(kwargs.get("cache_control_injection_points")))
    return tuple(point for point in points if point is not None)


def _user_agent(kwargs: Mapping[str, object]) -> str | None:
    request: Final = _str_mapping(kwargs.get("proxy_server_request")) or {}
    headers: Final = _str_mapping(request.get("headers")) or {}
    return next(
        (value for key, value in headers.items() if key.lower() == "user-agent" and isinstance(value, str)),
        None,
    )


def _prompt_caching_ttl() -> str | None:
    value: Final = getattr(litellm, "anthropic_prompt_caching_ttl", None)
    return value if value in ("5m", "1h") else None


def settings(kwargs: Mapping[str, object]) -> dict[str, object]:
    return asdict(
        MessagesSettings(
            additional_drop_params=_additional_drop_params(kwargs),
            request_timeout=get_configured_request_timeout(),
            cache_control_injection_points=_cache_control_injection_points(kwargs),
            enable_prompt_caching=kwargs.get("enable_prompt_caching") is True
            or litellm.enable_anthropic_prompt_caching is True,
            prompt_caching_ttl=_prompt_caching_ttl(),
            user_agent=_user_agent(kwargs),
        )
    )
