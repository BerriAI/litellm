import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Final

import httpx
from openai._streaming import ServerSentEvent, SSEDecoder
from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.openai_like.messages.transformation import (
    DEFAULT_ANTHROPIC_API_VERSION,
    OpenAILikeAnthropicMessagesConfig,
)
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse
from litellm.types.router import GenericLiteLLMParams

from ..common_utils import (
    bigmodel_json_body,
    get_bigmodel_api_key,
    get_bigmodel_url,
    is_unsupported_disabled_thinking,
)

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_OUTPUT_CONFIG: Final = TypeAdapter(dict[str, object])


def _with_lowest_effort_thinking(
    params: Mapping[str, object],
) -> dict[str, object]:  # mutable-ok: the parent transform takes a request dict
    requested_output_config: Final = params.get("output_config")
    output_config: Final = {
        **({} if "reasoning_effort" in params else {"effort": "low"}),
        **(
            _OUTPUT_CONFIG.validate_python(requested_output_config) if isinstance(requested_output_config, dict) else {}
        ),
    }
    return {**params, "thinking": {"type": "adaptive"}, **({"output_config": output_config} if output_config else {})}


def _normalize_message(body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:  # mutable-ok: native JSON response
    usage: Final = body.get("usage")
    if not isinstance(usage, dict):
        return dict(body)
    server_tools: Final = usage.get("server_tool_use")
    if not isinstance(server_tools, dict) or "web_fetch_requests" in server_tools:
        return dict(body)
    normalized: Final[JsonValue] = {
        **body,
        "usage": {**usage, "server_tool_use": {"web_fetch_requests": 0, **server_tools}},
    }
    return normalized


class _BigModelMessagesStream(httpx.AsyncByteStream):
    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for event in SSEDecoder().aiter_bytes(self._response.aiter_bytes()):
            yield _normalize_event(event)

    async def aclose(self) -> None:
        await self._response.aclose()


def _json_object(data: str) -> Mapping[str, JsonValue] | None:
    try:
        return _JSON_OBJECT.validate_json(data)
    except ValidationError:
        return None


def _normalized_data(data: str) -> str:
    body: Final = _json_object(data)
    if body is None:
        return data
    message: Final = body.get("message")
    normalized: Final[JsonValue] = (
        {**body, "message": _normalize_message(message)}
        if body.get("type") == "message_start" and isinstance(message, dict)
        else _normalize_message(body)
    )
    return data if normalized == body else json.dumps(normalized, ensure_ascii=False)


def _normalize_event(event: ServerSentEvent) -> bytes:
    metadata: Final = "".join(
        f"{name}: {value}\n"
        for name, value in (("event", event.event), ("id", event.id), ("retry", event.retry))
        if value is not None
    )
    data_lines: Final = "".join(f"data: {line}\n" for line in _normalized_data(event.data).split("\n"))
    return f"{metadata}{data_lines}\n".encode()


class BigModelMessagesConfig(OpenAILikeAnthropicMessagesConfig):
    @property
    def custom_llm_provider(self) -> str:
        return "bigmodel"

    def is_native_messages_route(self) -> bool:
        return True

    def should_strip_billing_metadata(self) -> bool:
        return True

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: list[dict[str, object]],  # mutable-ok: matches dict-typed base signature
        anthropic_messages_optional_request_params: dict[str, object],  # mutable-ok: matches dict-typed base signature
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: matches dict-typed base signature
    ) -> dict[str, object]:  # mutable-ok: matches dict-typed base signature
        request_params: Final = (
            _with_lowest_effort_thinking(anthropic_messages_optional_request_params)
            if is_unsupported_disabled_thinking(model, anthropic_messages_optional_request_params.get("thinking"))
            else anthropic_messages_optional_request_params
        )
        return super().transform_anthropic_messages_request(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the parent transform returns a bare dict
            model, messages, request_params, litellm_params, headers
        )

    def transform_anthropic_messages_response(
        self, model: str, raw_response: httpx.Response, logging_obj: Logging
    ) -> AnthropicMessagesResponse:
        normalized: Final = httpx.Response(
            raw_response.status_code,
            json=_normalize_message(bigmodel_json_body(raw_response)),
            request=raw_response.request,
        )
        return super().transform_anthropic_messages_response(model, normalized, logging_obj)

    def get_async_streaming_response_iterator(
        self,
        model: str,
        httpx_response: httpx.Response,
        request_body: dict[str, object],  # mutable-ok: inherited native stream API takes a request dict
        litellm_logging_obj: Logging,
    ) -> AsyncIterator[bytes]:
        normalized: Final = httpx.Response(
            httpx_response.status_code,
            headers={
                key: value
                for key, value in httpx_response.headers.items()
                if key.lower() not in ("content-encoding", "content-length")
            },
            stream=_BigModelMessagesStream(httpx_response),
            request=httpx_response.request,
        )
        return super().get_async_streaming_response_iterator(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # native parent yields bytes but its iterator annotation omits the element type
            model, normalized, request_body, litellm_logging_obj
        )

    @staticmethod
    def _translate_reasoning_effort_to_anthropic(
        model: str, optional_params: Mapping[str, object], max_tokens: int | None, custom_llm_provider: str
    ) -> None:
        return None

    @staticmethod
    def _translate_adaptive_effort_for_non_adaptive_model(
        model: str, optional_params: Mapping[str, object], max_tokens: int | None, custom_llm_provider: str
    ) -> None:
        return None

    @staticmethod
    def _drop_incompatible_temperature_for_thinking(
        model: str, optional_params: Mapping[str, object], custom_llm_provider: str
    ) -> None:
        return None

    def validate_anthropic_messages_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict[str, str], str | None]:  # mutable-ok: HTTP handler merges forwarded headers
        resolved_key: Final = get_bigmodel_api_key(api_key)
        if resolved_key is None:
            raise AuthenticationError(
                message="Set BIGMODEL_API_KEY or pass api_key", llm_provider="bigmodel", model=model
            )
        return {
            **{key: value for key, value in headers.items() if key.lower() not in ("x-api-key", "authorization")},
            "x-api-key": resolved_key,
            "anthropic-version": DEFAULT_ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }, api_base

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        return get_bigmodel_url(api_base, "messages")
