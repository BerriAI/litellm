from collections.abc import AsyncIterator, Mapping
from types import MappingProxyType
from typing import Final, Literal, Protocol

from pydantic import BaseModel, TypeAdapter

import litellm
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.rust_bridge import runtime
from litellm.rust_bridge.catalog import Route, RouteContext, RouteRule
from litellm.rust_bridge.chat_completions.entrypoints import NATIVE_ACOMPLETION, LiteLLMChatCompletionsRequest
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.dispatch import call_hook
from litellm.rust_bridge.messages.entrypoints import NATIVE_AMESSAGES, LiteLLMMessagesRequest
from litellm.rust_bridge.responses.entrypoints import NATIVE_ARESPONSES, LiteLLMResponsesRequest
from litellm.types.utils import ModelResponse
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import MESSAGES, MESSAGES_EVENTS, MESSAGES_MODEL, MESSAGES_RESPONSE
from tests.test_litellm_rust.test_inference import RESPONSES_MODEL, RESPONSES_RESPONSE


def payload(value: object) -> object:
    if isinstance(value, ModelResponse):
        return value.model_dump_json(exclude=MappingProxyType({"id": True, "created": True}))
    if isinstance(value, dict):
        fields: Final = TypeAdapter(dict[str, object]).validate_python(value)
        return {name: field for name, field in fields.items() if name != "_hidden_params"}
    return value.model_dump_json() if isinstance(value, BaseModel) else value


def cache_key(response: object) -> object:
    hidden: Final = get_hidden_params_dict(response)
    headers: Final = TypeAdapter(dict[str, object]).validate_python(hidden.get("additional_headers", {}))
    return headers.get("x-litellm-cache-key")


async def invoke(
    route: Literal["chat", "messages", "responses"],
    server: RecordingServer,
    options: Mapping[str, object],
    native: bool = True,
    events: tuple[tuple[str, object], ...] | None = None,
    ensure_ascii: bool = True,
) -> object:
    common: Final = {"api_key": "test-key", "api_base": server.base_url, **options}
    if route == "responses":
        server.default_response = ResponseSpec(body=RESPONSES_RESPONSE)
        arguments: Final = {"model": RESPONSES_MODEL, "input": "hello", **common}
        if not native:
            return await litellm.aresponses(**arguments)
        request: Final = LiteLLMResponsesRequest(
            RESPONSES_MODEL, "hello", None, "test-key", server.base_url, "openai", None, arguments
        )
        return await runtime.arun(
            RouteContext(Route.RESPONSES),
            binding=NATIVE_ARESPONSES,
            native=lambda hook: call_hook(hook, request, (), arguments),
            python=runtime.NO_PYTHON,
            rules=(RouteRule(Route.RESPONSES, Rollout.RUST_REQUIRED),),
        )
    server.default_response = (
        ResponseSpec(
            body=None,
            events=MESSAGES_EVENTS if events is None else events,
            ensure_ascii=ensure_ascii,
        )
        if options.get("stream")
        else ResponseSpec(body=MESSAGES_RESPONSE)
    )
    parameters: Final = {"model": MESSAGES_MODEL, "messages": list(MESSAGES), "max_tokens": 32, **common}
    if route == "chat":
        if not native:
            return await litellm.acompletion(**parameters)
        chat: Final = LiteLLMChatCompletionsRequest(
            MESSAGES_MODEL, list(MESSAGES), None, "test-key", server.base_url, None, None, parameters
        )
        return await runtime.arun(
            RouteContext(Route.CHAT_COMPLETIONS),
            binding=NATIVE_ACOMPLETION,
            native=lambda hook: call_hook(hook, chat, (), parameters),
            python=runtime.NO_PYTHON,
            rules=(RouteRule(Route.CHAT_COMPLETIONS, Rollout.RUST_REQUIRED),),
        )
    if not native:
        return await litellm.anthropic_messages(**parameters)
    messages: Final = LiteLLMMessagesRequest(
        MESSAGES_MODEL, list(MESSAGES), 32, None, "test-key", server.base_url, "anthropic", parameters
    )
    return await runtime.arun(
        RouteContext(Route.MESSAGES),
        binding=NATIVE_AMESSAGES,
        native=lambda hook: call_hook(hook, messages, (), parameters),
        python=runtime.NO_PYTHON,
        rules=(RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),),
    )


async def collect(stream: object) -> bytes:
    assert isinstance(stream, AsyncIterator)
    return b"".join([chunk_bytes(chunk) async for chunk in stream])


class ClosableByteStream(Protocol):
    def __aiter__(self) -> AsyncIterator[bytes]: ...

    async def __anext__(self) -> bytes: ...

    async def aclose(self) -> None: ...


async def collect_chunks(stream: object) -> tuple[bytes, ...]:
    assert isinstance(stream, AsyncIterator)
    return tuple([chunk_bytes(chunk) async for chunk in stream])


def chunk_bytes(value: object) -> bytes:
    assert isinstance(value, bytes)
    return value
