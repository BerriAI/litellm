from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from types import MappingProxyType
from typing import Final, Literal

from pydantic import BaseModel, TypeAdapter

import litellm
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.rust_bridge import runtime
from litellm.rust_bridge.catalog import Route, RouteContext, RouteRule
from litellm.rust_bridge.chat_completions.entrypoints import NATIVE_ACOMPLETION
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.messages.entrypoints import NATIVE_AMESSAGES
from litellm.rust_bridge.public_call import NativeCall
from litellm.rust_bridge.responses.entrypoints import NATIVE_ARESPONSES
from litellm.types.utils import ModelResponse
from tests.test_litellm_rust.messages.data import (
    MESSAGES,
    MESSAGES_MODEL,
    MESSAGES_SSE,
    messages_reply,
)
from tests.test_litellm_rust.responses.data import RESPONSES_MODEL, responses_reply
from tests._support.provider import SharedProvider
from tests._support.wire import Reply


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


def cache_reply(route: Literal["chat", "messages", "responses"], stream: bool = False) -> Reply:
    if route == "responses":
        return responses_reply()
    if route == "messages" and stream:
        return Reply(chunks=MESSAGES_SSE, content_type="text/event-stream")
    return messages_reply()


async def invoke(
    route: Literal["chat", "messages", "responses"],
    server: SharedProvider,
    options: Mapping[str, object],
    native: bool = True,
) -> object:
    common: Final = {"api_key": "test-key", "api_base": server.wire.url, **options}
    if route == "responses":
        arguments: Final = {"model": RESPONSES_MODEL, "input": "hello", **common}
        if not native:
            return await litellm.aresponses(**arguments)
        request: Final = NativeCall(
            args=(),
            kwargs=arguments,
            base={
                "model": RESPONSES_MODEL,
                "input": "hello",
                "stream": None,
                "api_key": "test-key",
                "api_base": server.wire.url,
                "custom_llm_provider": "openai",
                "extra_headers": None,
                **arguments,
            },
        )
        return await runtime.arun(
            RouteContext(Route.RESPONSES),
            binding=NATIVE_ARESPONSES,
            native=lambda hook: hook(request),
            python=runtime.NO_PYTHON,
            rules=(RouteRule(Route.RESPONSES, Rollout.RUST_REQUIRED),),
        )
    parameters: Final = {"model": MESSAGES_MODEL, "messages": list(MESSAGES), "max_tokens": 32, **common}
    if route == "chat":
        if not native:
            return await litellm.acompletion(**parameters)
        chat: Final = NativeCall(args=(), kwargs=parameters, base={})
        return await runtime.arun(
            RouteContext(Route.CHAT_COMPLETIONS),
            binding=NATIVE_ACOMPLETION,
            native=lambda hook: hook(chat),
            python=runtime.NO_PYTHON,
            rules=(RouteRule(Route.CHAT_COMPLETIONS, Rollout.RUST_REQUIRED),),
        )
    if not native:
        return await litellm.anthropic_messages(**parameters)
    messages: Final = NativeCall(
        args=(),
        kwargs=parameters,
        base={
            "model": MESSAGES_MODEL,
            "messages": list(MESSAGES),
            "max_tokens": 32,
            "stream": None,
            "api_key": "test-key",
            "api_base": server.wire.url,
            "custom_llm_provider": "anthropic",
            **parameters,
        },
    )
    return await runtime.arun(
        RouteContext(Route.MESSAGES),
        binding=NATIVE_AMESSAGES,
        native=lambda hook: hook(messages),
        python=runtime.NO_PYTHON,
        rules=(RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),),
    )


async def collect(stream: object) -> bytes:
    assert isinstance(stream, AsyncIterator)
    return b"".join([chunk_bytes(chunk) async for chunk in stream])


def chunk_bytes(value: object) -> bytes:
    assert isinstance(value, bytes)
    return value
