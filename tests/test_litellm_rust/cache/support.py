from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from types import MappingProxyType
from typing import Final, Literal

from pydantic import BaseModel, TypeAdapter

from litellm.chat_completions.dispatch import ACOMPLETION
from litellm.messages.dispatch import ANTHROPIC_MESSAGES
from litellm.responses.dispatch import ARESPONSES
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.rust_bridge.catalog import Route
from litellm.types.utils import ModelResponse
from tests.test_litellm_rust.messages.data import MESSAGES, MESSAGES_EVENTS, MESSAGES_MODEL, MESSAGES_RESPONSE
from tests.test_litellm_rust.responses.data import RESPONSES_MODEL, RESPONSES_RESPONSE
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.routing import python_only, rust_required


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
) -> object:
    common: Final = {"api_key": "test-key", "api_base": server.base_url, **options}
    rules: Final = rust_required if native else python_only
    if route == "responses":
        server.default_response = ResponseSpec(body=RESPONSES_RESPONSE)
        arguments: Final = {"model": RESPONSES_MODEL, "input": "hello", **common}
        return await ARESPONSES.arun((), arguments, rules=rules(Route.RESPONSES))
    server.default_response = (
        ResponseSpec(body=None, events=MESSAGES_EVENTS)
        if options.get("stream")
        else ResponseSpec(body=MESSAGES_RESPONSE)
    )
    parameters: Final = {"model": MESSAGES_MODEL, "messages": list(MESSAGES), "max_tokens": 32, **common}
    if route == "chat":
        return await ACOMPLETION.arun((), parameters, rules=rules(Route.CHAT_COMPLETIONS))
    return await ANTHROPIC_MESSAGES.arun((), parameters, rules=rules(Route.MESSAGES))


async def collect(stream: object) -> bytes:
    assert isinstance(stream, AsyncIterator)
    return b"".join([chunk_bytes(chunk) async for chunk in stream])


def chunk_bytes(value: object) -> bytes:
    assert isinstance(value, bytes)
    return value
