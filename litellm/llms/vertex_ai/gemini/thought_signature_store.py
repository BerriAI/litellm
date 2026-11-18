import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from itertools import chain
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import (
    GEMINI_THOUGHT_SIGNATURE_STORE_MAX_SIZE_IN_MEMORY,
    GEMINI_THOUGHT_SIGNATURE_STORE_TTL_SECONDS,
)
from litellm.litellm_core_utils.prompt_templates.factory import get_thought_signature_from_tool
from litellm.types.llms.openai import AllMessageValues, ChatCompletionAssistantToolCall
from litellm.types.utils import Choices, ModelResponse, ModelResponseStream, StreamingChoices

thought_signature_cache: Final = DualCache(
    in_memory_cache=InMemoryCache(
        max_size_in_memory=GEMINI_THOUGHT_SIGNATURE_STORE_MAX_SIZE_IN_MEMORY,
        default_ttl=GEMINI_THOUGHT_SIGNATURE_STORE_TTL_SECONDS,
    ),
    default_in_memory_ttl=GEMINI_THOUGHT_SIGNATURE_STORE_TTL_SECONDS,
    default_redis_ttl=GEMINI_THOUGHT_SIGNATURE_STORE_TTL_SECONDS,
)

_KEY_PREFIX: Final = "gemini_thought_signature:v1:"
_JSON_VALUE: Final = TypeAdapter[JsonValue](JsonValue)


class _CalledFunction(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    name: str
    arguments: str


class _ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    id: str
    function: _CalledFunction


class _SignatureFields(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    thought_signature: str


class _SignedToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    id: str
    function: _CalledFunction
    provider_specific_fields: _SignatureFields


class _StoredSignature(TypedDict):
    thought_signature: ReadOnly[str]


_STORED_SIGNATURE: Final = TypeAdapter(_StoredSignature)


def _canonical_arguments(arguments: str) -> str:
    try:
        parsed: Final = _JSON_VALUE.validate_json(arguments)
    except ValidationError:
        return arguments
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _key(backend: str, call: _ToolCall | _SignedToolCall) -> str:
    material: Final = json.dumps([backend, call.id, call.function.name, _canonical_arguments(call.function.arguments)])
    return _KEY_PREFIX + hashlib.sha256(material.encode()).hexdigest()


def _replayed_key(backend: str, call: ChatCompletionAssistantToolCall) -> str | None:
    try:
        return _key(backend, _ToolCall.model_validate(call))
    except ValidationError:
        return None


def _signed(call: object) -> _SignedToolCall | None:
    try:
        return _SignedToolCall.model_validate(call)
    except ValidationError:
        return None


def _carried_tool_calls(choice: Choices | StreamingChoices) -> Sequence[object]:
    carrier: Final = choice.message if isinstance(choice, Choices) else choice.delta
    return carrier.tool_calls or ()


def _signature_entries(
    backend: str, response: ModelResponse | ModelResponseStream
) -> tuple[tuple[str, _StoredSignature], ...]:
    calls: Final = chain.from_iterable(map(_carried_tool_calls, response.choices))
    signed: Final = (call for call in map(_signed, calls) if call is not None)
    return tuple(
        (_key(backend, call), {"thought_signature": call.provider_specific_fields.thought_signature}) for call in signed
    )


def remember_thought_signatures(backend: str, response: ModelResponse | ModelResponseStream) -> None:
    for key, value in _signature_entries(backend, response):
        thought_signature_cache.set_cache(  # pyright: ignore[reportUnknownMemberType]  # DualCache is untyped
            key, value, ttl=GEMINI_THOUGHT_SIGNATURE_STORE_TTL_SECONDS
        )


async def aremember_thought_signatures(backend: str, response: ModelResponse | ModelResponseStream) -> None:
    entries: Final = _signature_entries(backend, response)
    if not entries:
        return
    await thought_signature_cache.async_set_cache_pipeline(  # pyright: ignore[reportUnknownMemberType]  # DualCache is untyped
        entries, ttl=GEMINI_THOUGHT_SIGNATURE_STORE_TTL_SECONDS
    )


def _assistant_tool_calls(message: AllMessageValues) -> Sequence[ChatCompletionAssistantToolCall]:
    if message["role"] != "assistant":
        return ()
    return message.get("tool_calls") or ()


def _unsigned_tool_calls(messages: Sequence[AllMessageValues]) -> Iterator[ChatCompletionAssistantToolCall]:
    calls: Final = chain.from_iterable(map(_assistant_tool_calls, messages))
    return (call for call in calls if get_thought_signature_from_tool(call) is None)


def _lookup_keys(backend: str, messages: Sequence[AllMessageValues]) -> tuple[str, ...]:
    replayed_keys: Final = (_replayed_key(backend, call) for call in _unsigned_tool_calls(messages))
    keys: Final = (key for key in replayed_keys if key is not None)
    return tuple(dict.fromkeys(keys))


def _stored_signature(value: object) -> str | None:
    try:
        return _STORED_SIGNATURE.validate_python(value)["thought_signature"]
    except ValidationError:
        return None


def _found_signatures(keys: Sequence[str], values: Sequence[object] | None) -> Mapping[str, str]:
    found: Final = ((key, _stored_signature(value)) for key, value in zip(keys, values or ()))
    return MappingProxyType({key: signature for key, signature in found if signature is not None})


def _restored_tool_call(
    backend: str, call: ChatCompletionAssistantToolCall, signatures: Mapping[str, str]
) -> ChatCompletionAssistantToolCall:
    key: Final = _replayed_key(backend, call)
    signature: Final = signatures.get(key) if key is not None else None
    if signature is None:
        return call
    return {**call, "function": {**call["function"], "provider_specific_fields": {"thought_signature": signature}}}


def _restored_message(backend: str, message: AllMessageValues, signatures: Mapping[str, str]) -> AllMessageValues:
    if message["role"] != "assistant":
        return message
    calls: Final = message.get("tool_calls") or ()
    restored: Final = [_restored_tool_call(backend, call, signatures) for call in calls]
    if all(new is old for new, old in zip(restored, calls)):
        return message
    return {**message, "tool_calls": restored}


def _restored_messages(
    backend: str, messages: Sequence[AllMessageValues], signatures: Mapping[str, str]
) -> tuple[AllMessageValues, ...] | None:
    if not signatures:
        return None
    return tuple(_restored_message(backend, message, signatures) for message in messages)


def restore_thought_signatures(
    backend: str,
    messages: list[AllMessageValues],  # mutable-ok: the request transform pops and appends on the list it is given
) -> list[AllMessageValues]:  # mutable-ok: the request transform pops and appends on the list it is given
    keys: Final = _lookup_keys(backend, messages)
    stored: Final = thought_signature_cache.batch_get_cache(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # DualCache is untyped
        list(keys), throttle_redis=False
    )
    restored: Final = _restored_messages(backend, messages, _found_signatures(keys, stored))  # pyright: ignore[reportUnknownArgumentType]  # DualCache is untyped
    return messages if restored is None else list(restored)


async def arestore_thought_signatures(
    backend: str,
    messages: list[AllMessageValues],  # mutable-ok: the request transform pops and appends on the list it is given
) -> list[AllMessageValues]:  # mutable-ok: the request transform pops and appends on the list it is given
    keys: Final = _lookup_keys(backend, messages)
    stored: Final = await thought_signature_cache.async_batch_get_cache(  # pyright: ignore[reportUnknownMemberType]  # DualCache is untyped
        list(keys), throttle_redis=False
    )
    restored: Final = _restored_messages(backend, messages, _found_signatures(keys, stored))
    return messages if restored is None else list(restored)
