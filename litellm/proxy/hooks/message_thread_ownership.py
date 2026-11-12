import json
import re
from collections.abc import AsyncGenerator, AsyncIterable, Mapping
from typing import TYPE_CHECKING, Final, Protocol

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm._internal_context import with_service_target
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.anthropic.common_utils import merge_anthropic_beta_headers
from litellm.llms.anthropic.messages.native.utils import continued_message_id, message_thread_not_found_body
from litellm.llms.custom_httpx.sse_lines import split_sse_lines_keepends
from litellm.types.utils import CallTypesLiteral, LLMResponseTypes

if TYPE_CHECKING:
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import InternalUsageCache

_PASS_THROUGH_CALL_TYPE: Final = "pass_through_endpoint"
_ANTHROPIC_BETA_HEADER: Final = "anthropic-beta"
_OWNER_KEY_PREFIX: Final = "anthropic_message_thread_owner:"
_OWNER_TTL_SECONDS: Final = 3600
_SSE_DATA_FIELD: Final = "data:"
_SSE_FRAME_END_PATTERN: Final = re.compile(rb"\r\n\r\n|\r\r|\n\n")
_MAX_SCANNED_BYTES: Final = 65536
_UNOWNED_THREAD_MESSAGE: Final = (
    "message thread not found for this key. Send the full conversation to start a new thread"
)

_MAPPING_ADAPTER: Final = TypeAdapter(Mapping[str, object])
_MAPPINGS_ADAPTER: Final = TypeAdapter(tuple[Mapping[str, object], ...])


class _FirstClaimStore(Protocol):
    async def async_set_cache(self, key: str, value: object, *, nx: bool, ttl: int) -> object: ...

    async def async_get_cache(self, key: str) -> object: ...


class _ThreadBinding(BaseModel):
    model_config = ConfigDict(frozen=True)

    owner: str
    anthropic_beta: str | None = None


_BINDING_ADAPTER: Final = TypeAdapter(_ThreadBinding)


def _as_mapping(value: object) -> Mapping[str, object] | None:
    try:
        return _MAPPING_ADAPTER.validate_python(value, strict=True)
    except ValidationError:
        return None


def _as_binding(value: object) -> _ThreadBinding | None:
    try:
        return _BINDING_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def _anthropic_beta(headers: Mapping[str, object]) -> str | None:
    return next(
        (
            value
            for name, value in headers.items()
            if name.lower() == _ANTHROPIC_BETA_HEADER and isinstance(value, str) and value
        ),
        None,
    )


def _sent_anthropic_beta(request_data: Mapping[str, object]) -> str | None:
    logging_obj: Final = request_data.get("litellm_logging_obj")
    if not isinstance(logging_obj, Logging):
        return None
    response: Final[object] = logging_obj.model_call_details.get("httpx_response")
    if not isinstance(response, httpx.Response):
        return None
    try:
        sent_headers: Final = response.request.headers
    except RuntimeError:
        return None
    return sent_headers.get(_ANTHROPIC_BETA_HEADER) or None


def _with_anthropic_beta(extra_headers: object, anthropic_beta: str) -> Mapping[str, object]:
    headers: Final = _as_mapping(extra_headers) or {}
    return {
        **{name: value for name, value in headers.items() if name.lower() != _ANTHROPIC_BETA_HEADER},
        _ANTHROPIC_BETA_HEADER: merge_anthropic_beta_headers(_anthropic_beta(headers), anthropic_beta),
    }


def _as_mappings(value: object) -> tuple[Mapping[str, object], ...]:
    try:
        return _MAPPINGS_ADAPTER.validate_python(value)
    except ValidationError:
        return ()


def _client_headers_with_anthropic_beta(scoped: Mapping[str, object], anthropic_beta: str) -> Mapping[str, object]:
    client_headers: Final = _as_mapping(scoped.get("extra_headers"))
    if client_headers is None or _anthropic_beta(client_headers) is None:
        return scoped
    return {**scoped, "extra_headers": dict(_with_anthropic_beta(client_headers, anthropic_beta))}


def _provider_headers_with_anthropic_beta(
    provider_specific_header: object, anthropic_beta: str
) -> Mapping[str, object]:
    single: Final = _as_mapping(provider_specific_header)
    if single is not None:
        return {"provider_specific_header": _client_headers_with_anthropic_beta(single, anthropic_beta)}
    scoped_headers: Final = _as_mappings(provider_specific_header)
    if not scoped_headers:
        return {}
    return {
        "provider_specific_header": tuple(
            _client_headers_with_anthropic_beta(scoped, anthropic_beta) for scoped in scoped_headers
        )
    }


def _owner_key(message_id: str) -> str:
    return f"{_OWNER_KEY_PREFIX}{message_id}"


def _owner(user_api_key_dict: "UserAPIKeyAuth") -> str:
    if user_api_key_dict.token:
        return f"key:{user_api_key_dict.token}"
    return "identity:" + json.dumps(
        [
            user_api_key_dict.user_id,
            user_api_key_dict.team_id,
            user_api_key_dict.end_user_id,
            user_api_key_dict.org_id,
            user_api_key_dict.agent_id,
        ]
    )


def _threads(data: Mapping[str, object]) -> tuple[object, object]:
    extra_body: Final = _as_mapping(data.get("extra_body")) or {}
    return data.get("thread"), extra_body.get("thread")


def _sends_thread(data: Mapping[str, object]) -> bool:
    return any(thread is not None for thread in _threads(data))


def _continued_message_ids(data: Mapping[str, object]) -> tuple[str, ...]:
    continued: Final = (continued_message_id(thread) for thread in _threads(data))
    return tuple(message_id for message_id in continued if message_id is not None)


def _may_continue(binding: _ThreadBinding | None, owner: str, pass_through: bool) -> bool:
    if binding is None:
        return pass_through
    return binding.owner == owner


def _response_message_id(response: object) -> str | None:
    message: Final = _as_mapping(response)
    if message is None or message.get("type") != "message":
        return None
    message_id: Final = message.get("id")
    return message_id if isinstance(message_id, str) and message_id else None


def _message_start_id(event: Mapping[str, object]) -> str | None:
    return _response_message_id(event.get("message")) if event.get("type") == "message_start" else None


def _frame_event(frame: bytes) -> Mapping[str, object] | None:
    data_lines: Final = tuple(
        line.strip()[len(_SSE_DATA_FIELD) :].strip()
        for line in split_sse_lines_keepends(frame.decode("utf-8", errors="ignore"))
        if line.strip().startswith(_SSE_DATA_FIELD)
    )
    payload: Final = "".join(data_lines)
    if not payload or payload == "[DONE]":
        return None
    try:
        return _MAPPING_ADAPTER.validate_json(payload)
    except ValidationError:
        return None


class _StreamedMessageIdScanner:
    def __init__(self, finished: bool) -> None:
        self._head = b""
        self.finished = finished

    def scan(self, chunk: object) -> str | None:
        if isinstance(chunk, (bytes, bytearray)):
            return self._scan_encoded(bytes(chunk))
        if isinstance(chunk, str):
            return self._scan_encoded(chunk.encode("utf-8"))
        event: Final = _as_mapping(chunk)
        if event is None:
            self.finished = True
            return None
        return self._settled(event)

    def _settled(self, event: Mapping[str, object]) -> str | None:
        if event.get("type") == "ping":
            return None
        self.finished = True
        return _message_start_id(event)

    def _scan_encoded(self, data: bytes) -> str | None:
        head: Final = self._head + data
        boundaries: Final = tuple(match.end() for match in _SSE_FRAME_END_PATTERN.finditer(head))
        frames: Final = tuple(head[start:end] for start, end in zip((0, *boundaries[:-1]), boundaries))
        events: Final = (event for frame in frames if (event := _frame_event(frame)) is not None)
        decider: Final = next((event for event in events if event.get("type") != "ping"), None)
        if decider is not None:
            return self._settled(decider)
        self._head = head[boundaries[-1] :] if boundaries else head
        if len(self._head) > _MAX_SCANNED_BYTES:
            self.finished = True
        return None


class MessageThreadOwnership(CustomLogger):
    def __init__(self, message_thread_owner_cache: "InternalUsageCache") -> None:
        self._owners: Final = message_thread_owner_cache

    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: "DualCache",
        data: Mapping[str, object],
        call_type: CallTypesLiteral,
    ) -> dict[str, object] | None:  # mutable-ok: the proxy swaps in a pre-call hook's result only when it is a dict
        previous_message_ids: Final = _continued_message_ids(data)
        if not previous_message_ids:
            return None
        bindings: Final = tuple(
            [await self._binding(message_id, user_api_key_dict) for message_id in previous_message_ids]
        )
        owner: Final = _owner(user_api_key_dict)
        pass_through: Final = call_type == _PASS_THROUGH_CALL_TYPE
        if not all(_may_continue(binding, owner, pass_through) for binding in bindings):
            raise HTTPException(status_code=404, detail=message_thread_not_found_body(_UNOWNED_THREAD_MESSAGE))
        anthropic_beta: Final = next(
            (binding.anthropic_beta for binding in bindings if binding is not None and binding.anthropic_beta), None
        )
        if anthropic_beta is None or pass_through:
            return None
        return {
            **data,
            "extra_headers": _with_anthropic_beta(data.get("extra_headers"), anthropic_beta),
            **_provider_headers_with_anthropic_beta(data.get("provider_specific_header"), anthropic_beta),
        }

    async def async_post_call_success_hook(
        self,
        data: Mapping[str, object],
        user_api_key_dict: "UserAPIKeyAuth",
        response: LLMResponseTypes,
    ) -> LLMResponseTypes:
        message_id: Final = _response_message_id(response) if _sends_thread(data) else None
        if message_id is not None:
            await self._record_owner(message_id, user_api_key_dict, _sent_anthropic_beta(data))
        return response

    async def async_post_call_streaming_iterator_hook(  # pyright: ignore[reportIncompatibleMethodOverride]  # the proxy streams bytes, str and dict chunks here, not the base's ModelResponseStream
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        response: AsyncIterable[object],
        request_data: Mapping[str, object],
    ) -> AsyncGenerator[object, None]:
        scanner: Final = _StreamedMessageIdScanner(finished=not _sends_thread(request_data))
        async for chunk in response:
            yield chunk
            if not scanner.finished and (message_id := scanner.scan(chunk)) is not None:
                await self._record_owner(message_id, user_api_key_dict, _sent_anthropic_beta(request_data))

    @with_service_target("message_thread_owners")
    async def _binding(self, message_id: str, user_api_key_dict: "UserAPIKeyAuth") -> _ThreadBinding | None:
        cached: Final[object] = await self._owners.async_get_cache(
            key=_owner_key(message_id), litellm_parent_otel_span=user_api_key_dict.parent_otel_span
        )
        return _as_binding(cached)

    @with_service_target("message_thread_owners")
    async def _record_owner(
        self, message_id: str, user_api_key_dict: "UserAPIKeyAuth", anthropic_beta: str | None
    ) -> None:
        if await self._binding(message_id, user_api_key_dict) is not None:
            return
        key: Final = _owner_key(message_id)
        claimed: Final = _ThreadBinding(owner=_owner(user_api_key_dict), anthropic_beta=anthropic_beta).model_dump()
        await self._owners.async_set_cache(
            key=key,
            value=await _first_claim(self._owners.dual_cache.redis_cache, key, claimed),
            litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
            local_only=True,
            ttl=_OWNER_TTL_SECONDS,
        )


async def _first_claim(store: _FirstClaimStore | None, key: str, claimed: Mapping[str, object]) -> object:
    if store is None:
        return claimed
    try:
        await store.async_set_cache(key, claimed, nx=True, ttl=_OWNER_TTL_SECONDS)
        stored: Final = await store.async_get_cache(key)
    except Exception:  # noqa: BLE001  # a Redis outage keeps the binding in this worker, as DualCache's writes do
        return claimed
    return claimed if stored is None else stored
