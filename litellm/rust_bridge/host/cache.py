from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from litellm.types.utils import CallTypes

if TYPE_CHECKING:
    from litellm.caching.caching import Cache
    from litellm.caching.dual_cache import DualCache


class ResponseOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["Response"] = "Response"
    value: object


class StreamOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["Stream"] = "Stream"
    value: str


CachedOutput: TypeAlias = ResponseOutput | StreamOutput
_OUTPUT: Final = TypeAdapter[CachedOutput](Annotated[CachedOutput, Field(discriminator="kind")])
_ENTRY: Final = TypeAdapter(dict[str, object])
_EVENTS: Final = TypeAdapter(list[str])
_SSE_EVENT_BOUNDARY: Final = re.compile(r"(?<=\n\n)")


@dataclass(frozen=True, slots=True)
class EntryFormat:
    entry: Callable[[CachedOutput], object | None]
    output: Callable[[object, bool], CachedOutput | None]
    tiers_responses: bool
    tiers_streams: bool

    def tiered(self, output: CachedOutput) -> bool:
        match output:
            case ResponseOutput():
                return self.tiers_responses
            case StreamOutput():
                return self.tiers_streams


def _response_entry(output: CachedOutput) -> object | None:
    match output:
        case ResponseOutput(value=response):
            return response
        case StreamOutput():
            return None


def _response_output(entry: object, stream: bool) -> CachedOutput | None:
    return None if stream else ResponseOutput(value=entry)


def _messages_entry(output: CachedOutput) -> object:
    from litellm.types.caching import CACHED_STREAM_EVENTS_KEY

    match output:
        case ResponseOutput(value=response):
            return response
        case StreamOutput(value=text):
            return {CACHED_STREAM_EVENTS_KEY: [event for event in _SSE_EVENT_BOUNDARY.split(text) if event]}


def _stream_events(entry: object) -> list[str] | None:
    from litellm.types.caching import CACHED_STREAM_EVENTS_KEY

    if not isinstance(entry, dict):
        return None
    events: Final = _ENTRY.validate_python(entry).get(CACHED_STREAM_EVENTS_KEY)
    return None if events is None else _EVENTS.validate_python(events)


def _messages_output(entry: object, stream: bool) -> CachedOutput:
    events: Final = _stream_events(entry)
    if events is None:
        return ResponseOutput(value=entry)
    return StreamOutput(value="".join(events))


RESPONSE_ENTRY: Final = EntryFormat(
    entry=_response_entry, output=_response_output, tiers_responses=True, tiers_streams=False
)
MESSAGES_ENTRY: Final = EntryFormat(
    entry=_messages_entry, output=_messages_output, tiers_responses=False, tiers_streams=True
)


def entry_format(call_type: CallTypes) -> EntryFormat:
    match call_type:
        case CallTypes.anthropic_messages:
            return MESSAGES_ENTRY
        case _:
            return RESPONSE_ENTRY


@dataclass(frozen=True, slots=True)
class CacheAccess:
    cache: Cache
    request: Mapping[str, object]
    key: object
    read: bool
    write: bool
    memory_tier: DualCache | None
    format: EntryFormat


def _allowed(kwargs: Mapping[str, object], call_type: CallTypes) -> tuple[bool, bool]:
    from litellm._logging import verbose_logger
    from litellm.caching.caching_handler import cache_access

    model: Final = kwargs.get("model")
    try:
        return cache_access(model if isinstance(model, str) else None, kwargs, call_type)
    except Exception:  # noqa: BLE001  # a cache failure is a miss, never a request failure
        verbose_logger.exception("LiteLLM Cache: deciding cache access failed")
        return (False, False)


def _key(cache: Cache, kwargs: Mapping[str, object]) -> object:
    from litellm._logging import verbose_logger
    from litellm.caching.caching_handler import cache_key_kwargs, response_cache_key

    try:
        if cache.should_use_cache(**cache_key_kwargs(kwargs)) is not True:
            return None
        return response_cache_key(cache, kwargs)
    except Exception:  # noqa: BLE001  # a cache failure is a miss, never a request failure
        verbose_logger.exception("LiteLLM Cache: preparing the cache key failed")
        return None


def grant(cache: Cache, kwargs: Mapping[str, object], call_type: CallTypes) -> CacheAccess | None:
    from litellm.caching.caching_handler import cache_key_kwargs, response_memory_tier

    read, write = _allowed(kwargs, call_type)
    if not (read or write):
        return None
    key: Final = _key(cache, kwargs)
    if key is None:
        return None
    return CacheAccess(
        cache=cache,
        request=cache_key_kwargs(kwargs),
        key=key,
        read=read,
        write=write,
        memory_tier=response_memory_tier(cache),
        format=entry_format(call_type),
    )


def access(kwargs: Mapping[str, object], call_type: str) -> CacheAccess | None:
    import litellm

    cache: Final = litellm.cache
    return None if cache is None else grant(cache, kwargs, CallTypes(call_type))


@dataclass(frozen=True, slots=True)
class CacheHit:
    response: Mapping[str, object]
    key: str


def _hit(access: CacheAccess, entry: object) -> CacheHit | None:
    output: Final = access.format.output(entry, access.request.get("stream") is True)
    return None if output is None else CacheHit(response=output.model_dump(), key=str(access.key))


def lookup(access: CacheAccess) -> CacheHit | None:
    from litellm.caching.caching_handler import is_response_without_output

    result: Final = access.cache.get_cache(dynamic_cache_object=None, cache_key=access.key, **access.request)  # pyright: ignore[reportUnknownMemberType]  # Cache.get_cache kwargs are untyped
    if result is None or is_response_without_output(result):
        return None
    return _hit(access, result)


async def alookup(access: CacheAccess) -> CacheHit | None:
    from litellm.caching.caching_handler import is_response_without_output

    result: Final = await access.cache.async_get_cache(  # pyright: ignore[reportUnknownMemberType]  # Cache.async_get_cache kwargs are untyped
        dynamic_cache_object=access.memory_tier, cache_key=access.key, **access.request
    )
    if result is None:
        return None
    if is_response_without_output(result):
        if access.memory_tier is not None and isinstance(access.key, str):
            access.memory_tier.in_memory_cache.delete_cache(access.key)
        return None
    return _hit(access, result)


def _entry(access: CacheAccess, value: object) -> tuple[object, bool] | None:
    from litellm.caching.caching_handler import is_response_without_output

    output: Final = _OUTPUT.validate_python(value)
    entry: Final = access.format.entry(output)
    if entry is None or is_response_without_output(entry):
        return None
    return entry, access.format.tiered(output)


def store(access: CacheAccess, value: object) -> None:
    prepared: Final = _entry(access, value)
    if prepared is not None:
        access.cache.add_cache(prepared[0], cache_key=access.key, **access.request)


async def astore(access: CacheAccess, value: object) -> None:
    from litellm.caching.caching_handler import create_cache_write_task

    prepared: Final = _entry(access, value)
    if prepared is None:
        return
    entry, tiered = prepared
    tier: Final = access.memory_tier if tiered else None
    create_cache_write_task(
        lambda: access.cache.async_add_cache(entry, dynamic_cache_object=tier, cache_key=access.key, **access.request)
    )
