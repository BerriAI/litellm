from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Final, Literal

import httpx
from pydantic import ConfigDict, Field, ValidationError

from litellm._internal_context import with_service_target
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.anthropic.prompt_cache_prediction import (
    _RETENTION_SECONDS,  # pyright: ignore[reportPrivateUsage]  # moved to the SDK layer, still shared with this hook
    CacheObservation,
    _cache_key,  # pyright: ignore[reportPrivateUsage]  # moved to the SDK layer, still shared with this hook
    _read_exact,  # pyright: ignore[reportPrivateUsage]  # moved to the SDK layer, still shared with this hook
    parse_observed_cache,
)
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.proxy.utils import InternalUsageCache


class _Metadata(LiteLLMBaseModel):
    model_config = ConfigDict(strict=True)
    user_api_key_hash: str = Field(min_length=1)


class _Logged(LiteLLMBaseModel):
    model_config = ConfigDict(strict=True)
    status: Literal["success"]
    model_id: str = Field(min_length=1)
    metadata: _Metadata


class _Event(LiteLLMBaseModel):
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)
    call_type: Literal["anthropic_messages"]
    custom_llm_provider: Literal["anthropic"]
    cache_hit: bool | None = None
    httpx_response: httpx.Response
    first_api_call_start_time: datetime
    standard_logging_object: _Logged
    stream: bool = False
    prompt_cache_response_complete: bool = False


class PromptCacheObserver(CustomLogger):
    def __init__(self, internal_usage_cache: InternalUsageCache, clock: Callable[[], float] = time.time) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # base callback constructor accepts untyped kwargs
        self.cache = internal_usage_cache.dual_cache
        self.clock = clock

    @with_service_target("prompt_cache_predictions")
    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        if not isinstance(response_obj, ModelResponse):
            return
        try:
            event: Final = _Event.model_validate(kwargs)
            wire: Final = event.httpx_response.request
        except (ValidationError, RuntimeError, httpx.RequestNotRead):
            return
        if (
            event.cache_hit
            or event.httpx_response.status_code != 200
            or (event.stream and not event.prompt_cache_response_complete)
        ):
            return
        observed: Final = parse_observed_cache(
            wire,
            response_obj,
            event.standard_logging_object.metadata.user_api_key_hash,
            event.standard_logging_object.model_id,
        )
        if observed is None:
            return
        prefix: Final = observed.prefix
        scope: Final = observed.scope
        cache_tokens: Final = observed.cached_tokens
        now: Final = self.clock()
        started: Final = event.first_api_call_start_time.timestamp()
        if started > now:
            return
        if observed.cache_creation_tokens == 0:
            previous: Final = await _read_exact(self.cache, scope, prefix.fingerprint)
            if previous is None or previous.fingerprint != prefix.fingerprint or previous.cached_tokens != cache_tokens:
                return
        observation: Final = CacheObservation(
            fingerprint=prefix.fingerprint,
            cached_tokens=cache_tokens,
            observed_at=now,
            expires_at=started + prefix.ttl_seconds,
        )
        key: Final = _cache_key(scope, prefix.fingerprint)
        payload: Final = observation.model_dump_json()
        await self.cache.async_set_cache(key, payload, ttl=_RETENTION_SECONDS)  # pyright: ignore[reportUnknownMemberType]  # legacy cache accepts a serialized validated observation
        if self.cache.redis_cache is not None:
            await self.cache.async_set_cache(key, payload, local_only=True, ttl=1)  # pyright: ignore[reportUnknownMemberType]  # keep the local copy short-lived while Redis retains stale evidence
