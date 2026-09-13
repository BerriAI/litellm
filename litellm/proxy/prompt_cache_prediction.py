from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.prompt_cache_prediction import PromptCachePlan, observe


class _CacheObservationLogger(CustomLogger):
    async def async_log_success_event(
        self, kwargs: dict, response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await _record(kwargs)

    def log_success_event(self, kwargs: dict, response_obj: object, start_time: datetime, end_time: datetime) -> None:
        _record_sync(kwargs)


class _CacheEventMetadata(BaseModel):
    usage_object: Mapping[str, object] | None = None


class _CacheEvent(BaseModel):
    status: str
    custom_llm_provider: str | None = None
    cache_hit: bool | None = None
    model_id: str | None = None
    startTime: float
    metadata: _CacheEventMetadata


def _event(kwargs: Mapping[str, object]) -> tuple[_CacheEvent, PromptCachePlan, str, str] | None:
    plan: Final = kwargs.get("prompt_cache_plan")
    scope: Final = kwargs.get("prompt_cache_observation_scope")
    deployment_id: Final = kwargs.get("prompt_cache_observation_model_id")
    payload: Final = kwargs.get("standard_logging_object")
    if not isinstance(plan, PromptCachePlan) or not isinstance(scope, str) or not isinstance(deployment_id, str):
        return None
    try:
        event: Final = _CacheEvent.model_validate(payload)
    except ValidationError:
        return None
    if (
        event.status != "success"
        or event.custom_llm_provider != "anthropic"
        or event.cache_hit is True
        or event.metadata.usage_object is None
    ):
        return None
    return event, plan, scope, deployment_id


async def _record(kwargs: Mapping[str, object]) -> None:
    parsed: Final = _event(kwargs)
    if parsed is None:
        return
    event, plan, scope, deployment_id = parsed
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        return
    try:
        await observe(
            llm_router.cache,
            scope,
            deployment_id,
            plan,
            event.metadata.usage_object or MappingProxyType({}),
            event.startTime,
        )
    except Exception:  # noqa: BLE001  # optional cache telemetry must not interrupt success callbacks
        verbose_proxy_logger.debug("Could not record prompt cache observation")


def _record_sync(kwargs: Mapping[str, object]) -> None:
    import asyncio

    try:
        asyncio.run(_record(kwargs))
    except RuntimeError:
        verbose_proxy_logger.debug("Could not record synchronous prompt cache observation")


def prompt_cache_observation_logger() -> CustomLogger:
    return _CacheObservationLogger()
