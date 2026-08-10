"""
Tag-scoped token, request, and dollar rate limits.

Each limit entry is keyed by an arbitrary caller-supplied tag value (not a
DB-provisioned entity, not composed with the calling API key) and enforced on
every routing attempt for a chain/model-group -- the primary hop and every
fallback hop, each checked against its own configuration.

Opt-in via `litellm_settings.callbacks: ["tag_rate_limiter"]` (not part of
`PROXY_HOOKS`), following the `dynamic_rate_limiter_v3` precedent: this hook
reuses `_PROXY_MaxParallelRequestsHandler_v3`'s Redis/TTL-preserving increment
machinery rather than duplicating it, and is never joined onto the default
limiter every proxy already runs.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.exceptions import RateLimitType
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import (
    _get_parent_otel_span_from_kwargs,
    get_metadata_variable_name_from_kwargs,
)
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.router import Router
from litellm.router_strategy.tag_based_routing import _get_tags_from_request_kwargs
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import TagRateLimitEntry, TagRateLimits
from litellm.types.utils import StandardLoggingPayload

_LimitUnit = Literal["tokens", "requests", "dollars"]
_LIMIT_UNITS: tuple[_LimitUnit, ...] = ("tokens", "requests", "dollars")

_UNIT_TO_GROUP_FIELD: dict[_LimitUnit, str] = {
    "tokens": "token_limits",
    "requests": "request_limits",
    "dollars": "dollar_limits",
}
_UNIT_TO_RATE_LIMIT_TYPE: dict[_LimitUnit, RateLimitType] = {
    "tokens": RateLimitType.TOKENS,
    "requests": RateLimitType.REQUESTS,
    "dollars": RateLimitType.BUDGET,
}


@dataclass(frozen=True)
class _ConfiguredLimit:
    unit: _LimitUnit
    entry: TagRateLimitEntry
    # None => chain-wide (every deployment in the model_group shares one
    # bucket). Otherwise the sorted deployment ids that declared this exact
    # value -- the bucket is shared among only those deployments.
    deployment_scope: Optional[tuple[str, ...]]


def _extract_identity(tags: list[str], tag_id: str) -> Optional[str]:
    """
    First tag matching `f"{tag_id}:"`, value after the colon. Tags starting
    with `!` are tag-routing negation markers, not identity tags, and are
    skipped so they can never be misread as an identity value.
    """
    prefix = f"{tag_id}:"
    for tag in tags:
        if tag.startswith("!"):
            continue
        if tag.startswith(prefix):
            return tag[len(prefix) :]
    return None


def _deployment_id(deployment: dict) -> Optional[str]:
    return (deployment.get("model_info") or {}).get("id")


def _entries_for_unit(deployment: dict, unit: _LimitUnit) -> list[TagRateLimitEntry]:
    raw_tag_rate_limits = (deployment.get("model_info") or {}).get("tag_rate_limits")
    if not raw_tag_rate_limits:
        return []
    tag_rate_limits = TagRateLimits.model_validate(raw_tag_rate_limits)
    group = getattr(tag_rate_limits, _UNIT_TO_GROUP_FIELD[unit])
    return group.limits if group is not None else []


def _build_group_limits(deployments: list[dict], unit: _LimitUnit) -> list[_ConfiguredLimit]:
    """
    One `_ConfiguredLimit` per distinct (tag_id, name, limit, period_seconds)
    declared for `unit` across `deployments` (all sharing one `model_name`).

    A signature declared identically by every deployment in the group is
    chain-wide (one shared bucket, regardless of which deployment serves).
    A signature declared by only some deployments, or where deployments
    genuinely disagree on the value for the same (tag_id, name), becomes a
    per-deployment-scoped bucket shared by exactly the deployments that
    declared that value -- silently dropping a divergent deployment's config
    (as a naive dedupe-by-name index would) is the exact bug this guards
    against.
    """
    declaring_ids_by_signature: dict[tuple[str, str, float, int], list[str]] = {}
    for deployment in deployments:
        dep_id = _deployment_id(deployment)
        if dep_id is None:
            continue
        for entry in _entries_for_unit(deployment, unit):
            signature = (entry.tag_id, entry.name, entry.limit, entry.period_seconds)
            declaring_ids_by_signature.setdefault(signature, []).append(dep_id)

    distinct_signatures_by_name: dict[tuple[str, str], int] = {}
    for tag_id, name, _limit, _period in declaring_ids_by_signature:
        key = (tag_id, name)
        distinct_signatures_by_name[key] = distinct_signatures_by_name.get(key, 0) + 1

    total_deployments = len(deployments)
    configured: list[_ConfiguredLimit] = []
    for signature, declaring_ids in declaring_ids_by_signature.items():
        tag_id, name, limit, period_seconds = signature
        is_chain_wide = distinct_signatures_by_name[(tag_id, name)] == 1 and len(declaring_ids) == total_deployments
        configured.append(
            _ConfiguredLimit(
                unit=unit,
                entry=TagRateLimitEntry(name=name, tag_id=tag_id, limit=limit, period_seconds=period_seconds),
                deployment_scope=None if is_chain_wide else tuple(sorted(declaring_ids)),
            )
        )
    return configured


def _build_limits_index(model_list: list[dict]) -> dict[str, list[_ConfiguredLimit]]:
    groups: dict[str, list[dict]] = {}
    for deployment in model_list:
        groups.setdefault(deployment["model_name"], []).append(deployment)

    index: dict[str, list[_ConfiguredLimit]] = {}
    for model_name, deployments in groups.items():
        configured = [limit for unit in _LIMIT_UNITS for limit in _build_group_limits(deployments, unit)]
        if configured:
            index[model_name] = configured
    return index


class _TagRateLimitIndex:
    """Rebuilds the limits index only when `llm_router.model_list` changes."""

    def __init__(self) -> None:
        self._cache_key: Optional[tuple[int, int]] = None
        self._index: dict[str, list[_ConfiguredLimit]] = {}

    def get(self, llm_router: Router) -> dict[str, list[_ConfiguredLimit]]:
        model_list = llm_router.model_list or []
        cache_key = (id(llm_router), len(model_list))
        if cache_key != self._cache_key:
            self._index = _build_limits_index(model_list)
            self._cache_key = cache_key
        return self._index


def _scope_suffix(deployment_scope: Optional[tuple[str, ...]]) -> str:
    return "chain" if deployment_scope is None else "dep:" + "+".join(deployment_scope)


def _bucket_key(model_group: str, configured: _ConfiguredLimit, tag_value: str, bucket_id: int) -> str:
    scope = _scope_suffix(configured.deployment_scope)
    hash_tag = f"tag_rl:{model_group}:{configured.unit}:{configured.entry.name}:{scope}:{tag_value}"
    return f"{{{hash_tag}}}:{bucket_id}"


class _PROXY_TagRateLimiter(CustomLogger):
    def __init__(
        self,
        internal_usage_cache: DualCache,
        time_provider: Optional[Callable[[], datetime]] = None,
    ):
        self.internal_usage_cache = InternalUsageCache(dual_cache=internal_usage_cache)
        self._v3 = _PROXY_MaxParallelRequestsHandler_v3(self.internal_usage_cache, time_provider=time_provider)
        self._time_provider = time_provider or datetime.now
        self._index = _TagRateLimitIndex()
        self.llm_router: Optional[Router] = None

    def update_variables(self, llm_router: Router) -> None:
        self.llm_router = llm_router

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list[dict],
        messages: Optional[list[AllMessageValues]],
        request_kwargs: Optional[dict] = None,
        parent_otel_span=None,
    ) -> list[dict]:
        if not healthy_deployments or not isinstance(healthy_deployments, list) or self.llm_router is None:
            return healthy_deployments

        configured = self._index.get(self.llm_router).get(model)
        if not configured:
            return healthy_deployments

        request_kwargs = request_kwargs or {}
        metadata_variable_name = get_metadata_variable_name_from_kwargs(request_kwargs)
        tags = _get_tags_from_request_kwargs(request_kwargs, metadata_variable_name=metadata_variable_name)

        present_deployment_ids = {dep_id for d in healthy_deployments if (dep_id := _deployment_id(d)) is not None}

        now = self._time_provider().timestamp()
        to_check: list[tuple[_ConfiguredLimit, str, str]] = []
        for configured_limit in configured:
            if configured_limit.deployment_scope is not None and not (
                present_deployment_ids & set(configured_limit.deployment_scope)
            ):
                continue
            tag_value = _extract_identity(tags, configured_limit.entry.tag_id)
            if tag_value is None:
                continue
            bucket_id = int(now) // configured_limit.entry.period_seconds
            key = _bucket_key(model, configured_limit, tag_value, bucket_id)
            to_check.append((configured_limit, tag_value, key))

        if not to_check:
            return healthy_deployments

        keys = [key for _, _, key in to_check]
        current_values = await self.internal_usage_cache.async_batch_get_cache(
            keys=keys,
            parent_otel_span=parent_otel_span,
            local_only=False,
        )
        if current_values is None:
            current_values = [None] * len(keys)

        for (configured_limit, tag_value, _key), current_value in zip(to_check, current_values):
            current = float(current_value) if current_value is not None else 0.0
            if current < configured_limit.entry.limit:
                continue
            verbose_proxy_logger.debug(
                "tag_rate_limiter: OVER_LIMIT model=%s unit=%s name=%s tag_id=%s tag_value=%s current=%s limit=%s",
                model,
                configured_limit.unit,
                configured_limit.entry.name,
                configured_limit.entry.tag_id,
                tag_value,
                current,
                configured_limit.entry.limit,
            )
            raise ProxyRateLimitError(
                detail={
                    "error": "tag_rate_limit_exceeded",
                    "type": configured_limit.unit,
                    "tag_id": configured_limit.entry.tag_id,
                    "tag_value": tag_value,
                    "limit_name": configured_limit.entry.name,
                    "limit": configured_limit.entry.limit,
                    "period_seconds": configured_limit.entry.period_seconds,
                },
                headers={"retry-after": str(configured_limit.entry.period_seconds)},
                rate_limit_type=_UNIT_TO_RATE_LIMIT_TYPE[configured_limit.unit],
                model=model,
                llm_provider="litellm_proxy",
            )

        return healthy_deployments

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        if self.llm_router is None:
            return

        standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get("standard_logging_object")
        if standard_logging_object is None:
            return

        model_group = standard_logging_object.get("model_group")
        if not model_group:
            return

        configured = self._index.get(self.llm_router).get(model_group)
        if not configured:
            return

        metadata_variable_name = get_metadata_variable_name_from_kwargs(kwargs)
        tags = _get_tags_from_request_kwargs(kwargs, metadata_variable_name=metadata_variable_name)
        if not tags:
            return

        deployment_id = standard_logging_object.get("model_id")
        now = self._time_provider().timestamp()
        increment_by_unit: dict[_LimitUnit, float] = {
            "tokens": float(standard_logging_object.get("total_tokens") or 0),
            "requests": 1.0,
            "dollars": float(standard_logging_object.get("response_cost") or 0),
        }

        operations: list[RedisPipelineIncrementOperation] = []
        for configured_limit in configured:
            if configured_limit.deployment_scope is not None and deployment_id not in configured_limit.deployment_scope:
                continue
            tag_value = _extract_identity(tags, configured_limit.entry.tag_id)
            if tag_value is None:
                continue
            increment_value = increment_by_unit[configured_limit.unit]
            if increment_value == 0:
                continue
            bucket_id = int(now) // configured_limit.entry.period_seconds
            key = _bucket_key(model_group, configured_limit, tag_value, bucket_id)
            operations.append(
                RedisPipelineIncrementOperation(
                    key=key,
                    increment_value=increment_value,
                    ttl=configured_limit.entry.period_seconds + 3600,
                )
            )

        if not operations:
            return

        asyncio.create_task(
            self._v3.async_increment_tokens_with_ttl_preservation(
                pipeline_operations=operations,
                parent_otel_span=_get_parent_otel_span_from_kwargs(kwargs),
            )
        )
