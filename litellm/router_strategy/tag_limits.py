"""
Tag-based rate limiting.

A model group declares one grouping per unit on `model_info`, and each limit entry
carries its own `tag_id`, so limits of the same unit can key off different tags:

```yaml
model_info:
  token_limits:
    limits:
      - name: daily
        tag_id: end_user_id
        limit: 500000
        period_days: 1
  request_limits:
    limits:
      - name: daily
        tag_id: end_user_id
        limit: 1000
        period_days: 1
  dollar_limits:
    limits:
      - name: monthly
        tag_id: team_id
        limit: 50.0
        period_days: 30
```

Callers identify themselves with the existing tag mechanism, e.g.
`X-Litellm-Tags: end_user_id:user-123,team_id:team-a`. Usage is counted in fixed
windows of `period_days` and checked on every routing attempt, so a fallback model
group's own limits are enforced on its own hop. Requests carrying no matching
`tag_id` are unaffected.
"""

import json
import time
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache
from litellm.exceptions import RateLimitErrorCategory, RateLimitType
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.litellm_core_utils.core_helpers import (
    get_metadata_variable_name_from_kwargs,
)
from litellm.router_strategy.tag_based_routing import _get_tags_from_request_kwargs
from litellm.types.llms.openai import AllMessageValues
from litellm.types.tag_limits import (
    TAG_LIMIT_FIELD_BY_UNIT,
    DeploymentTagLimits,
    TagLimit,
    TagLimitUnit,
)

SECONDS_PER_DAY = 86400

_LIMIT_FIELDS: tuple[str, ...] = tuple(TAG_LIMIT_FIELD_BY_UNIT.values())

_RATE_LIMIT_TYPE_BY_UNIT: Mapping[TagLimitUnit, RateLimitType] = {
    "tokens": RateLimitType.TOKENS,
    "requests": RateLimitType.REQUESTS,
    "dollars": RateLimitType.BUDGET,
}


_MODEL_INFO_ADAPTER: TypeAdapter[Mapping[str, object]] = TypeAdapter(Mapping[str, object])


@lru_cache(maxsize=1024)
def _parse_tag_limits(serialized_limits: str) -> DeploymentTagLimits | None:
    try:
        limits = DeploymentTagLimits.model_validate_json(serialized_limits)
    except ValidationError as e:
        verbose_router_logger.warning("tag_limits: ignoring invalid limit config %s: %s", serialized_limits, e)
        return None
    return limits if limits.limits_by_unit() else None


def get_tag_limits(model_info: object) -> DeploymentTagLimits | None:
    """Parse (and memoize) the tag limits declared on a deployment's `model_info`."""
    if not isinstance(model_info, dict) or all(field not in model_info for field in _LIMIT_FIELDS):
        return None
    info = _MODEL_INFO_ADAPTER.validate_python(model_info)
    declared = {field: info[field] for field in _LIMIT_FIELDS if field in info}
    return _parse_tag_limits(json.dumps(declared, sort_keys=True, default=str))


def has_tag_limits(model_list: Sequence[Mapping[str, object]] | None) -> bool:
    if model_list is None:
        return False
    return any(_deployment_limits(deployment).limits_by_unit() for deployment in model_list)


def _tag_values_from_request(kwargs: Mapping[str, object] | None) -> Mapping[str, str]:
    """
    Turn `metadata.tags` entries of the form `<tag_id>:<value>` into a {tag_id: value} map.

    Tags without a `:` separator carry no identity and are ignored. On duplicate tag ids
    the last one wins.
    """
    request_kwargs = dict(kwargs or {})
    tags = _get_tags_from_request_kwargs(
        request_kwargs=request_kwargs,
        metadata_variable_name=get_metadata_variable_name_from_kwargs(request_kwargs),
    )
    return {tag_id: tag_value for tag_id, _, tag_value in (tag.partition(":") for tag in tags) if tag_id and tag_value}


def _window_seconds(limit: TagLimit) -> int:
    return max(int(limit.period_days * SECONDS_PER_DAY), 1)


def _usage_key(unit: TagLimitUnit, limit: TagLimit, tag_value: str, now: float) -> str:
    window = _window_seconds(limit)
    bucket = int(now // window)
    return f"tag_limit:{unit}:{limit.tag_id}:{tag_value}:{limit.name}:{limit.period_days}:{bucket}"


class _RequestUsage(BaseModel):
    """The `StandardLoggingPayload` fields tag counters are incremented by."""

    model_config = ConfigDict(extra="ignore")

    total_tokens: float = 0.0
    response_cost: float = 0.0


def _nested(container: object, key: str) -> object:
    return container.get(key) if isinstance(container, dict) else None


def _deployment_limits(deployment: Mapping[str, object]) -> DeploymentTagLimits:
    return get_tag_limits(deployment.get("model_info")) or DeploymentTagLimits()


class TagLimitCheck(CustomLogger):
    """Enforces `model_info` tag limits on every routing attempt, and counts usage on success."""

    def __init__(self, dual_cache: DualCache, now: Callable[[], float] = time.time):
        self.dual_cache = dual_cache
        self.now = now

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list[dict[str, object]]  # mutable-ok: router hands the live deployment dicts around
        | dict[str, object],
        messages: Sequence[AllMessageValues] | None,
        request_kwargs: Mapping[str, object] | None = None,
        parent_otel_span: Span | None = None,
    ) -> list[dict[str, object]]:  # mutable-ok: CustomLogger.async_filter_deployments returns a list of deployments
        deployments: list[dict[str, object]] = (  # mutable-ok: mirrors the list contract above
            [healthy_deployments] if isinstance(healthy_deployments, dict) else list(healthy_deployments)
        )
        if not deployments:
            return deployments

        tag_values = _tag_values_from_request(request_kwargs)
        if not tag_values:
            return deployments

        now = self.now()
        checks: tuple[tuple[int, TagLimitUnit, TagLimit, str, str], ...] = tuple(
            (idx, unit, limit, tag_values[limit.tag_id], _usage_key(unit, limit, tag_values[limit.tag_id], now))
            for idx, deployment in enumerate(deployments)
            for unit, limit in _deployment_limits(deployment).limits_by_unit()
            if limit.tag_id in tag_values
        )
        if not checks:
            return deployments

        keys = tuple(dict.fromkeys(check[4] for check in checks))
        values = await self.dual_cache.async_batch_get_cache(keys=list(keys), parent_otel_span=parent_otel_span)
        usage: Mapping[str, float] = {key: float(value or 0.0) for key, value in zip(keys, values or [0.0] * len(keys))}

        violations = tuple(check for check in checks if usage[check[4]] >= check[2].limit)
        blocked_indexes = frozenset(violation[0] for violation in violations)
        allowed = [deployment for idx, deployment in enumerate(deployments) if idx not in blocked_indexes]
        if allowed:
            return allowed

        _, unit, limit, tag_value, key = violations[0]
        error = {
            "message": "tag_rate_limit_exceeded",
            "type": unit,
            "tag_id": limit.tag_id,
            "tag_value": tag_value,
            "limit_name": limit.name,
            "limit": limit.limit,
            "period_days": limit.period_days,
        }
        verbose_router_logger.info("tag_limits: blocking request, usage=%s, %s", usage[key], error)
        raise litellm.RateLimitError(
            message=json.dumps({"error": error}),
            llm_provider="litellm",
            model=model,
            category=RateLimitErrorCategory.LITELLM_RATE_LIMIT,
            rate_limit_type=_RATE_LIMIT_TYPE_BY_UNIT[unit],
            detail={"error": error},
        )

    async def async_log_success_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        tag_limits = get_tag_limits(_nested(kwargs.get("litellm_params"), "model_info"))
        if tag_limits is None:
            return

        tag_values = _tag_values_from_request(kwargs)
        if not tag_values:
            return

        standard_logging_payload = kwargs.get("standard_logging_object")
        if not isinstance(standard_logging_payload, dict):
            return

        usage = _RequestUsage.model_validate(standard_logging_payload)
        usage_by_unit: Mapping[TagLimitUnit, float] = {
            "tokens": usage.total_tokens,
            "requests": 1.0,
            "dollars": usage.response_cost,
        }

        now = self.now()
        for unit, limit in tag_limits.limits_by_unit():
            tag_value = tag_values.get(limit.tag_id)
            if tag_value is None or usage_by_unit[unit] == 0:
                continue
            await self.dual_cache.async_increment_cache(
                key=_usage_key(unit, limit, tag_value, now),
                value=usage_by_unit[unit],
                ttl=2 * _window_seconds(limit),
            )
