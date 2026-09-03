from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm._logging import verbose_router_logger

from .config import AutoSetupCandidate, AutoSetupObjective

_CACHE_PREFIX: Final = "auto_setup_response_latency:v1"
_CACHE_TTL_SECONDS: Final = 7 * 24 * 60 * 60
_MAX_SAMPLES_PER_MODEL: Final = 20
_MAX_EXPECTED_OUTPUT_SAMPLES: Final = 100
_MIN_SAMPLES_PER_MODEL: Final = 2
_OBJECT_ADAPTER: Final = TypeAdapter(object)
_OBJECT_MAPPING_ADAPTER: Final = TypeAdapter(dict[str, object])


class ResponseLatencyCache(Protocol):
    async def async_get_cache(
        self,
        key: str,
        parent_otel_span: object | None = None,
        local_only: bool = False,
        **kwargs: object,
    ) -> object: ...

    async def async_set_cache(
        self,
        key: str,
        value: object,
        local_only: bool = False,
        **kwargs: object,
    ) -> None: ...


class ResponseLatencySample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ttft_ms: float = Field(ge=0)
    output_tokens_per_ms: float = Field(gt=0)


class ResponseLatencyState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    samples_by_model: dict[str, list[ResponseLatencySample]] = Field(default_factory=dict)
    visible_output_tokens: list[int] = Field(default_factory=list)


def _cache_key(router_model_name: str, tier: str) -> str:
    namespace: Final = hashlib.sha256(router_model_name.encode("utf-8")).hexdigest()[:24]
    return f"{_CACHE_PREFIX}:{namespace}:{tier}"


def _state(value: object) -> ResponseLatencyState:
    try:
        return ResponseLatencyState.model_validate(value or {})
    except ValidationError:
        return ResponseLatencyState()


def _candidate_order(
    candidates: Sequence[AutoSetupCandidate],
    available_models: Sequence[str],
    cold_start_model: str,
) -> tuple[str, ...]:
    available: Final = frozenset(available_models)
    ranked: Final = tuple(candidate.model_name for candidate in candidates if candidate.model_name in available)
    extras: Final = tuple(model for model in available_models if model not in frozenset(ranked))
    ordered: Final = tuple(dict.fromkeys((*ranked, *extras)))
    if cold_start_model not in ordered:
        return ordered
    return (cold_start_model, *(model for model in ordered if model != cold_start_model))


def _median_prediction_ms(samples: Sequence[ResponseLatencySample], expected_output_tokens: float) -> float:
    return statistics.median(
        sample.ttft_ms + expected_output_tokens / sample.output_tokens_per_ms for sample in samples
    )


def _normalized_log(value: float, values: Sequence[float]) -> float:
    logged: Final = tuple(math.log(max(item, 1e-12)) for item in values)
    low: Final = min(logged)
    high: Final = max(logged)
    if low == high:
        return 0.0
    return (math.log(max(value, 1e-12)) - low) / (high - low)


def _best_observed_model(
    state: ResponseLatencyState,
    candidates: Sequence[AutoSetupCandidate],
    ordered_models: Sequence[str],
    objective: AutoSetupObjective,
) -> str:
    expected_tokens: Final = statistics.median(state.visible_output_tokens) if state.visible_output_tokens else 1.0
    latencies: Final = {
        model: _median_prediction_ms(state.samples_by_model[model], expected_tokens) for model in ordered_models
    }
    if objective == "task_completion_speed":
        return min(ordered_models, key=lambda model: (latencies[model], ordered_models.index(model)))

    configured_costs: Final = {candidate.model_name: candidate.cost_per_completed_task_usd for candidate in candidates}
    known_costs: Final = tuple(configured_costs.values())
    conservative_unknown_cost: Final = max(known_costs) if known_costs else 1.0
    costs: Final = {model: configured_costs.get(model, conservative_unknown_cost) for model in ordered_models}
    latency_values: Final = tuple(latencies.values())
    cost_values: Final = tuple(costs.values())
    return min(
        ordered_models,
        key=lambda model: (
            math.hypot(
                _normalized_log(latencies[model], latency_values),
                _normalized_log(costs[model], cost_values),
            ),
            ordered_models.index(model),
        ),
    )


async def select_runtime_response_model(
    *,
    router_cache: ResponseLatencyCache,
    router_model_name: str,
    tier: str,
    candidates: Sequence[AutoSetupCandidate],
    available_models: Sequence[str],
    cold_start_model: str,
    objective: AutoSetupObjective,
) -> str:
    """Explore quality-admitted groups, then minimize equal-output response time."""

    ordered: Final = _candidate_order(candidates, available_models, cold_start_model)
    if not ordered:
        raise ValueError(f"No Auto setup candidates remain for tier {tier}")
    try:
        cached = await router_cache.async_get_cache(key=_cache_key(router_model_name, tier))
        state: Final = _state(cached)
    except Exception as exc:  # noqa: BLE001 -- optional observations must never make a request unroutable
        verbose_router_logger.warning("Auto setup response-latency cache read failed: %s", exc)
        return ordered[0]

    counts: Final = {model: len(state.samples_by_model.get(model, ())) for model in ordered}
    least_observed: Final = min(counts.values())
    if least_observed < _MIN_SAMPLES_PER_MODEL:
        return next(model for model in ordered if counts[model] == least_observed)
    return _best_observed_model(state, candidates, ordered, objective)


def _seconds(value: object) -> float | None:
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _field(value: object, key: str) -> object | None:
    if isinstance(value, Mapping):
        try:
            return _OBJECT_MAPPING_ADAPTER.validate_python(value).get(key)
        except ValidationError:
            return None
    try:
        return _OBJECT_ADAPTER.validate_python(object.__getattribute__(value, key))
    except AttributeError:
        return None


def _visible_output_tokens(response_obj: object) -> int | None:
    usage = _field(response_obj, "usage")
    if usage is None:
        return None
    total = _field(usage, "completion_tokens")
    if total is None:
        total = _field(usage, "output_tokens")
    if isinstance(total, bool) or not isinstance(total, int | float) or total <= 0:
        return None
    details = _field(usage, "completion_tokens_details")
    if details is None:
        details = _field(usage, "output_tokens_details")
    reasoning = _field(details, "reasoning_tokens") if details is not None else None
    reasoning_tokens = float(reasoning) if isinstance(reasoning, int | float) and not isinstance(reasoning, bool) else 0
    return max(1, round(float(total) - reasoning_tokens))


def response_latency_sample(
    kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
) -> tuple[ResponseLatencySample, int] | None:
    start: Final = _seconds(start_time)
    end: Final = _seconds(end_time)
    visible_tokens: Final = _visible_output_tokens(response_obj)
    if start is None or end is None or end <= start or visible_tokens is None:
        return None
    total_ms: Final = (end - start) * 1000
    completion_start: Final = _seconds(kwargs.get("completion_start_time"))
    ttft_ms = (
        (completion_start - start) * 1000 if completion_start is not None and start < completion_start < end else 0.0
    )
    generation_ms: Final = max(total_ms - ttft_ms, 1e-6)
    return ResponseLatencySample(
        ttft_ms=ttft_ms,
        output_tokens_per_ms=visible_tokens / generation_ms,
    ), visible_tokens


async def record_runtime_response_latency(
    *,
    router_cache: ResponseLatencyCache,
    router_model_name: str,
    tier: str,
    routed_model: str,
    kwargs: Mapping[str, object],
    response_obj: object,
    start_time: object,
    end_time: object,
) -> None:
    measured: Final = response_latency_sample(kwargs, response_obj, start_time, end_time)
    if measured is None:
        return
    sample, visible_tokens = measured
    key: Final = _cache_key(router_model_name, tier)
    state: Final = _state(await router_cache.async_get_cache(key=key))
    samples: Final = [*state.samples_by_model.get(routed_model, ()), sample][-_MAX_SAMPLES_PER_MODEL:]
    state.samples_by_model[routed_model] = samples
    state.visible_output_tokens = [*state.visible_output_tokens, visible_tokens][-_MAX_EXPECTED_OUTPUT_SAMPLES:]
    await router_cache.async_set_cache(key=key, value=state.model_dump(mode="json"), ttl=_CACHE_TTL_SECONDS)
