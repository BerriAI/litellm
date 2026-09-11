from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import accumulate
from typing import Final, Literal, TypeAlias

from litellm.caching.dual_cache import DualCache
from litellm.litellm_core_utils.llm_cost_calc.utils import parse_prompt_tokens_details
from litellm.types.utils import Usage
from litellm.utils import get_prompt_cache_min_tokens, token_counter

PredictionState: TypeAlias = Literal["warm", "partial", "stale", "unknown", "disabled"]
_FIVE_MINUTES: Final = 300
_ONE_HOUR: Final = 3600
_OBSERVATION_TTL: Final = 7200
_LOOKBACK: Final = 20
_MAX_BODY_BYTES: Final = 2_000_000
_MAX_SEGMENTS: Final = 4096
_NAMESPACE: Final = "prompt-cache-prediction:v1"
_CONFIG_FIELDS: Final = ("tool_choice", "speed", "service_tier", "inference_geo", "output_config")
_UNSUPPORTED_FIELDS: Final = ("thinking", "context_management", "mcp_servers", "container", "compaction")
_ALLOWED_BLOCK_TYPES: Final = frozenset(("text", "tool_use", "tool_result"))


@dataclass(frozen=True, slots=True)
class PromptCacheTokenBudget:
    cache_read_input_tokens: int
    cache_creation_input_tokens_5m: int
    cache_creation_input_tokens_1h: int
    uncached_input_tokens: int


@dataclass(frozen=True, slots=True)
class PromptCacheBoundary:
    index: int
    digest: str
    estimated_tokens: int
    ttl_seconds: int


@dataclass(frozen=True, slots=True)
class PromptCachePlan:
    model: str
    boundaries: tuple[PromptCacheBoundary, ...]
    final_checkpoint_index: int
    final_ttl_seconds: int
    total_estimated_tokens: int
    candidate_count: int
    current_min_tokens: int

    @property
    def final_boundary(self) -> PromptCacheBoundary:
        return next(boundary for boundary in reversed(self.boundaries) if boundary.index == self.final_checkpoint_index)

    def cold_budget(self) -> PromptCacheTokenBudget:
        final_tokens: Final = self.final_boundary.estimated_tokens
        return PromptCacheTokenBudget(
            0,
            final_tokens if self.final_ttl_seconds == _FIVE_MINUTES else 0,
            final_tokens if self.final_ttl_seconds == _ONE_HOUR else 0,
            max(self.total_estimated_tokens - final_tokens, 0),
        )

    def fully_warm_budget(self) -> PromptCacheTokenBudget:
        final_tokens: Final = self.final_boundary.estimated_tokens
        return PromptCacheTokenBudget(final_tokens, 0, 0, max(self.total_estimated_tokens - final_tokens, 0))


@dataclass(frozen=True, slots=True)
class Prediction:
    state: PredictionState
    cache_read_input_tokens: int
    cache_creation_input_tokens_5m: int
    cache_creation_input_tokens_1h: int
    uncached_input_tokens: int
    as_of: float | None
    expires_at: float | None
    reason: str


@dataclass(frozen=True, slots=True)
class _Segment:
    serialized: str
    ttl: int | None


@dataclass(frozen=True, slots=True)
class _Observation:
    provider_tokens: int
    estimated_tokens: int
    observed_at: float
    expires_at: float
    ttl_seconds: int


def _serialize(value: object) -> str | None:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return None


def _ttl(control: object) -> int | None:
    if not isinstance(control, Mapping) or control.get("type") != "ephemeral":
        return None
    ttl: Final = control.get("ttl", "5m")
    return _FIVE_MINUTES if ttl == "5m" else _ONE_HOUR if ttl == "1h" else None


def _without_marker(value: Mapping[object, object]) -> Mapping[object, object]:
    return {  # mutable-ok: JSON serialization needs the original mapping shape without its marker
        key: item for key, item in value.items() if key != "cache_control"
    }


def _segment(value: object, context: tuple[object, ...], marker: object = None) -> _Segment | None:
    normalized: Final = _without_marker(value) if isinstance(value, Mapping) else value
    serialized: Final = _serialize((context, normalized))
    ttl: Final = _ttl(marker) if marker is not None else None
    return None if serialized is None or (marker is not None and ttl is None) else _Segment(serialized, ttl)


def _tools(value: object) -> tuple[_Segment, ...] | None:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    tools: Final = tuple(value)
    if any(
        not isinstance(tool, Mapping)
        or not isinstance(tool.get("name"), str)
        or not isinstance(tool.get("input_schema"), Mapping)
        or tool.get("type", "custom") != "custom"
        for tool in tools
    ):
        return None
    parsed: Final = tuple(
        _segment(tool, ("tool", index), tool.get("cache_control"))
        for index, tool in enumerate(tool for tool in tools if isinstance(tool, Mapping))
    )
    return None if any(item is None for item in parsed) else tuple(item for item in parsed if item is not None)


def _system(value: object) -> tuple[_Segment, ...] | None:
    if value is None:
        return ()
    blocks: Final = (value,) if isinstance(value, str) else tuple(value) if isinstance(value, Sequence) else ()
    if not blocks:
        return None
    parsed: Final = tuple(
        _segment(
            ("text", block) if isinstance(block, str) else block,
            ("system", index),
            block.get("cache_control") if isinstance(block, Mapping) else None,
        )
        for index, block in enumerate(blocks)
    )
    return None if any(item is None for item in parsed) else tuple(item for item in parsed if item is not None)


def _valid_block(block: object) -> bool:
    if isinstance(block, str):
        return True
    if not isinstance(block, Mapping) or block.get("type") not in _ALLOWED_BLOCK_TYPES:
        return False
    block_type: Final = block.get("type")
    if block_type == "text":
        return isinstance(block.get("text"), str)
    if block_type == "tool_use":
        return all(isinstance(block.get(field), str) for field in ("id", "name")) and isinstance(
            block.get("input"), Mapping
        )
    content: Final = block.get("content")
    return isinstance(block.get("tool_use_id"), str) and (
        isinstance(content, str)
        or (
            isinstance(content, Sequence)
            and not isinstance(content, (str, bytes, bytearray))
            and all(isinstance(item, Mapping) and item.get("type") == "text" for item in content)
        )
    )


def _message(message: object, message_index: int) -> tuple[_Segment, ...] | None:
    if not isinstance(message, Mapping) or message.get("role") not in ("user", "assistant"):
        return None
    role: Final = message.get("role")
    content: Final = message.get("content")
    blocks: Final = (content,) if isinstance(content, str) else tuple(content) if isinstance(content, Sequence) else ()
    if not isinstance(role, str) or not blocks or not all(_valid_block(block) for block in blocks):
        return None
    parsed: Final = tuple(
        _segment(
            ("text", block) if isinstance(block, str) else block,
            ("message", message_index, index, role),
            block.get("cache_control") if isinstance(block, Mapping) else None,
        )
        for index, block in enumerate(blocks)
    )
    if any(item is None for item in parsed):
        return None
    segments: Final = tuple(item for item in parsed if item is not None)
    if "cache_control" not in message:
        return segments
    message_ttl: Final = _ttl(message["cache_control"])
    return (
        None
        if message_ttl is None or any(item.ttl is not None for item in segments)
        else (*segments[:-1], replace(segments[-1], ttl=message_ttl))
    )


def _messages(value: object) -> tuple[_Segment, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    parsed: Final = tuple(_message(message, index) for index, message in enumerate(value))
    if not parsed:
        return None
    valid: Final = tuple(items for items in parsed if items is not None)
    return None if len(valid) != len(parsed) else tuple(item for items in valid for item in items)


def make_plan(body: object) -> PromptCachePlan | None:
    if not isinstance(body, Mapping):
        return None
    serialized_body: Final = _serialize(body)
    if serialized_body is None or len(serialized_body.encode()) > _MAX_BODY_BYTES:
        return None
    model: Final = body.get("model")
    if (
        not isinstance(model, str)
        or "claude-" not in model
        or any(field in body and body[field] is not None for field in _UNSUPPORTED_FIELDS)
    ):
        return None
    output_config: Final = body.get("output_config")
    if output_config is not None and (
        not isinstance(output_config, Mapping) or frozenset(output_config) - frozenset(("effort",))
    ):
        return None
    envelope: Final = _serialize((model, tuple((field, body[field]) for field in _CONFIG_FIELDS if field in body)))
    tools: Final = _tools(body.get("tools"))
    system: Final = _system(body.get("system"))
    messages: Final = _messages(body.get("messages"))
    if envelope is None or tools is None or system is None or messages is None:
        return None
    segments: Final = (*tools, *system, *messages)
    if len(segments) > _MAX_SEGMENTS:
        return None
    explicit: Final = tuple((index, item.ttl) for index, item in enumerate(segments) if item.ttl is not None)
    root_control: Final = body.get("cache_control")
    root_ttl: Final = _ttl(root_control) if root_control is not None else None
    if root_control is not None and root_ttl is None:
        return None
    automatic: Final = ((len(segments) - 1, root_ttl),) if root_ttl is not None else ()
    markers: Final = (*explicit, *automatic)
    ttl_values: Final = frozenset(ttl for _, ttl in markers)
    marked: Final = tuple(sorted(frozenset(index for index, _ in markers)))
    if len(marked) != 1 or len(ttl_values) != 1:
        return None
    uniform_ttl: Final = next(iter(ttl_values))
    marker: Final = marked[0]
    eligible: Final = tuple(range(max(0, marker - _LOOKBACK + 1), marker + 1))
    envelope_digest: Final = hashlib.sha256(envelope.encode()).digest()
    digest_chain: Final = tuple(
        accumulate(
            (item.serialized.encode() for item in segments),
            lambda digest, item: hashlib.sha256(digest + len(item).to_bytes(8, "big") + item).digest(),
            initial=envelope_digest,
        )
    )[1:]
    digests: Final = tuple(digest_chain[index].hex() for index in eligible)
    try:
        envelope_tokens: Final = token_counter(model=model, text=envelope)
        segment_estimates: Final = tuple(token_counter(model=model, text=segment.serialized) for segment in segments)
    except Exception:
        return None
    if envelope_tokens <= 0 or any(estimate <= 0 for estimate in segment_estimates):
        return None
    cumulative_estimates: Final = tuple(envelope_tokens + estimate for estimate in accumulate(segment_estimates))
    estimates: Final = tuple(cumulative_estimates[index] for index in eligible)
    total: Final = cumulative_estimates[-1]
    try:
        minimum: Final = get_prompt_cache_min_tokens(model=model)
    except Exception:
        return None
    boundaries: Final = tuple(
        PromptCacheBoundary(index, digest, estimate, uniform_ttl)
        for index, digest, estimate in zip(eligible, digests, estimates)
    )
    return PromptCachePlan(model, boundaries, marker, uniform_ttl, total, len(boundaries), minimum)


def _usage(usage: Mapping[str, object]) -> tuple[int, int, int, int] | None:
    try:
        parsed: Final = parse_prompt_tokens_details(Usage(**usage))  # mutable-ok: Pydantic validates the usage copy
    except Exception:  # noqa: BLE001  # malformed provider telemetry means no observation
        return None
    read: Final = parsed["cache_hit_tokens"]
    write: Final = parsed["cache_creation_tokens"]
    details: Final = parsed["cache_creation_token_details"]
    five: Final = details.ephemeral_5m_input_tokens or 0 if details is not None else 0
    hour: Final = details.ephemeral_1h_input_tokens or 0 if details is not None else 0
    return (read, write, five, hour) if five + hour == write else None


def _key(scope: str, deployment_id: str, model: str, digest: str) -> str:
    identity: Final = _serialize((_NAMESPACE, scope, deployment_id, model, digest)) or ""
    return f"{_NAMESPACE}:{hashlib.sha256(identity.encode()).hexdigest()}"


def _observation(value: object) -> _Observation | None:
    try:
        decoded: Final = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return None
    if not isinstance(decoded, Sequence) or isinstance(decoded, (str, bytes, bytearray)) or len(decoded) != 5:
        return None
    provider, estimate, observed, expires, ttl = decoded
    if (
        not isinstance(provider, int)
        or isinstance(provider, bool)
        or provider <= 0
        or not isinstance(estimate, int)
        or isinstance(estimate, bool)
        or estimate <= 0
        or not isinstance(observed, (int, float))
        or isinstance(observed, bool)
        or not isinstance(expires, (int, float))
        or isinstance(expires, bool)
        or ttl not in (_FIVE_MINUTES, _ONE_HOUR)
    ):
        return None
    return _Observation(provider, estimate, float(observed), float(expires), ttl)


def _value(observation: _Observation) -> str:
    return (
        _serialize(
            (
                observation.provider_tokens,
                observation.estimated_tokens,
                observation.observed_at,
                observation.expires_at,
                observation.ttl_seconds,
            )
        )
        or ""
    )


async def observe(
    cache: DualCache,
    scope: str,
    deployment_id: str,
    plan: PromptCachePlan,
    usage: Mapping[str, object],
    started_at: float,
) -> None:
    counts: Final = _usage(usage)
    if not scope or not deployment_id or not math.isfinite(started_at) or counts is None:
        return
    read, write, five, hour = counts
    if read <= 0 and write <= 0:
        return
    boundary: Final = plan.final_boundary
    key: Final = _key(scope, deployment_id, plan.model, boundary.digest)
    if write > 0:
        matching_write: Final = five if plan.final_ttl_seconds == _FIVE_MINUTES else hour
        if matching_write != write:
            return
        observation: Final = _Observation(
            read + write,
            boundary.estimated_tokens,
            started_at,
            started_at + plan.final_ttl_seconds,
            plan.final_ttl_seconds,
        )
        await cache.async_set_cache(key, _value(observation), ttl=_OBSERVATION_TTL)
        return
    existing: Final = _observation(await cache.async_get_cache(key))
    if existing is None or existing.provider_tokens != read or existing.estimated_tokens != boundary.estimated_tokens:
        return
    refreshed: Final = replace(existing, observed_at=started_at, expires_at=started_at + existing.ttl_seconds)
    await cache.async_set_cache(key, _value(refreshed), ttl=_OBSERVATION_TTL)


def _prediction(
    state: PredictionState, budget: PromptCacheTokenBudget, reason: str, observation: _Observation | None = None
) -> Prediction:
    return Prediction(
        state,
        budget.cache_read_input_tokens,
        budget.cache_creation_input_tokens_5m,
        budget.cache_creation_input_tokens_1h,
        budget.uncached_input_tokens,
        observation.observed_at if observation else None,
        observation.expires_at if observation else None,
        reason,
    )


def _cold(
    plan: PromptCachePlan, state: PredictionState, reason: str, observation: _Observation | None = None
) -> Prediction:
    return _prediction(state, plan.cold_budget(), reason, observation)


async def predict(cache: DualCache, scope: str, deployment_id: str, plan: PromptCachePlan, now: float) -> Prediction:
    final: Final = plan.final_boundary
    if final.estimated_tokens < plan.current_min_tokens:
        return _prediction(
            "disabled",
            PromptCacheTokenBudget(0, 0, 0, plan.total_estimated_tokens),
            "cacheable_prefix_below_model_minimum",
        )
    if not scope or not deployment_id or not math.isfinite(now):
        return _cold(plan, "unknown", "invalid_scope_deployment_or_time_cold_assumption")
    boundaries: Final = tuple(reversed(plan.boundaries))
    values: Final = await cache.async_batch_get_cache(
        list(
            _key(scope, deployment_id, plan.model, item.digest) for item in boundaries
        )  # mutable-ok: DualCache batch API requires list
    )
    if values is None or len(values) != len(boundaries):
        return _cold(plan, "unknown", "observation_lookup_failed_cold_assumption")
    parsed: Final = tuple((boundary, value, _observation(value)) for boundary, value in zip(boundaries, values))
    compatible: Final = tuple(
        (boundary, observation)
        for boundary, _, observation in parsed
        if observation is not None
        and observation.estimated_tokens == boundary.estimated_tokens
        and observation.ttl_seconds == plan.final_ttl_seconds
        and boundary.estimated_tokens <= final.estimated_tokens
    )
    live: Final = next(((boundary, item) for boundary, item in compatible if item.expires_at > now), None)
    if live is not None:
        boundary, item = live
        suffix: Final = max(final.estimated_tokens - boundary.estimated_tokens, 0)
        budget: Final = PromptCacheTokenBudget(
            item.provider_tokens,
            suffix if plan.final_ttl_seconds == _FIVE_MINUTES else 0,
            suffix if plan.final_ttl_seconds == _ONE_HOUR else 0,
            max(plan.total_estimated_tokens - boundary.estimated_tokens - suffix, 0),
        )
        exact: Final = boundary.index == plan.final_checkpoint_index
        return _prediction(
            "warm" if exact else "partial",
            budget,
            "live_exact_observation" if exact else "live_ancestor_observation",
            item,
        )
    stale: Final = next((item for _, item in compatible if item.expires_at <= now), None)
    if stale is not None:
        return _cold(plan, "stale", "compatible_observation_expired", stale)
    if any(value is not None and item is None for _, value, item in parsed):
        return _cold(plan, "unknown", "invalid_observation_cold_assumption")
    return _cold(plan, "unknown", "no_observation_cold_assumption")
