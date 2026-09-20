"""Pure, chronological cache accounting for the recorded baseline comparison.

Observation collection, pricing and durable publication belong to their existing
owners. Replaying these values in event order is independent of callback order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby
from math import isfinite
from types import MappingProxyType
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from litellm.llms.anthropic.prompt_cache_prediction import CountedBreakpoint, CountedPromptCachePlan
from litellm.types.utils import CacheCreationTokenDetails, PromptTokensDetailsWrapper, Usage

MAX_CACHE_TTL: Final = 3600
MAX_CACHE_ENTRIES: Final = 1024


class BaselineObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[3] = 3
    request_id: str = Field(min_length=1)
    started_at: float = Field(allow_inf_nan=False, ge=0)
    available_at: float = Field(allow_inf_nan=False, ge=0)
    outcome: Literal["complete", "uncertain", "response_cache"]
    baseline_equivalent: bool
    usage: Usage | None = None
    plan: CountedPromptCachePlan | None = None
    minimum_cache_tokens: int = Field(default=0, ge=0)
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class BaselineEstimate:
    request_id: str
    reason: str
    provenance: Literal["observed_identical", "modeled"] | None = None
    usage: Usage | None = None


@dataclass(frozen=True, slots=True)
class CacheEntry:
    fingerprint: str
    content_fingerprint: str
    tokens: int
    ttl_seconds: int
    available_at: float
    expires_at: float
    uncertain: bool = False


@dataclass(frozen=True, slots=True)
class BaselineHistory:
    first_at: float | None = None
    last_at: float | None = None
    equivalent: bool = True
    uncertain_before: float = 0.0
    entries: tuple[CacheEntry, ...] = ()
    blocked_until: float = 0.0


def _complete_usage(usage: Usage | None) -> bool:
    if usage is None or usage.prompt_tokens < 0 or usage.completion_tokens < 0:
        return False
    details: Final = usage.prompt_tokens_details
    if details is None:
        return False
    values: Final = (details.text_tokens, details.cached_tokens, details.cache_creation_tokens)
    if any(value is None or value < 0 for value in values):
        return False
    split: Final = details.cache_creation_token_details
    writes: Final = details.cache_creation_tokens or 0
    return (
        usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
        and sum(value or 0 for value in values) == usage.prompt_tokens
        and (
            writes == 0
            or (
                split is not None
                and split.ephemeral_5m_input_tokens is not None
                and split.ephemeral_1h_input_tokens is not None
                and min(split.ephemeral_5m_input_tokens, split.ephemeral_1h_input_tokens) >= 0
                and split.ephemeral_5m_input_tokens + split.ephemeral_1h_input_tokens == writes
            )
        )
    )


def _valid_plan(plan: CountedPromptCachePlan | None) -> bool:
    if plan is None or plan.total_tokens < 0 or len(plan.breakpoints) > 4:
        return False
    return all(
        marker.fingerprint
        and marker.content_fingerprint
        and marker.fingerprint in marker.lookback_fingerprints
        and marker.content_fingerprint in marker.lookback_content_fingerprints
        and marker.ttl_seconds in (300, 3600)
        and 0 <= marker.prefix_tokens <= plan.total_tokens
        for marker in plan.breakpoints
    ) and all(
        left.prefix_tokens <= right.prefix_tokens and left.ttl_seconds >= right.ttl_seconds
        for left, right in zip(plan.breakpoints, plan.breakpoints[1:])
    )


def _markers(observation: BaselineObservation) -> tuple[CountedBreakpoint, ...]:
    return (
        tuple(
            marker
            for marker in observation.plan.breakpoints
            if marker.prefix_tokens >= observation.minimum_cache_tokens
        )
        if observation.plan is not None
        else ()
    )


def _matches(entry: CacheEntry, markers: tuple[CountedBreakpoint, ...], started: float) -> bool:
    return entry.available_at <= started < entry.expires_at and any(
        entry.fingerprint in marker.lookback_fingerprints
        and entry.tokens <= marker.prefix_tokens
        and entry.ttl_seconds == marker.ttl_seconds
        for marker in markers
    )


def _ambiguous(entry: CacheEntry, markers: tuple[CountedBreakpoint, ...], started: float) -> bool:
    return entry.available_at <= started < entry.expires_at and any(
        entry.content_fingerprint in marker.lookback_content_fingerprints
        and (entry.uncertain or entry.ttl_seconds != marker.ttl_seconds)
        for marker in markers
    )


def _usage_with_cache(usage: Usage, total: int, read: int, write_5m: int, write_1h: int) -> Usage:
    writes: Final = write_5m + write_1h
    original_details: Final = usage.prompt_tokens_details or PromptTokensDetailsWrapper()
    details: Final = original_details.model_copy(
        deep=True,
        update=MappingProxyType(
            {
                "text_tokens": total - read - writes,
                "cached_tokens": read,
                "cache_creation_tokens": writes,
                "cache_write_tokens": writes,
                "cache_creation_token_details": CacheCreationTokenDetails(
                    ephemeral_5m_input_tokens=write_5m,
                    ephemeral_1h_input_tokens=write_1h,
                ),
            }
        ),
    )
    return Usage.model_validate(
        {  # mutable-ok: Usage only runs its normalizing constructor for a plain dictionary
            **usage.model_dump(),
            "prompt_tokens": total,
            "total_tokens": total + usage.completion_tokens,
            "prompt_tokens_details": details,
            "cache_read_input_tokens": read,
            "cache_creation_input_tokens": writes,
        },
    )


def _estimate(history: BaselineHistory, observation: BaselineObservation, equivalent: bool) -> BaselineEstimate:
    if observation.outcome != "complete" or not _complete_usage(observation.usage):
        return BaselineEstimate(observation.request_id, observation.reason or observation.outcome)
    usage: Final = observation.usage
    if usage is None:
        return BaselineEstimate(observation.request_id, "missing_usage")
    if equivalent and observation.baseline_equivalent:
        return BaselineEstimate(
            observation.request_id, "identical_baseline_path", "observed_identical", usage.model_copy(deep=True)
        )
    if observation.started_at < history.blocked_until:
        return BaselineEstimate(observation.request_id, "concurrent_uncertainty")
    plan: Final = observation.plan
    if not _valid_plan(plan) or plan is None:
        return BaselineEstimate(observation.request_id, observation.reason or "unsupported_cache_plan")
    markers: Final = _markers(observation)
    if any(_ambiguous(entry, markers, observation.started_at) for entry in history.entries):
        return BaselineEstimate(observation.request_id, "cache_ttl_changed")
    read: Final = max(
        (
            entry.tokens
            for entry in history.entries
            if not entry.uncertain and _matches(entry, markers, observation.started_at)
        ),
        default=0,
    )
    end: Final = markers[-1].prefix_tokens if markers else 0
    if read < end and observation.started_at < history.uncertain_before + max(marker.ttl_seconds for marker in markers):
        return BaselineEstimate(observation.request_id, "history_unavailable")
    one_hour: Final = max(
        (marker.prefix_tokens for marker in markers if marker.ttl_seconds == 3600 and marker.prefix_tokens > read),
        default=read,
    )
    expired: Final = any(
        entry.expires_at <= observation.started_at
        and any(entry.fingerprint in marker.lookback_fingerprints for marker in markers)
        for entry in history.entries
    )
    reason: Final = (
        "cache_prefix_available"
        if read
        else "cache_prefix_expired"
        if expired
        else "cache_prefix_cold"
        if markers
        else "below_cache_minimum"
        if plan.breakpoints
        else "no_cache_breakpoints"
    )
    return BaselineEstimate(
        observation.request_id,
        reason,
        "modeled",
        _usage_with_cache(usage, plan.total_tokens, read, end - one_hour, one_hour - read),
    )


def _writes(history: BaselineHistory, observation: BaselineObservation) -> tuple[CacheEntry, ...]:
    if (
        observation.outcome != "complete"
        or observation.started_at < history.blocked_until
        or not _complete_usage(observation.usage)
        or not _valid_plan(observation.plan)
    ):
        return ()
    markers: Final = _markers(observation)
    ambiguous: Final = tuple(entry for entry in history.entries if _ambiguous(entry, markers, observation.started_at))
    hit: Final = (
        max(
            (
                entry
                for entry in history.entries
                if not entry.uncertain and _matches(entry, markers, observation.started_at)
            ),
            key=lambda entry: entry.tokens,
            default=None,
        )
        if not ambiguous
        else None
    )
    refresh: Final = (
        (
            CacheEntry(
                hit.fingerprint,
                hit.content_fingerprint,
                hit.tokens,
                hit.ttl_seconds,
                observation.available_at,
                observation.started_at + hit.ttl_seconds,
            ),
        )
        if hit is not None and all(marker.fingerprint != hit.fingerprint for marker in markers)
        else ()
    )
    return (
        *refresh,
        *(
            CacheEntry(
                marker.fingerprint,
                marker.content_fingerprint,
                marker.prefix_tokens,
                marker.ttl_seconds,
                observation.available_at,
                observation.started_at + max((marker.ttl_seconds, *(entry.ttl_seconds for entry in ambiguous))),
                uncertain=bool(ambiguous),
            )
            for marker in markers
        ),
    )


def _entry_key(entry: CacheEntry) -> tuple[str, str, int, int, bool]:
    return entry.fingerprint, entry.content_fingerprint, entry.tokens, entry.ttl_seconds, entry.uncertain


def _compact_entries(entries: tuple[CacheEntry, ...], started: float) -> tuple[CacheEntry, ...]:
    ordered: Final = sorted((entry for entry in entries if entry.expires_at >= started - MAX_CACHE_TTL), key=_entry_key)
    return tuple(
        retained
        for _, values in groupby(ordered, key=_entry_key)
        for group in (tuple(values),)
        for retained in (
            max(
                (entry for entry in group if entry.available_at <= started),
                key=lambda entry: entry.expires_at,
                default=None,
            ),
            *(entry for entry in group if entry.available_at > started),
        )
        if retained is not None
    )


def advance_baseline_history(
    history: BaselineHistory,
    simultaneous: Sequence[BaselineObservation],
) -> tuple[BaselineHistory, tuple[BaselineEstimate, ...]]:
    """Apply one request-start timestamp; ties cannot manufacture initial equality.

    The storage owner groups and orders observations before calling this function.
    Equal timestamps are evaluated against the same preceding cache snapshot.
    """
    if not simultaneous:
        return history, ()
    started: Final = simultaneous[0].started_at
    valid_order: Final = (
        isfinite(started)
        and all(item.started_at == started and item.available_at >= started for item in simultaneous)
        and (history.last_at is None or started > history.last_at)
    )
    if not valid_order:
        return history, tuple(BaselineEstimate(item.request_id, "invalid_observation_order") for item in simultaneous)
    first: Final = started if history.first_at is None else history.first_at
    uncertain: Final = max(history.uncertain_before, first)
    relevant: Final = tuple(item for item in simultaneous if item.outcome != "response_cache")
    equivalent: Final = history.equivalent and all(item.baseline_equivalent for item in relevant)
    before: Final = BaselineHistory(
        first, history.last_at, equivalent, uncertain, history.entries, history.blocked_until
    )
    estimates: Final = tuple(_estimate(before, item, equivalent) for item in simultaneous)
    invalidated: Final = any(
        item.outcome != "complete" or not _complete_usage(item.usage) or not _valid_plan(item.plan) for item in relevant
    )
    blocked: Final = max((history.blocked_until, *(item.available_at for item in relevant if invalidated)))
    entries: Final = _compact_entries(
        () if invalidated else (*history.entries, *(entry for item in relevant for entry in _writes(before, item))),
        started,
    )
    overflow: Final = len(entries) > MAX_CACHE_ENTRIES
    return BaselineHistory(
        first_at=first,
        last_at=started,
        equivalent=equivalent,
        uncertain_before=max(started, blocked) if invalidated or overflow else uncertain,
        entries=() if overflow else entries,
        blocked_until=blocked,
    ), estimates
