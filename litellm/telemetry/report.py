"""The wire shape of one telemetry report: what an exporter sends or stores for one window"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Final, TypeAlias, TypeVar

from litellm.telemetry.histogram import ATTEMPT_BOUNDS, BLOCK_COUNT_BOUNDS, LATENCY_BOUNDS_MS, Histogram
from litellm.telemetry.records import (
    AttemptRecord,
    BlockType,
    InstanceInfo,
    RequestRecord,
    StatusClass,
    TelemetryGroup,
    TokenCounts,
    UIAction,
    UIEvent,
)

REPORT_SCHEMA_VERSION: Final = 1

JsonValue: TypeAlias = "str | int | float | bool | None | Sequence[JsonValue] | Mapping[str, JsonValue]"

_K: Final = TypeVar("_K", bound=str)


def add_counts(counts: tuple[tuple[_K, int], ...], increments: Iterable[tuple[_K, int]]) -> tuple[tuple[_K, int], ...]:
    pairs: Final = tuple(chain(counts, increments))
    keys: Final = sorted({key for key, _ in pairs})
    return tuple((key, sum(n for k, n in pairs if k == key)) for key in keys)


@dataclass(frozen=True, slots=True)
class RequestKey:
    endpoint: str
    provider: str | None
    deployment_hash: str | None
    litellm_status: StatusClass
    provider_status: StatusClass
    litellm_cache_hit: bool
    handled_by_rust: bool
    provider_cache_hit: bool
    stream: bool

    @classmethod
    def of(cls, record: RequestRecord) -> "RequestKey":
        return cls(
            endpoint=record.endpoint,
            provider=record.provider,
            deployment_hash=record.deployment_hash,
            litellm_status=record.litellm_status,
            provider_status=record.provider_status,
            litellm_cache_hit=record.litellm_cache_hit,
            handled_by_rust=record.handled_by_rust,
            provider_cache_hit=record.provider_cache_hit,
            stream=record.stream,
        )


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    request_count: int
    tokens: TokenCounts
    block_count: Histogram
    block_types: tuple[tuple[BlockType, int], ...]
    header_keys: tuple[tuple[str, int], ...]
    provider_attempts: Histogram
    latency_to_headers: Histogram
    latency_to_first_byte: Histogram

    @classmethod
    def of(cls, record: RequestRecord) -> "RequestMetrics":
        return cls(
            request_count=1,
            tokens=record.tokens,
            block_count=Histogram.of(BLOCK_COUNT_BOUNDS, None if record.blocks is None else record.blocks.total),
            block_types=add_counts((), () if record.blocks is None else record.blocks.by_type),
            header_keys=add_counts((), ((key, 1) for key in record.header_keys)),
            provider_attempts=Histogram.of(ATTEMPT_BOUNDS, record.provider_attempts),
            latency_to_headers=Histogram.of(LATENCY_BOUNDS_MS, record.latency_to_headers_ms),
            latency_to_first_byte=Histogram.of(LATENCY_BOUNDS_MS, record.latency_to_first_byte_ms),
        )

    def merge(self, other: "RequestMetrics") -> "RequestMetrics":
        return RequestMetrics(
            request_count=self.request_count + other.request_count,
            tokens=TokenCounts(
                input=self.tokens.input + other.tokens.input,
                output=self.tokens.output + other.tokens.output,
                cache_read=self.tokens.cache_read + other.tokens.cache_read,
            ),
            block_count=self.block_count.merge(other.block_count),
            block_types=add_counts(self.block_types, other.block_types),
            header_keys=add_counts(self.header_keys, other.header_keys),
            provider_attempts=self.provider_attempts.merge(other.provider_attempts),
            latency_to_headers=self.latency_to_headers.merge(other.latency_to_headers),
            latency_to_first_byte=self.latency_to_first_byte.merge(other.latency_to_first_byte),
        )


@dataclass(frozen=True, slots=True)
class AttemptKey:
    provider: str
    deployment_hash: str | None
    provider_status: StatusClass
    stream: bool

    @classmethod
    def of(cls, record: AttemptRecord) -> "AttemptKey":
        return cls(
            provider=record.provider,
            deployment_hash=record.deployment_hash,
            provider_status=record.provider_status,
            stream=record.stream,
        )


@dataclass(frozen=True, slots=True)
class AttemptMetrics:
    attempt_count: int
    latency_to_first_token: Histogram

    @classmethod
    def of(cls, record: AttemptRecord) -> "AttemptMetrics":
        return cls(
            attempt_count=1,
            latency_to_first_token=Histogram.of(LATENCY_BOUNDS_MS, record.latency_to_first_token_ms),
        )

    def merge(self, other: "AttemptMetrics") -> "AttemptMetrics":
        return AttemptMetrics(
            attempt_count=self.attempt_count + other.attempt_count,
            latency_to_first_token=self.latency_to_first_token.merge(other.latency_to_first_token),
        )


@dataclass(frozen=True, slots=True)
class Report:
    instance: InstanceInfo
    window_start: float
    window_end: float
    requests: tuple[tuple[RequestKey, RequestMetrics], ...] = ()
    attempts: tuple[tuple[AttemptKey, AttemptMetrics], ...] = ()
    ui_events: tuple[tuple[UIEvent, int], ...] = ()
    dropped_records: int = 0
    schema_version: int = REPORT_SCHEMA_VERSION


def _enum_value(value: StatusClass | BlockType | UIAction | TelemetryGroup) -> str:
    return value.value


def _histogram_json(histogram: Histogram) -> JsonValue:
    return list(histogram.counts)


def _request_json(key: RequestKey, metrics: RequestMetrics, groups: frozenset[TelemetryGroup]) -> JsonValue:
    success: Final[Mapping[str, JsonValue]] = {
        "endpoint": key.endpoint,
        "litellm_status": _enum_value(key.litellm_status),
        "provider_status": _enum_value(key.provider_status),
        "litellm_cache_hit": key.litellm_cache_hit,
        "handled_by_rust": key.handled_by_rust,
        "stream": key.stream,
        "request_count": metrics.request_count,
        "provider_attempts": _histogram_json(metrics.provider_attempts),
        "latency_to_headers_ms": _histogram_json(metrics.latency_to_headers),
        "latency_to_first_byte_ms": _histogram_json(metrics.latency_to_first_byte),
    }
    tokens: Final[Mapping[str, JsonValue]] = {
        "provider_cache_hit": key.provider_cache_hit,
        "input_tokens": metrics.tokens.input,
        "output_tokens": metrics.tokens.output,
        "cache_read_tokens": metrics.tokens.cache_read,
    }
    taxonomy: Final[Mapping[str, JsonValue]] = {"provider": key.provider, "deployment_hash": key.deployment_hash}
    details: Final[Mapping[str, JsonValue]] = {
        "block_count": _histogram_json(metrics.block_count),
        "block_types": {_enum_value(block_type): n for block_type, n in metrics.block_types},
        "header_keys": dict(metrics.header_keys),
    }
    return {
        **success,
        **(tokens if TelemetryGroup.TOKEN_INFO in groups else {}),
        **(taxonomy if TelemetryGroup.REQUEST_TAXONOMY in groups else {}),
        **(details if TelemetryGroup.EVENT_DETAILS in groups else {}),
    }


def _attempt_json(key: AttemptKey, metrics: AttemptMetrics) -> JsonValue:
    return {
        "provider": key.provider,
        "deployment_hash": key.deployment_hash,
        "provider_status": _enum_value(key.provider_status),
        "stream": key.stream,
        "attempt_count": metrics.attempt_count,
        "latency_to_first_token_ms": _histogram_json(metrics.latency_to_first_token),
    }


def _ui_event_json(event: UIEvent, count: int) -> JsonValue:
    return {"page": event.page, "action": _enum_value(event.action), "target": event.target, "count": count}


def _instance_json(instance: InstanceInfo) -> JsonValue:
    configuration: Final[Mapping[str, JsonValue]] = {"config_keys": sorted(instance.config_keys)}
    return {
        "instance_id": instance.instance_id,
        "litellm_version": instance.litellm_version,
        "groups": sorted(_enum_value(group) for group in instance.groups),
        **(configuration if TelemetryGroup.INSTANCE_CONFIGURATION in instance.groups else {}),
    }


def report_to_json(report: Report) -> Mapping[str, JsonValue]:
    """Only the fields of the report's enabled groups, so a stored or sent report shows exactly what was collected"""
    groups: Final = report.instance.groups
    requests: Final[Mapping[str, JsonValue]] = {
        "requests": [_request_json(key, metrics, groups) for key, metrics in report.requests]
    }
    attempts: Final[Mapping[str, JsonValue]] = {
        "attempts": [_attempt_json(key, metrics) for key, metrics in report.attempts]
    }
    ui_events: Final[Mapping[str, JsonValue]] = {
        "ui_events": [_ui_event_json(event, count) for event, count in report.ui_events]
    }
    return {
        "schema_version": report.schema_version,
        "instance": _instance_json(report.instance),
        "window_start": report.window_start,
        "window_end": report.window_end,
        "dropped_records": report.dropped_records,
        **(requests if TelemetryGroup.REQUEST_SUCCESS in groups else {}),
        **(attempts if TelemetryGroup.REQUEST_TAXONOMY in groups else {}),
        **(ui_events if TelemetryGroup.PAGE_NAVIGATION in groups else {}),
    }
