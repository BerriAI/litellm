"""The wire shape of one telemetry report: what an exporter sends or stores for one window"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

from litellm.telemetry.histogram import ATTEMPT_BOUNDS, BLOCK_COUNT_BOUNDS, LATENCY_BOUNDS_MS, Histogram
from litellm.telemetry.records import (
    AttemptRecord,
    BlockType,
    InstanceInfo,
    RequestRecord,
    StatusClass,
    TelemetryGroup,
    UIAction,
    UIEvent,
)

REPORT_SCHEMA_VERSION: Final = 1

JsonValue: TypeAlias = "str | int | float | bool | None | Sequence[JsonValue] | Mapping[str, JsonValue]"


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


@dataclass(slots=True)
class RequestMetrics:
    """One request row's running totals, added to in place so each record costs no allocation"""

    request_count: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    block_count: Histogram
    block_types: dict[BlockType, int]  # mutable-ok: incremented in place once per record on the request path
    header_keys: dict[str, int]  # mutable-ok: incremented in place once per record on the request path
    provider_attempts: Histogram
    latency_to_headers: Histogram
    latency_to_first_byte: Histogram

    @classmethod
    def empty(cls) -> "RequestMetrics":
        return cls(
            request_count=0,
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            block_count=Histogram.empty(BLOCK_COUNT_BOUNDS),
            block_types={},
            header_keys={},
            provider_attempts=Histogram.empty(ATTEMPT_BOUNDS),
            latency_to_headers=Histogram.empty(LATENCY_BOUNDS_MS),
            latency_to_first_byte=Histogram.empty(LATENCY_BOUNDS_MS),
        )

    def add(self, record: RequestRecord) -> None:
        self.request_count += 1
        self.input_tokens += record.tokens.input
        self.output_tokens += record.tokens.output
        self.cache_read_tokens += record.tokens.cache_read
        self.provider_attempts.add(record.provider_attempts)
        self.latency_to_headers.add(record.latency_to_headers_ms)
        self.latency_to_first_byte.add(record.latency_to_first_byte_ms)
        for key in record.header_keys:
            self.header_keys[key] = self.header_keys.get(key, 0) + 1
        if record.blocks is None:
            return
        self.block_count.add(record.blocks.total)
        for block_type, count in record.blocks.by_type:
            self.block_types[block_type] = self.block_types.get(block_type, 0) + count

    def add_metrics(self, other: "RequestMetrics") -> None:
        self.request_count += other.request_count
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.block_count.add_histogram(other.block_count)
        self.provider_attempts.add_histogram(other.provider_attempts)
        self.latency_to_headers.add_histogram(other.latency_to_headers)
        self.latency_to_first_byte.add_histogram(other.latency_to_first_byte)
        for block_type, count in other.block_types.items():
            self.block_types[block_type] = self.block_types.get(block_type, 0) + count
        for key, count in other.header_keys.items():
            self.header_keys[key] = self.header_keys.get(key, 0) + count


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


@dataclass(slots=True)
class AttemptMetrics:
    attempt_count: int
    latency_to_first_token: Histogram

    @classmethod
    def empty(cls) -> "AttemptMetrics":
        return cls(attempt_count=0, latency_to_first_token=Histogram.empty(LATENCY_BOUNDS_MS))

    def add(self, record: AttemptRecord) -> None:
        self.attempt_count += 1
        self.latency_to_first_token.add(record.latency_to_first_token_ms)

    def add_metrics(self, other: "AttemptMetrics") -> None:
        self.attempt_count += other.attempt_count
        self.latency_to_first_token.add_histogram(other.latency_to_first_token)


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
    return {"counts": list(histogram.counts), "invalid": histogram.invalid}


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
        "input_tokens": metrics.input_tokens,
        "output_tokens": metrics.output_tokens,
        "cache_read_tokens": metrics.cache_read_tokens,
    }
    taxonomy: Final[Mapping[str, JsonValue]] = {"provider": key.provider, "deployment_hash": key.deployment_hash}
    details: Final[Mapping[str, JsonValue]] = {
        "block_count": _histogram_json(metrics.block_count),
        "block_types": {_enum_value(block_type): n for block_type, n in sorted(metrics.block_types.items())},
        "header_keys": dict(sorted(metrics.header_keys.items())),
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
