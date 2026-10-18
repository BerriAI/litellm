from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import chain
from typing import Final, Literal
from urllib.parse import urlencode

from e2e_http import (
    NetworkError,
    RateLimitedError,
    Result,
    Success,
    UnauthorizedError,
    UnknownApiError,
    ValidationError,
    without_retries,
)
from e2e_metadata import step
from prometheus_client.parser import text_string_to_metric_families
from pydantic import BaseModel, ConfigDict, JsonValue, RootModel
from transport import HttpTransport


class Query(BaseModel):
    model_config = ConfigDict(frozen=True)


class Document(RootModel[JsonValue]):
    model_config = ConfigDict(frozen=True)


@dataclass(frozen=True, slots=True)
class Contract:
    collection: tuple[str, ...] = ()
    row_fields: tuple[str, ...] = ()
    positive: tuple[str, ...] = ()
    object_fields: tuple[str, ...] = ()
    excluded_ids: frozenset[str] = frozenset()

    def count(self, document: Document) -> int:
        value: Final = descend(document.root, self.collection)
        if self.object_fields:
            assert isinstance(value, dict), "Expected a JSON object"
            assert set(self.object_fields) <= value.keys(), f"Missing fields {self.object_fields}"
            return 1
        assert isinstance(value, list) and value, f"Empty/missing collection {self.collection}, check the fixture"
        for row in value:
            if self.excluded_ids:
                assert isinstance(row, dict) and isinstance(row_id := row.get("request_id"), str), "Missing request_id"
                assert row_id not in self.excluded_ids, "Repeated page-one row"
            if self.row_fields or self.positive:
                assert isinstance(row, dict), "Expected object rows"
                assert set(self.row_fields) <= row.keys(), f"Missing row fields {self.row_fields}"
        for name in self.positive:
            assert any(tuple(_positive(row, name) for row in value)), f"Expected positive {name}"
        return len(value)


def _positive(row: JsonValue, name: str) -> bool:
    assert isinstance(row, dict), "Expected object row"
    number: Final = row.get(name)
    assert isinstance(number, (int, float)) and not isinstance(number, bool), f"Expected numeric {name}"
    assert math.isfinite(number), f"Expected finite {name}"
    return number > 0


def descend(value: JsonValue, path: tuple[str, ...]) -> JsonValue:
    if not path:
        return value
    assert isinstance(value, dict) and path[0] in value, f"Missing response path {path}"
    return descend(value[path[0]], path[1:])


@dataclass(frozen=True, slots=True)
class UiQueryCase:
    name: str
    path: str
    params: BaseModel
    contract: Contract
    surface: Literal["current"] = "current"
    body: BaseModel | None = None
    cursor: Literal["none", "session"] = "none"


class Sample(BaseModel):
    model_config = ConfigDict(frozen=True)

    seconds: float
    status: int
    rows: int = 0
    error: str = ""


class Timings(BaseModel):
    model_config = ConfigDict(frozen=True)

    first_seconds: float | None
    repeated_samples: int
    repeated_p50_seconds: float | None
    repeated_p95_seconds: float | None
    repeated_p99_seconds: float | None
    repeated_max_seconds: float | None


def summarize(samples: tuple[Sample, ...]) -> Timings:
    assert samples, "No UI requests were measured"
    repeated: Final = tuple(sorted(sample.seconds for sample in samples[1:] if not sample.error))
    return Timings(
        first_seconds=samples[0].seconds if not samples[0].error else None,
        repeated_samples=len(repeated),
        repeated_p50_seconds=statistics.median(repeated) if repeated else None,
        repeated_p95_seconds=repeated[math.ceil(len(repeated) * 0.95) - 1] if len(repeated) >= 20 else None,
        repeated_p99_seconds=repeated[math.ceil(len(repeated) * 0.99) - 1] if len(repeated) >= 100 else None,
        repeated_max_seconds=max(repeated) if repeated else None,
    )


def error_code(sample: Sample) -> str:
    if not sample.error:
        return ""
    if sample.status > 0 and sample.status != 200:
        return f"http_{sample.status}"
    if sample.error in ("timeout", "network"):
        return sample.error
    if sample.error == "Response is not valid JSON":
        return "invalid_json"
    return "contract"


def ui_junit_properties(
    samples: tuple[Sample, ...], prerequisite: Sample | None = None
) -> tuple[tuple[str, float], ...]:
    requests: Final = (prerequisite, *samples) if prerequisite is not None and not prerequisite.error else samples
    codes: Final = tuple(error_code(sample) for sample in requests if sample.error)
    timings: Final = summarize(samples)
    latency: Final = (
        ("first_seconds", timings.first_seconds),
        ("repeated_seconds_p50", timings.repeated_p50_seconds),
        ("repeated_seconds_p95", timings.repeated_p95_seconds),
        ("repeated_seconds_p99", timings.repeated_p99_seconds),
        ("repeated_seconds_max", timings.repeated_max_seconds),
    )
    return (
        ("perf_ui_requests", len(requests)),
        ("perf_ui_failures", len(codes)),
        ("perf_ui_repeated_samples", timings.repeated_samples),
        *((f"perf_ui_error_{code}", codes.count(code)) for code in sorted(frozenset(codes))),
        *((f"perf_ui_{name}", value) for name, value in latency if value is not None),
    )


def _failure(result: Result[Document], seconds: float) -> Sample:
    match result:
        case Success(status_code=status):
            return Sample(seconds=seconds, status=status, error=f"Unexpected HTTP {status}")
        case UnknownApiError(status_code=status):
            return Sample(seconds=seconds, status=status, error=f"HTTP {status}")
        case UnauthorizedError():
            return Sample(seconds=seconds, status=401, error="Unauthorized")
        case RateLimitedError():
            return Sample(seconds=seconds, status=429, error="Rate limited")
        case NetworkError(message=message):
            return Sample(seconds=seconds, status=-1, error="timeout" if "timed out" in message.lower() else "network")
        case ValidationError():
            return Sample(seconds=seconds, status=-1, error="Response is not valid JSON")


def measure(
    issue: Callable[[], Result[Document]],
    contract: Contract,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[Sample, Document | None]:
    started: Final = clock()
    with without_retries():
        result: Final = issue()
    seconds: Final = clock() - started
    match result:
        case Success(status_code=200, data=document):
            try:
                count: Final = contract.count(document)
            except AssertionError as error:
                return Sample(seconds=seconds, status=200, error=str(error)), document
            return Sample(seconds=seconds, status=200, rows=count), document
        case _:
            return _failure(result, seconds), None


def query_path(case: UiQueryCase) -> str:
    query: Final = urlencode(case.params.model_dump(by_alias=True, exclude_none=True), doseq=True)
    return f"{case.path}?{query}" if query else case.path


@step("Measure UI query {case.name} at {case.path}")
def issue_query(transport: HttpTransport, case: UiQueryCase, *, timeout: float) -> Result[Document]:
    if case.body is not None:
        return transport.post(
            case.path, headers=transport.master, json=case.body, response_type=Document, timeout=timeout
        )
    return transport.get(
        query_path(case), headers=transport.master, params=Query(), response_type=Document, timeout=timeout
    )


def database_failures(text: str) -> float:
    families: Final = tuple(text_string_to_metric_families(text))
    required: Final = frozenset(("litellm_postgres_failed_requests", "litellm_batch_write_to_db_failed_requests"))
    missing: Final = required - frozenset(family.name for family in families)
    all_samples: Final = tuple(chain.from_iterable(family.samples for family in families))
    has_activity: Final = any(
        sample.name == "litellm_postgres_total_requests_total" and math.isfinite(sample.value) and sample.value > 0
        for sample in all_samples
    )
    assert not missing or has_activity, f"Missing PostgreSQL failure counters {missing}"
    names: Final = frozenset(f"{name}_total" for name in required)
    values: Final = tuple(sample.value for sample in all_samples if sample.name in names)
    assert all(math.isfinite(value) and value >= 0 for value in values), "Invalid PostgreSQL failure counter"
    return sum(values)
