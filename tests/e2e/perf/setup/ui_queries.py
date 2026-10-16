from __future__ import annotations

import hashlib
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import chain
from typing import Final, Literal

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
    recovered_alias: str | None = None
    excluded_ids: frozenset[str] = frozenset()

    def count(self, document: Document) -> int:
        if self.recovered_alias is not None:
            assert contains_alias(document.root, self.recovered_alias), "SpendLog key metadata was not recovered"
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
            assert any(_positive(row, name) for row in value), f"Expected positive {name}"
        return len(value)


def _positive(row: JsonValue, name: str) -> bool:
    assert isinstance(row, dict), "Expected object row"
    number: Final = row.get(name)
    assert isinstance(number, (int, float)), f"Expected numeric {name}"
    return number > 0


def contains_alias(value: JsonValue, alias: str) -> bool:
    if isinstance(value, dict):
        return value.get("key_alias") == alias or any(contains_alias(child, alias) for child in value.values())
    return isinstance(value, list) and any(contains_alias(child, alias) for child in value)


def orphan_key(name: str) -> str:
    return hashlib.sha256(f"ui-perf-orphan-{name}".encode()).hexdigest()


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
    surface: Literal["current", "old_usage", "liteadmin", "key_recovery"] = "current"
    body: BaseModel | None = None
    cursor: Literal["none", "session", "prompt_cache"] = "none"


class Sample(BaseModel):
    model_config = ConfigDict(frozen=True)

    seconds: float
    status: int
    rows: int = 0
    error: str = ""


class Timings(BaseModel):
    model_config = ConfigDict(frozen=True)

    first_seconds: float
    repeated_samples: int
    repeated_p50_seconds: float | None
    repeated_p95_seconds: float | None
    repeated_p99_seconds: float | None
    repeated_max_seconds: float | None


def summarize(samples: tuple[Sample, ...]) -> Timings:
    assert samples, "No UI requests were measured"
    repeated: Final = tuple(sorted(sample.seconds for sample in samples[1:] if not sample.error))
    return Timings(
        first_seconds=samples[0].seconds,
        repeated_samples=len(repeated),
        repeated_p50_seconds=statistics.median(repeated) if repeated else None,
        repeated_p95_seconds=repeated[math.ceil(len(repeated) * 0.95) - 1] if len(repeated) >= 20 else None,
        repeated_p99_seconds=repeated[math.ceil(len(repeated) * 0.99) - 1] if len(repeated) >= 100 else None,
        repeated_max_seconds=max(repeated) if repeated else None,
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


@step("Measure UI query {case.name} at {case.path}")
def issue_query(transport: HttpTransport, case: UiQueryCase, *, timeout: float) -> Result[Document]:
    if case.body is not None:
        return transport.post(
            case.path, headers=transport.master, json=case.body, response_type=Document, timeout=timeout
        )
    return transport.get(
        case.path, headers=transport.master, params=case.params, response_type=Document, timeout=timeout
    )


def database_failures(text: str) -> float:
    families: Final = tuple(text_string_to_metric_families(text))
    required: Final = frozenset(("litellm_postgres_failed_requests", "litellm_batch_write_to_db_failed_requests"))
    missing: Final = required - frozenset(family.name for family in families)
    assert not missing, f"Missing PostgreSQL failure counters {missing}"
    all_samples: Final = tuple(chain.from_iterable(family.samples for family in families))
    names: Final = frozenset(f"{name}_total" for name in required)
    values: Final = tuple(sample.value for sample in all_samples if sample.name in names)
    assert all(math.isfinite(value) and value >= 0 for value in values), "Invalid PostgreSQL failure counter"
    return sum(values)
