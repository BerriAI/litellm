import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import ValidationError

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy.spend_tracking.usage_ingestion_endpoints import (
    ExternalUsageRecord,
    KeyAttribution,
    UsageIngestionDeps,
    build_spend_log_payload,
    process_external_usage_record,
)
from litellm.proxy.utils import hash_token

RAW_KEY = "sk-test-batch-dispatch-key"
GENERATED_ID = "generated-uuid-1"
DEFAULT_KEY = KeyAttribution(user_id="u-1", team_id="t-1", organization_id="o-1")


class RecordingDeps:
    def __init__(
        self,
        key: KeyAttribution | None = DEFAULT_KEY,
        existing_ids: frozenset[str] = frozenset(),
        compute_cost_result: float = 0.05,
        compute_cost_error: Exception | None = None,
        reserve_outcome: str = "reserved",
        reserve_raises: Exception | None = None,
    ):
        self._key = key
        self._existing_ids = existing_ids
        self._compute_cost_result = compute_cost_result
        self._compute_cost_error = compute_cost_error
        self._reserve_outcome = reserve_outcome
        self._reserve_raises = reserve_raises
        self.spend_calls: list[dict[str, Any]] = []
        self.reserve_calls: list[dict[str, Any]] = []
        self.compute_cost_calls: list[tuple[Any, str]] = []

    def as_deps(self) -> UsageIngestionDeps:
        async def lookup_key(hashed: str) -> KeyAttribution | None:
            self.looked_up_hashed = hashed
            return self._key

        async def reserve_spend_log(
            record: ExternalUsageRecord,
            request_id: str,
            hashed_token: str,
            key: KeyAttribution,
            cost: float,
        ) -> Any:
            self.reserve_calls.append(
                {
                    "record": record,
                    "request_id": request_id,
                    "hashed_token": hashed_token,
                    "key": key,
                    "cost": cost,
                }
            )
            if self._reserve_raises is not None:
                raise self._reserve_raises
            if self._reserve_outcome == "disabled":
                return "disabled"
            return "duplicate" if request_id in self._existing_ids else "reserved"

        async def record_spend(**kwargs: Any) -> None:
            self.spend_calls.append(kwargs)

        def compute_cost(response: Any, model: str) -> float:
            self.compute_cost_calls.append((response, model))
            if self._compute_cost_error is not None:
                raise self._compute_cost_error
            return self._compute_cost_result

        return UsageIngestionDeps(
            lookup_key=lookup_key,
            reserve_spend_log=reserve_spend_log,
            record_spend=record_spend,
            compute_cost=compute_cost,
            generate_request_id=lambda: GENERATED_ID,
        )


def make_record(**overrides: Any) -> ExternalUsageRecord:
    base: dict[str, Any] = {
        "api_key": RAW_KEY,
        "model": "gpt-4o-mini",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "start_time": datetime(2026, 8, 5, 12, 0, 0, tzinfo=timezone.utc),
    }
    base.update(overrides)
    return ExternalUsageRecord(**base)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_spend_log_payload_matches_funnel_shape():
    record = make_record(idempotency_key="batch-1-line-1", tags=["batch:job-42"], end_user_id="tenant-a")
    payload = build_spend_log_payload(
        record=record,
        request_id="batch-1-line-1",
        hashed_token=hash_token(RAW_KEY),
        key=DEFAULT_KEY,
        cost=0.123,
    )
    assert payload["request_id"] == "batch-1-line-1"
    assert payload["spend"] == 0.123
    assert payload["total_tokens"] == 150
    assert payload["prompt_tokens"] == 100
    assert payload["completion_tokens"] == 50
    assert payload["api_key"] == hash_token(RAW_KEY)
    assert payload["api_key"] != RAW_KEY
    assert payload["team_id"] == "t-1"
    assert payload["organization_id"] == "o-1"
    assert payload["end_user"] == "tenant-a"

    metadata = json.loads(payload["metadata"]) if isinstance(payload["metadata"], str) else payload["metadata"]
    assert metadata["user_api_key"] == hash_token(RAW_KEY)

    request_tags = (
        json.loads(payload["request_tags"]) if isinstance(payload["request_tags"], str) else payload["request_tags"]
    )
    assert request_tags == ["batch:job-42"]


def test_idempotent_record_books_through_atomic_reserve_not_funnel():
    deps = RecordingDeps(compute_cost_error=RuntimeError("pricing must not be consulted"))
    result = run(
        process_external_usage_record(make_record(cost=0.123, idempotency_key="batch-1-line-1"), deps.as_deps())
    )
    assert result.status == "recorded"
    assert result.spend == 0.123
    assert result.request_id == "batch-1-line-1"

    assert len(deps.reserve_calls) == 1
    reserve_call = deps.reserve_calls[0]
    assert reserve_call["request_id"] == "batch-1-line-1"
    assert reserve_call["hashed_token"] == hash_token(RAW_KEY)
    assert reserve_call["cost"] == 0.123
    assert reserve_call["key"] == DEFAULT_KEY

    assert deps.spend_calls == []
    assert deps.compute_cost_calls == []


def test_computed_cost_is_resolved_before_reserving():
    deps = RecordingDeps(compute_cost_result=0.07)
    result = run(process_external_usage_record(make_record(idempotency_key="k-2"), deps.as_deps()))
    assert result.status == "recorded"
    assert result.spend == 0.07
    assert len(deps.compute_cost_calls) == 1
    assert deps.compute_cost_calls[0][1] == "gpt-4o-mini"
    assert deps.reserve_calls[0]["cost"] == 0.07


def test_unpriceable_model_without_explicit_cost_is_error_and_books_nothing():
    deps = RecordingDeps(compute_cost_error=ValueError("unknown model"))
    result = run(process_external_usage_record(make_record(idempotency_key="k-3"), deps.as_deps()))
    assert result.status == "error"
    assert result.spend is None
    assert "explicit cost" in (result.error or "")
    assert deps.reserve_calls == []
    assert deps.spend_calls == []


def test_unknown_key_is_rejected_and_books_nothing():
    deps = RecordingDeps(key=None)
    result = run(process_external_usage_record(make_record(idempotency_key="k-4"), deps.as_deps()))
    assert result.status == "error"
    assert result.error == "api key not found"
    assert deps.reserve_calls == []
    assert deps.spend_calls == []


def test_duplicate_idempotency_key_is_skipped_and_never_rebooks():
    deps = RecordingDeps(existing_ids=frozenset({"k-5"}))
    result = run(process_external_usage_record(make_record(idempotency_key="k-5"), deps.as_deps()))
    assert result.status == "duplicate"
    assert result.request_id == "k-5"
    assert len(deps.reserve_calls) == 1
    assert deps.spend_calls == []


def test_missing_idempotency_key_uses_funnel_with_generated_request_id():
    deps = RecordingDeps()
    result = run(process_external_usage_record(make_record(cost=0.01), deps.as_deps()))
    assert result.status == "recorded"
    assert result.request_id == GENERATED_ID
    assert deps.reserve_calls == []
    assert len(deps.spend_calls) == 1

    call = deps.spend_calls[0]
    assert call["token"] == hash_token(RAW_KEY)
    assert call["user_id"] == "u-1"
    assert call["team_id"] == "t-1"
    assert call["org_id"] == "o-1"
    assert call["response_cost"] == 0.01
    response = call["completion_response"]
    assert response.id == GENERATED_ID
    assert response.usage.total_tokens == 150
    metadata = call["kwargs"]["litellm_params"]["metadata"]
    assert metadata["user_api_key"] == hash_token(RAW_KEY)
    assert metadata["user_api_key"] != RAW_KEY


def test_end_time_defaults_to_start_time_and_end_before_start_rejected():
    deps = RecordingDeps()
    start = datetime(2026, 8, 5, 12, 0, 0, tzinfo=timezone.utc)
    run(process_external_usage_record(make_record(cost=0.01, start_time=start), deps.as_deps()))
    assert deps.spend_calls[0]["start_time"] == start
    assert deps.spend_calls[0]["end_time"] == start

    earlier = datetime(2026, 8, 5, 11, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(ValidationError):
        make_record(start_time=start, end_time=earlier)


def test_record_without_idempotency_key_still_flows_tags_to_funnel_kwargs():
    deps = RecordingDeps()
    result = run(
        process_external_usage_record(
            make_record(cost=0.01, tags=["batch:job-42"], end_user_id="tenant-a"), deps.as_deps()
        )
    )
    assert result.status == "recorded"
    metadata = deps.spend_calls[0]["kwargs"]["litellm_params"]["metadata"]
    assert metadata["tags"] == ["batch:job-42"]
    assert metadata["user_api_key_end_user_id"] == "tenant-a"
    assert deps.spend_calls[0]["end_user_id"] == "tenant-a"


def test_disabled_spend_updates_reports_error_instead_of_fake_recorded():
    deps = RecordingDeps(reserve_outcome="disabled")
    result = run(process_external_usage_record(make_record(cost=0.01, idempotency_key="k-9"), deps.as_deps()))
    assert result.status == "error"
    assert "disabled" in (result.error or "")
    assert result.spend is None
    assert deps.spend_calls == []


def test_failed_booking_is_retry_safe_error_not_permanent_duplicate():
    deps = RecordingDeps(reserve_raises=RuntimeError("db gone mid-tx"))
    result = run(process_external_usage_record(make_record(cost=0.01, idempotency_key="k-10"), deps.as_deps()))
    assert result.status == "error"
    assert "safe to retry" in (result.error or "")
    assert result.spend is None
    assert deps.spend_calls == []


def test_key_hash_resolves_without_raw_key_in_body():
    deps = RecordingDeps()
    record = make_record(cost=0.01, idempotency_key="k-11")
    record = ExternalUsageRecord(**{**record.model_dump(), "api_key": None, "api_key_hash": hash_token(RAW_KEY)})
    result = run(process_external_usage_record(record, deps.as_deps()))
    assert result.status == "recorded"
    assert deps.looked_up_hashed == hash_token(RAW_KEY)
    assert deps.reserve_calls[0]["hashed_token"] == hash_token(RAW_KEY)


def test_exactly_one_key_identifier_required():
    with pytest.raises(ValidationError):
        make_record(api_key=None)
    with pytest.raises(ValidationError):
        make_record(api_key_hash=hash_token(RAW_KEY))
