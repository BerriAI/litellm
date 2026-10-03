import json
import re
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Final

import pytest
from prisma import Json
from pydantic import InstanceOf, TypeAdapter

from litellm.rust_bridge.trace.storage import span_rows
from litellm.tracing.types import SpendLogRecord
from scripts.seed_tracing_fixtures import (
    JSON,
    SPEND_FIXTURE,
    SPEND_ROWS,
    TRACE_FIXTURES,
    fixture_capture,
    fixture_replays,
    postgres_row,
    rebase,
    rebase_spend,
    response_ids,
    response_pattern,
    seed_id,
    spend_fixtures,
    timestamps,
)

CALL_KEYS: Final = TypeAdapter(tuple[str, ...])
DATETIMES: Final = TypeAdapter(tuple[datetime, datetime])
SPAN_IDENTITY: Final = TypeAdapter(tuple[str, str, str, int])
JSON_FIELDS: Final[TypeAdapter[tuple[Json, Json, Json]]] = TypeAdapter(
    tuple[InstanceOf[Json], InstanceOf[Json], InstanceOf[Json]]
)


@pytest.mark.requires_rust_extension
@pytest.mark.parametrize(
    "path",
    sorted(TRACE_FIXTURES.glob("*.json")),
    ids=tuple(path.stem for path in sorted(TRACE_FIXTURES.glob("*.json"))),
)
def test_all_fixture_replays_are_recent_and_preserve_spans(path: Path) -> None:
    export: Final = JSON.validate_json(path.read_bytes())
    now_ms: Final = max(timestamps(export)) // 1_000_000 + 86_400_000
    replays: Final = fixture_replays(TRACE_FIXTURES, now_ms, "all-fixtures", re.compile(r"(?!)"))
    replay: Final = next(item for item in replays if item.name == path.stem)
    original: Final = span_rows(path.read_bytes(), "application/json")
    replayed: Final = span_rows(json.dumps(replay.export).encode(), "application/json")
    group: Final = tuple(item for item in replays if item.namespace == replay.namespace)

    assert max(max(timestamps(item.export)) for item in group) // 1_000_000 == now_ms - 1000
    assert len(frozenset(item.offset_ms for item in group)) == 1
    assert tuple(timestamps(replay.export)) == tuple(
        timestamp + replay.offset_ms * 1_000_000 for timestamp in timestamps(export)
    )
    for before, after in zip(original, replayed, strict=True):
        trace_id, span_id, parent_id, timestamp = SPAN_IDENTITY.validate_python(
            (before["TraceId"], before["SpanId"], before["ParentSpanId"], before["Timestamp"])
        )
        assert after["TraceId"] == seed_id(trace_id, replay.namespace, 32)
        assert after["SpanId"] == seed_id(span_id, replay.namespace, 16)
        assert after["ParentSpanId"] == seed_id(parent_id, replay.namespace, 16)
        assert after["Timestamp"] == timestamp + replay.offset_ms * 1_000_000
        assert (after["Duration"], after["InputTokens"], after["OutputTokens"], after["StatusCode"]) == (
            before["Duration"],
            before["InputTokens"],
            before["OutputTokens"],
            before["StatusCode"],
        )
    if path.stem.startswith("query_"):
        assert all(item.namespace == replay.namespace for item in replays if item.name.startswith("query_"))
    else:
        assert all(item.namespace != replay.namespace for item in replays if item.name != path.stem)


@pytest.mark.requires_rust_extension
def test_replay_preserves_trace_topology_usage_and_event_timing() -> None:
    export: Final = JSON.validate_json((TRACE_FIXTURES / "deeplite_swarm.json").read_bytes())
    original: Final = span_rows(json.dumps(export).encode(), "application/json")
    spend_rows: Final = SPEND_ROWS.validate_python(
        tuple(json.loads(line) for line in SPEND_FIXTURE.read_text().splitlines())
    )
    pattern: Final = re.compile("|".join(re.escape(row["response_id"]) for row in spend_rows))
    shifted: Final = rebase(export, 123_000_000, "first-run", pattern)
    replayed: Final = span_rows(json.dumps(shifted).encode(), "application/json")
    other_run: Final = span_rows(
        json.dumps(rebase(export, 123_000_000, "second-run", pattern)).encode(), "application/json"
    )
    span_ids: Final = {before["SpanId"]: after["SpanId"] for before, after in zip(original, replayed, strict=True)}

    assert tuple(timestamps(shifted)) == tuple(timestamp + 123_000_000 for timestamp in timestamps(export))
    assert {span["TraceId"] for span in original}.isdisjoint(span["TraceId"] for span in replayed)
    assert {span["TraceId"] for span in replayed}.isdisjoint(span["TraceId"] for span in other_run)
    for before, after in zip(original, replayed, strict=True):
        assert after["ParentSpanId"] == span_ids.get(before["ParentSpanId"], "")
        assert after["Timestamp"] == before["Timestamp"] + 123_000_000
        assert after["Duration"] == before["Duration"]
        assert after["InputTokens"] == before["InputTokens"]
        assert after["OutputTokens"] == before["OutputTokens"]
        assert after["StatusCode"] == before["StatusCode"]
        assert after["LiteLLMRequestId"] == (
            f"seed-first-run-{before['LiteLLMRequestId']}" if before["LiteLLMRequestId"] else ""
        )


@pytest.mark.requires_rust_extension
def test_paired_fixture_joins_every_successful_llm_span_after_replay() -> None:
    export: Final = JSON.validate_json((TRACE_FIXTURES / "deeplite_swarm.json").read_bytes())
    spends: Final = SPEND_ROWS.validate_python(
        tuple(json.loads(line) for line in SPEND_FIXTURE.read_text().splitlines())
    )
    pattern: Final = re.compile("|".join(re.escape(row["response_id"]) for row in spends))
    replays: Final = fixture_replays(TRACE_FIXTURES, max(timestamps(export)) // 1_000_000 + 1123, "paired-run", pattern)
    replay: Final = next(item for item in replays if item.name == "deeplite_swarm")
    rebased_spends: Final = rebase_spend(spends, replay.offset_ms, replay.namespace, pattern)
    spans: Final = span_rows(json.dumps(replay.export).encode(), "application/json")
    llm_spans: Final = tuple(span for span in spans if span["ObservationType"] == "llm")
    by_response: Final = {row["response_id"]: row for row in rebased_spends}

    assert len(by_response) == len(llm_spans) == len(rebased_spends)
    assert frozenset(by_response) == frozenset(span["LiteLLMRequestId"] for span in llm_spans)
    for span, spend in ((span, by_response[span["LiteLLMRequestId"]]) for span in llm_spans):
        assert spend["request_id"] == span["LiteLLMRequestId"]
        assert spend["trace_id"] == spend["session_id"] == span["TraceId"]
        assert spend["span_id"] == span["SpanId"]
        assert spend["start_time"] == span["Timestamp"] // 1_000_000
        assert spend["end_time"] == (span["Timestamp"] + span["Duration"]) // 1_000_000
        assert spend["prompt_tokens"] == span["InputTokens"]
        assert spend["completion_tokens"] == span["OutputTokens"]
        assert spend["total_tokens"] == spend["prompt_tokens"] + spend["completion_tokens"]
        assert json.loads(spend["response"])["id"] == spend["response_id"]
        assert json.loads(spend["response"])["usage"]["total_tokens"] == spend["total_tokens"]
        assert json.loads(spend["metadata"])["synthetic_spend"] is True


def test_postgres_rows_preserve_clickhouse_cost_identity_and_payloads() -> None:
    spends: Final = SPEND_ROWS.validate_python(
        tuple(json.loads(line) for line in SPEND_FIXTURE.read_text().splitlines())
    )

    for spend, postgres in ((spend, postgres_row(spend)) for spend in spends):
        start_time, end_time = DATETIMES.validate_python((postgres["startTime"], postgres["endTime"]))
        messages, response, proxy_request = JSON_FIELDS.validate_python(
            (postgres["messages"], postgres["response"], postgres["proxy_server_request"])
        )
        assert postgres["request_id"] == spend["response_id"]
        assert (postgres["api_key"], postgres["team_id"], postgres["user"], postgres["session_id"]) == (
            spend["api_key"],
            spend["team_id"],
            spend["user"],
            spend["trace_id"],
        )
        assert postgres["spend"] == spend["spend"]
        assert postgres["total_tokens"] == spend["prompt_tokens"] + spend["completion_tokens"]
        assert round(start_time.timestamp() * 1000) == spend["start_time"]
        assert round(end_time.timestamp() * 1000) == spend["end_time"]
        assert postgres["request_duration_ms"] == spend["end_time"] - spend["start_time"]
        assert JSON.validate_python(getattr(messages, "data")) == JSON.validate_json(spend["messages"])
        assert JSON.validate_python(getattr(response, "data")) == JSON.validate_json(spend["response"])
        assert JSON.validate_python(getattr(proxy_request, "data")) is None


@pytest.mark.requires_rust_extension
@pytest.mark.parametrize("name,spends", tuple(item for item in spend_fixtures() if item[0] != "deeplite_swarm"))
def test_captured_spend_replay_preserves_real_cost_and_call_identity(
    name: str, spends: tuple[SpendLogRecord, ...]
) -> None:
    export: Final = JSON.validate_json((TRACE_FIXTURES / f"{name}.json").read_bytes())
    pattern: Final = response_pattern(spends)
    offset_ms: Final = 1123
    namespace: Final = f"captured-{name}"
    shifted: Final = rebase(export, offset_ms * 1_000_000, namespace, pattern)
    spans: Final = span_rows(json.dumps(shifted).encode(), "application/json")
    replayed: Final = rebase_spend(spends, offset_ms, namespace, pattern)
    keys: Final = frozenset(chain.from_iterable(CALL_KEYS.validate_python(span["CallKeys"]) for span in spans))
    capture: Final = fixture_capture(name, replayed[0])

    assert capture.trace_id in frozenset(span["TraceId"] for span in spans)
    for before, after in zip(spends, replayed, strict=True):
        assert after["spend"] == before["spend"]
        assert (after["prompt_tokens"], after["completion_tokens"], after["total_tokens"]) == (
            before["prompt_tokens"],
            before["completion_tokens"],
            before["total_tokens"],
        )
        assert after["request_id"] != before["request_id"]
        assert after["start_time"] == before["start_time"] + offset_ms
        assert after["end_time"] == before["end_time"] + offset_ms
        assert bool(frozenset(f"provider_response:{identity}" for identity in response_ids((after,))) & keys) is (
            capture.spend_linked
        )
