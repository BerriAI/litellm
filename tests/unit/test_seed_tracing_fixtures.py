import json
import re
from datetime import datetime
from typing import Final

import pytest
from prisma import Json
from pydantic import InstanceOf, TypeAdapter

from litellm.tracing.decode import decode_otlp
from scripts.seed_tracing_fixtures import (
    JSON,
    SPEND_FIXTURE,
    SPEND_ROWS,
    TRACE_FIXTURES,
    postgres_row,
    rebase,
    rebase_spend,
    timestamps,
)

DATETIMES: Final = TypeAdapter(tuple[datetime, datetime])
JSON_FIELDS: Final[TypeAdapter[tuple[Json, Json, Json]]] = TypeAdapter(
    tuple[InstanceOf[Json], InstanceOf[Json], InstanceOf[Json]]
)


@pytest.mark.requires_rust_extension
def test_replay_preserves_trace_topology_usage_and_event_timing() -> None:
    export: Final = JSON.validate_json((TRACE_FIXTURES / "deeplite_swarm.json").read_bytes())
    original: Final = decode_otlp(json.dumps(export).encode(), "application/json")
    spend_rows: Final = SPEND_ROWS.validate_python(
        tuple(json.loads(line) for line in SPEND_FIXTURE.read_text().splitlines())
    )
    pattern: Final = re.compile("|".join(re.escape(row["response_id"]) for row in spend_rows))
    shifted: Final = rebase(export, 123_000_000, "first-run", pattern)
    replayed: Final = decode_otlp(json.dumps(shifted).encode(), "application/json")
    other_run: Final = decode_otlp(
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
    rebased_spends: Final = rebase_spend(spends, 123, "paired-run", pattern)
    spans: Final = decode_otlp(
        json.dumps(rebase(export, 123_000_000, "paired-run", pattern)).encode(), "application/json"
    )
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
