import json
import re
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Final
from unittest.mock import AsyncMock

import httpx
import pytest
from prisma import Json, Prisma
from pydantic import InstanceOf, TypeAdapter

from litellm.rust_bridge.trace.queries import TraceSQLResponse
from litellm.rust_bridge.trace.storage import Tenant, span_rows
from litellm.tracing.types import SpendLogRecord
from scripts.seed_tracing_fixtures import (
    JSON,
    TRACE_FIXTURES,
    bulk_span_rows,
    fixture_capture,
    fixture_replays,
    postgres_row,
    rebase,
    rebase_spend,
    response_ids,
    response_pattern,
    seed_arguments,
    seed_copy,
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
    export: Final = JSON.validate_json((TRACE_FIXTURES / "deepagents_swarm.json").read_bytes())
    original: Final = span_rows(json.dumps(export).encode(), "application/json")
    spend_rows: Final = dict(spend_fixtures())["deepagents_swarm"]
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


def test_postgres_rows_preserve_clickhouse_cost_identity_and_payloads() -> None:
    spends: Final = dict(spend_fixtures())["deepagents_swarm"]

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
            spend["session_id"],
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
@pytest.mark.parametrize("name,spends", spend_fixtures())
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
        if before["litellm_call_id"]:
            assert after["litellm_call_id"] != before["litellm_call_id"]
        identities: Final = frozenset(f"provider_response:{identity}" for identity in response_ids((after,))) | {
            f"litellm_request:{after['litellm_call_id']}"
        }
        assert bool(identities & keys) is capture.spend_linked


@pytest.mark.parametrize("call_id", (None, "gateway"))
def test_spend_fixture_loading_preserves_gateway_ids_and_defaults_legacy_rows(
    tmp_path: Path, call_id: str | None
) -> None:
    original: Final = dict(spend_fixtures())["deepagents_swarm"][0]
    fields: Final = {key: value for key, value in original.items() if key != "litellm_call_id"}
    supplied: Final = fields if call_id is None else {**fields, "litellm_call_id": call_id}
    (tmp_path / "example_spend_logs.jsonl").write_text(json.dumps(supplied) + "\n")
    loaded: Final = spend_fixtures(tmp_path)
    assert loaded == (("example", ({**original, "litellm_call_id": call_id or ""},)),)


@pytest.mark.requires_rust_extension
def test_bulk_export_preserves_all_spans_and_disjoint_copy_ids() -> None:
    first: Final = fixture_replays(TRACE_FIXTURES, 1_800_000_000_000, "copy-1", re.compile(r"(?!)"))
    second: Final = fixture_replays(TRACE_FIXTURES, 1_800_000_001_000, "copy-2", re.compile(r"(?!)"))
    merged: Final = bulk_span_rows(first + second, Tenant("", ""))
    separate: Final = tuple(
        span for replay in first + second for span in span_rows(json.dumps(replay.export).encode(), "application/json")
    )
    assert tuple(merged) == separate
    first_ids: Final = frozenset(span["TraceId"] for span in bulk_span_rows(first, Tenant("", "")))
    second_ids: Final = frozenset(span["TraceId"] for span in bulk_span_rows(second, Tenant("", "")))
    assert first_ids.isdisjoint(second_ids)


@pytest.mark.requires_rust_extension
@pytest.mark.asyncio
async def test_first_copy_stamps_the_authenticated_tenant_and_writes_both_stores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.rust_bridge.trace.storage import ClickHouseStorage

    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-local")
    fixtures: Final = spend_fixtures()
    pattern: Final = response_pattern(tuple(chain.from_iterable(rows for _, rows in fixtures)))
    replays: Final = fixture_replays(TRACE_FIXTURES, 1_800_000_000_000, "first", pattern)
    storage: Final = AsyncMock(spec=ClickHouseStorage)
    storage.query_sql.return_value = TraceSQLResponse.model_validate(
        {
            "meta": (),
            "data": [{"team_id": "local-team", "api_key": "local-hash", "user": "admin"}],
            "rows": 1,
            "statistics": {"elapsed": 0, "rows_read": 1, "bytes_read": 1},
        }
    )
    database: Final = AsyncMock(spec=Prisma, litellm_spendlogs=AsyncMock())
    client: Final = AsyncMock(spec=httpx.AsyncClient)
    client.post.return_value = httpx.Response(200, request=httpx.Request("POST", "http://proxy/v1/traces"))
    captures: Final = await seed_copy(client, storage, database, replays, fixtures, pattern)
    assert tuple(JSON.validate_json(call.kwargs["content"]) for call in client.post.call_args_list) == tuple(
        replay.export for replay in replays
    )
    rows: Final = tuple(chain.from_iterable(rows for _, rows in captures))
    assert {name for name, _ in captures} == {name for name, _ in fixtures}
    assert storage.insert_rows.call_args.args == ("spend_logs", rows)
    assert len(rows) == sum(len(original) for _, original in fixtures)
    assert all((row["team_id"], row["api_key"], row["user"]) == ("local-team", "local-hash", "admin") for row in rows)
    saved: Final = database.litellm_spendlogs.create_many.call_args.kwargs["data"]
    assert tuple(row["request_id"] for row in saved) == tuple(row["request_id"] for row in rows)
    assert tuple(row["spend"] for row in saved) == tuple(row["spend"] for row in rows)


def test_seed_cli_rejects_nonpositive_copies() -> None:
    with pytest.raises(SystemExit) as error:
        seed_arguments(["--copies", "0"])
    assert error.value.code == 2
    assert seed_arguments(["--profile", "large", "--copies", "5"]).copies == 5


@pytest.mark.parametrize("timeout", ("0", "-1", "inf", "nan"))
def test_seed_cli_rejects_invalid_http_timeouts(timeout: str) -> None:
    with pytest.raises(SystemExit) as error:
        seed_arguments(["--timeout-seconds", timeout])
    assert error.value.code == 2
