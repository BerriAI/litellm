import json
import math
import re
import time
from collections.abc import Generator, Iterator
from contextlib import closing
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import Final

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm.rust_bridge import _native  # noqa: F401
from litellm.rust_bridge.trace.generated.models import TraceQueryHelp
from litellm.rust_bridge.trace.generated.types import Trace
from litellm.rust_bridge.trace.storage import ClickHouseStorage, TraceStorageConfig, span_rows
from litellm.tracing import Tenant, TraceReceiver
from litellm.tracing.types import SpendLogRecord
from seed_tracing_fixtures import (
    TRACE,
    TRACE_FIXTURES,
    Copies,
    FixtureReplay,
    bulk_span_rows,
    copied_trace_id,
    copy_clickhouse,
    fixture_capture,
    fixture_replays,
    long_sessions,
    rebase_spend,
    response_pattern,
    spend_fixtures,
)
from tests._support.native_isolation import isolate_ocr_test_state as isolate_ocr_test_state
from tests._support.recording_server import RecordingServer, ResponseSpec
from tests.integration._support.native.clickhouse import clickhouse_service


pytestmark = pytest.mark.requires_rust_extension


QUERY_ROWS: Final = TypeAdapter(tuple[dict[str, JsonValue], ...])


class CapturedSpendRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    request_id: str
    spend: float
    prompt_tokens: int
    completion_tokens: int


class CapturedSpendQuery(BaseModel):
    model_config = ConfigDict(frozen=True)
    data: tuple[CapturedSpendRow, ...]


@pytest.mark.parametrize(
    ("role", "user_id", "expected_status"),
    (
        ("proxy_admin", None, 200),
        ("proxy_admin_viewer", None, 200),
        ("internal_user", "user", 200),
        ("internal_user", None, 403),
    ),
)
def test_trace_sql_endpoint_returns_data_only_and_enforces_ownership(
    recording_server: RecordingServer, role: str, user_id: str | None, expected_status: int
) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.authorization_dependencies import get_log_team_lookup
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.tracing_endpoints import provide_receiver, provide_trace_query_secret, router

    envelope: Final = {
        "meta": [{"name": "answer", "type": "UInt8"}],
        "data": [{"answer": 42}],
        "rows": 1,
        "statistics": {"elapsed": 0.01, "rows_read": 1, "bytes_read": 1},
        "rows_before_limit_at_least": 1,
    }
    recording_server.expected_requests = 12 if expected_status == 200 else 0
    if expected_status == 200:
        for _ in range(11):
            recording_server.enqueue(ResponseSpec(body=""))
        recording_server.enqueue(ResponseSpec(body=envelope))
    storage: Final = ClickHouseStorage(TraceStorageConfig(recording_server.base_url, "trace_test"))
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[provide_trace_query_secret] = lambda: "test-master-secret"
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=role, user_id=user_id, token="test")
    app.dependency_overrides[provide_receiver] = lambda: TraceReceiver(storage)

    async def permitted_teams(auth: UserAPIKeyAuth) -> tuple[str, ...]:
        return ()

    app.dependency_overrides[get_log_team_lookup] = lambda: permitted_teams
    with TestClient(app) as client:
        result: Final = client.post("/v1/traces/query", json={"sql": "SELECT 42 AS answer"})
        assert result.status_code == expected_status, result.text
        if expected_status == 403:
            assert result.json() == {"detail": "Not allowed to view logs"}
            return
        assert result.json() == {"data": envelope["data"]}
        assert recording_server.requests[-1].raw_body == b"SELECT 42 AS answer"
        assert client.post("/v1/traces/query", json={"sql": "  "}).status_code == 400
        assert client.post("/v1/traces/query", json={}).status_code == 422


@pytest.mark.parametrize("discovery_fails", (False, True))
def test_trace_help_endpoint_runs_native_schema_and_metadata_discovery(
    recording_server: RecordingServer, discovery_fails: bool
) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.tracing_endpoints import provide_receiver, provide_trace_query_secret, router

    recording_server.expected_requests = 17
    for _ in range(11):
        recording_server.enqueue(ResponseSpec(body=""))
    for response in (
        {"data": [{"name": "Model", "type": "String"}]},
        {"data": []},
        {"data": []},
    ):
        recording_server.enqueue(ResponseSpec(body=response))
    metadata: Final = (
        ResponseSpec(status=503, body="discovery failed")
        if discovery_fails
        else ResponseSpec(body={"data": [{"metadata": '{"custom": {"label": "hello"}}'}]})
    )
    recording_server.enqueue(metadata)
    recording_server.enqueue(ResponseSpec(body={"data": [{"key": "custom.span"}]}))
    recording_server.enqueue(ResponseSpec(body={"data": [{"key": "custom.resource"}]}))
    storage: Final = ClickHouseStorage(TraceStorageConfig(recording_server.base_url, "trace_test"))
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[provide_trace_query_secret] = lambda: "test-master-secret"
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role="proxy_admin", token="test")
    app.dependency_overrides[provide_receiver] = lambda: TraceReceiver(storage)
    with TestClient(app) as client:
        result: Final = client.get("/v1/traces/query/help")
    assert result.status_code == 200, result.text
    body: Final = result.json()
    assert body["guide"].startswith("Trace SQL query guide")
    assert body["tables"][0]["columns"] == [{"name": "Model", "type": "String"}]
    if discovery_fails:
        assert body["metadata"]["fields"] == []
        assert "503" in body["metadata"]["error"]
    else:
        assert "JSONExtractRaw(metadata, 'custom', 'label')" in body["guide"]
        assert body["metadata"]["fields"][1] == {
            "path": ["custom", "label"],
            "types": ["string"],
            "expression": "JSONExtractRaw(metadata, 'custom', 'label')",
        }
    assert body["attributes"][0]["fields"][0]["expression"] == "SpanAttributes['custom.span']"
    assert body["attributes"][1]["fields"][0]["expression"] == "ResourceAttributes['custom.resource']"


@pytest.mark.parametrize(
    ("clickhouse_status", "body", "expected_status"),
    (
        (400, b"ClickHouse rejected the query", 400),
        (404, b"ClickHouse rejected the query", 400),
        (500, b"ClickHouse rejected the query", 503),
        (503, b"ClickHouse rejected the query", 503),
        (200, b'{"data":[]}', 503),
    ),
)
def test_trace_sql_endpoint_distinguishes_query_errors_from_reader_failures(
    recording_server: RecordingServer, clickhouse_status: int, body: bytes, expected_status: int
) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.tracing_endpoints import provide_receiver, provide_trace_query_secret, router

    recording_server.expected_requests = 13
    for _ in range(11):
        recording_server.enqueue(ResponseSpec(body=""))
    recording_server.enqueue(ResponseSpec(status=clickhouse_status, body=body))
    envelope: Final = {
        "meta": [{"name": "answer", "type": "UInt8"}],
        "data": [{"answer": 42}],
        "rows": 1,
        "statistics": {"elapsed": 0.01, "rows_read": 1, "bytes_read": 1},
        "rows_before_limit_at_least": 1,
    }
    recording_server.enqueue(ResponseSpec(body=envelope))
    storage: Final = ClickHouseStorage(TraceStorageConfig(recording_server.base_url, "trace_test"))
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[provide_trace_query_secret] = lambda: "test-master-secret"
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role="proxy_admin", token="test")
    app.dependency_overrides[provide_receiver] = lambda: TraceReceiver(storage)
    with TestClient(app) as client:
        failed: Final = client.post("/v1/traces/query", json={"sql": "SELEC 42"})
        assert failed.status_code == expected_status, failed.text
        recovered: Final = client.post("/v1/traces/query", json={"sql": "SELECT 42 AS answer"})
        assert recovered.status_code == 200, recovered.text
        assert recovered.json() == {"data": envelope["data"]}
    assert recording_server.requests[-2].raw_body == b"SELEC 42"


@dataclass(frozen=True, slots=True)
class SeededTraceAPI:
    client: TestClient
    storage: ClickHouseStorage
    spends: tuple[SpendLogRecord, ...]
    help: TraceQueryHelp

    def query_example(self, name: str) -> tuple[dict[str, JsonValue], ...]:
        example: Final = next(example for example in self.help.examples if example.name == name)
        response: Final = self.client.post("/v1/traces/query", json={"sql": example.sql})
        assert response.status_code == 200, response.text
        return QUERY_ROWS.validate_python(response.json()["data"])


@pytest.fixture
def seeded_trace_api(clickhouse_url: str) -> Iterator[SeededTraceAPI]:
    from seed_tracing_fixtures import (
        TRACE_FIXTURES,
        fixture_replays,
        rebase_spend,
    )

    spends: Final = dict(spend_fixtures())["openai_agents_swarm"]
    pattern: Final = re.compile("|".join(re.escape(row["response_id"]) for row in spends))
    replays: Final = fixture_replays(TRACE_FIXTURES, time.time_ns() // 1_000_000, "query-api", pattern)
    swarm: Final = next(replay for replay in replays if replay.name == "openai_agents_swarm")
    rebased: Final = rebase_spend(spends, swarm.offset_ms, swarm.namespace, pattern)
    stamped: Final[tuple[SpendLogRecord, ...]] = tuple(
        {**row, "team_id": "team-a", "api_key": "fixture-key", "user": "fixture-user"} for row in rebased
    )
    yield from _fixture_trace_api(clickhouse_url, replays, stamped)


def _fixture_trace_api(
    clickhouse_url: str, replays: tuple[FixtureReplay, ...], stamped: tuple[SpendLogRecord, ...]
) -> Generator[SeededTraceAPI]:
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.tracing_endpoints import provide_receiver, provide_trace_query_secret, router

    storage: Final = ClickHouseStorage(TraceStorageConfig(clickhouse_url, "trace_test"))
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[provide_trace_query_secret] = lambda: "fixture-secret"
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, team_id="team-a", token="fixture-key", user_id="fixture-user"
    )
    app.dependency_overrides[provide_receiver] = lambda: TraceReceiver(storage)
    with TestClient(app) as client:
        assert client.portal is not None
        client.portal.call(storage.ensure_schema)
        for replay in replays:
            client.portal.call(
                TraceReceiver(storage).ingest,
                json.dumps(replay.export).encode(),
                "application/json",
                None,
                Tenant(team_id="team-a", api_key_hash="fixture-key", user_id="fixture-user"),
            )
        client.portal.call(storage.insert_rows, "spend_logs", stamped)
        response: Final = client.get("/v1/traces/query/help")
        assert response.status_code == 200, response.text
        yield SeededTraceAPI(client, storage, stamped, TraceQueryHelp.model_validate(response.json()))


def test_fixture_backed_help_examples_execute_through_query_api(seeded_trace_api: SeededTraceAPI) -> None:
    api: Final = seeded_trace_api
    assert {table.name for table in api.help.tables} == {"otel_traces", "spend_logs", "agent_traces_by_key"}
    assert api.help.metadata.error is None
    assert api.help.metadata.sampled_rows == len(api.spends)
    assert any(field.path == ("fixture_capture", "name") for field in api.help.metadata.fields)
    for example in api.help.examples:
        api.query_example(example.name)
    records: Final = api.query_example("Recent spend records")
    assert {str(row["request_id"]) for row in records} == {row["request_id"] for row in api.spends}
    total: Final = sum(row["spend"] or 0 for row in api.spends)
    recorded: Final = api.query_example("Recorded spend by trace")
    assert len(recorded) == 1
    assert recorded[0]["trace_id"] == api.spends[0]["trace_id"]
    assert int(str(recorded[0]["requests"])) == len(api.spends)
    assert math.isclose(float(str(recorded[0]["recorded_spend"])), total)
    detail: Final = api.client.get(f"/v1/traces/{api.spends[0]['trace_id']}")
    assert detail.status_code == 200, detail.text
    assert math.isclose(TRACE.validate_json(detail.content)["summary"]["spend"] or 0, total)
    unmatched: Final = api.query_example("LLM spans without a direct spend match")
    assert unmatched
    assert all(row["TraceId"] != api.spends[0]["trace_id"] for row in unmatched)
    unpriced: Final = api.client.get(f"/v1/traces/{unmatched[0]['TraceId']}")
    assert unpriced.status_code == 200, unpriced.text
    assert unpriced.json()["summary"]["spend"] is None


@pytest.mark.parametrize("spend", (None, 0.0, 0.125), ids=("unknown", "free", "paid"))
def test_query_model_totals_deduplicate_and_preserve_unknown_cost(
    seeded_trace_api: SeededTraceAPI, spend: float | None
) -> None:
    api: Final = seeded_trace_api
    original: Final = api.spends[0]
    replacement: Final[SpendLogRecord] = {**original, "end_time": original["end_time"] + 1, "spend": spend}
    assert api.client.portal is not None
    api.client.portal.call(api.storage.insert_rows, "spend_logs", (replacement,))
    totals: Final = api.query_example("Spend and tokens by model")
    row: Final = next(row for row in totals if row["model"] == original["model"])
    model_spends: Final = tuple(row for row in api.spends if row["model"] == original["model"])
    assert int(str(row["requests"])) == len(model_spends)
    assert int(str(row["input_tokens"])) == sum(row["prompt_tokens"] for row in model_spends)
    assert int(str(row["output_tokens"])) == sum(row["completion_tokens"] for row in model_spends)
    assert int(str(row["unknown_cost_requests"])) == int(spend is None)
    if spend is None:
        assert row["spend"] is None
    else:
        assert math.isclose(
            float(str(row["spend"])), sum(row["spend"] or 0 for row in model_spends) - (original["spend"] or 0) + spend
        )


def test_query_correlation_requires_key_or_user_ownership_within_a_team(seeded_trace_api: SeededTraceAPI) -> None:
    api: Final = seeded_trace_api
    original: Final = api.spends[0]
    unrelated: Final[SpendLogRecord] = {
        **original,
        "request_id": "unrelated-request",
        "api_key": "other-key",
        "user": "other-user",
    }
    assert api.client.portal is not None
    api.client.portal.call(api.storage.insert_rows, "spend_logs", (unrelated,))
    matches: Final = api.query_example("Traces correlated with LLM call metadata")
    assert {str(row["request_id"]) for row in matches} == {row["request_id"] for row in api.spends}
    assert all(row["request_id"] != unrelated["request_id"] for row in matches)


def _captured_replays(
    namespace: str,
) -> tuple[tuple[FixtureReplay, ...], tuple[tuple[str, tuple[SpendLogRecord, ...]], ...]]:
    captures: Final = spend_fixtures()
    pattern: Final = response_pattern(tuple(chain.from_iterable(rows for _, rows in captures)))
    replays: Final = fixture_replays(TRACE_FIXTURES, time.time_ns() // 1_000_000, namespace, pattern)
    by_name: Final = MappingProxyType(dict(captures))
    return replays, tuple(
        (
            replay.name,
            tuple(
                _stamp(row) for row in rebase_spend(by_name[replay.name], replay.offset_ms, replay.namespace, pattern)
            ),
        )
        for replay in replays
        if replay.name in by_name
    )


def _stamp(row: SpendLogRecord) -> SpendLogRecord:
    return {**row, "team_id": "team-a", "api_key": "fixture-key", "user": "fixture-user"}


@pytest.fixture(scope="module")
def captured_trace_api() -> Iterator[SeededTraceAPI]:
    replays, paired = _captured_replays("captured-api")
    with clickhouse_service() as url:
        yield from _fixture_trace_api(url, replays, tuple(chain.from_iterable(rows for _, rows in paired)))


@pytest.mark.parametrize("name", tuple(name for name, _ in spend_fixtures()))
def test_captured_sdk_cost_survives_seeding_and_is_queryable(name: str, captured_trace_api: SeededTraceAPI) -> None:
    api: Final = captured_trace_api
    rows: Final = tuple(row for row in api.spends if fixture_capture("", row).name == name)
    assert rows
    capture: Final = fixture_capture(name, rows[0])
    response: Final = api.client.get(f"/v1/traces/{capture.trace_id}")
    assert response.status_code == 200, response.text
    detail: Final = TRACE.validate_json(response.content)
    original: Final = span_rows((TRACE_FIXTURES / f"{name}.json").read_bytes(), "application/json")
    assert detail["summary"]["span_count"] == len(original)
    if capture.spend_linked and capture.spend_complete:
        assert detail["summary"]["spend"] is not None
        assert math.isclose(detail["summary"]["spend"], sum(row["spend"] or 0 for row in rows))
    else:
        assert detail["summary"]["spend"] is None
    query: Final = api.client.post(
        "/v1/traces/query",
        json={
            "sql": "SELECT request_id, spend, prompt_tokens, completion_tokens FROM spend_logs FINAL "
            f"WHERE JSONExtractString(metadata, 'fixture_capture', 'name') = '{name}' LIMIT 100"
        },
    )
    assert query.status_code == 200, query.text
    records: Final = CapturedSpendQuery.model_validate_json(query.content).data
    assert {row.request_id for row in records} == {row["request_id"] for row in rows}
    assert math.isclose(sum(row.spend for row in records), sum(row["spend"] or 0 for row in rows))
    assert sum(row.prompt_tokens for row in records) == sum(row["prompt_tokens"] for row in rows)
    assert sum(row.completion_tokens for row in records) == sum(row["completion_tokens"] for row in rows)


def test_server_side_copies_keep_every_capture_linked_to_its_spend() -> None:
    replays, paired = _captured_replays("copied-api")
    copies: Final = Copies(
        trace_ids=tuple(sorted(frozenset(str(span["TraceId"]) for span in bulk_span_rows(replays, Tenant("", ""))))),
        request_ids=tuple(row["request_id"] for _, rows in paired for row in rows),
        numbers=range(1, 3),
        step_ms=60_000,
        source="seed-copied-api-",
        target="seed-copied-api-c",
    )
    (session,) = long_sessions(replays, paired, "seed-copied-api-", "seed-copied-api-c", (3,))
    session_spend: Final = sum(row["spend"] or 0 for row in dict(paired)["openai_agents_swarm"])
    session_spans: Final = len(
        span_rows((TRACE_FIXTURES / "openai_agents_swarm.json").read_bytes(), "application/json")
    )
    with (
        clickhouse_service() as url,
        closing(_fixture_trace_api(url, replays, tuple(chain.from_iterable(rows for _, rows in paired)))) as seeded,
    ):
        api: Final = next(seeded)
        assert api.client.portal is not None
        for plan in (copies, session):
            api.client.portal.call(_copy_clickhouse, url, plan)
        for name, rows in paired:
            _assert_capture(api, name, rows, fixture_capture(name, rows[0]).trace_id)
            _assert_capture(api, name, rows, copied_trace_id(fixture_capture(name, rows[0]).trace_id, "2"))
        trace: Final = _trace(api, copied_trace_id(session.trace_ids[0], session.session))
        assert trace["summary"]["span_count"] == 1 + 3 * (session_spans - 1)
        (root,) = (span for span in trace["spans"] if not span["parent_span_id"])
        assert {span["parent_span_id"] for span in trace["spans"] if span["parent_span_id"]} <= {
            span["span_id"] for span in trace["spans"]
        }
        assert root["start_offset_ms"] == min(span["start_offset_ms"] for span in trace["spans"])
        assert root["start_offset_ms"] + root["duration_ms"] >= max(
            span["start_offset_ms"] + span["duration_ms"] for span in trace["spans"]
        )
        assert trace["summary"]["spend"] == pytest.approx(3 * session_spend)


def _trace(api: SeededTraceAPI, trace_id: str) -> Trace:
    response: Final = api.client.get(f"/v1/traces/{trace_id}")
    assert response.status_code == 200, response.text
    return TRACE.validate_json(response.content)


def _assert_capture(api: SeededTraceAPI, name: str, rows: tuple[SpendLogRecord, ...], trace_id: str) -> None:
    capture: Final = fixture_capture(name, rows[0])
    summary: Final = _trace(api, trace_id)["summary"]
    assert summary["span_count"] == len(span_rows((TRACE_FIXTURES / f"{name}.json").read_bytes(), "application/json"))
    assert summary["spend"] == (
        pytest.approx(sum(row["spend"] or 0 for row in rows))
        if capture.spend_linked and capture.spend_complete
        else None
    )


async def _copy_clickhouse(url: str, copies: Copies) -> None:
    async with httpx.AsyncClient(base_url=url, params={"database": "trace_test"}) as client:
        await copy_clickhouse(client, "trace_test", copies)
