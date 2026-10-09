import base64
import hashlib
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

from litellm.tracing.generated.responses import TraceSQLResponse
from litellm.tracing.types import SpendLogRecord
from scripts.seed_tracing_fixtures import (
    JSON,
    TRACE_FIXTURES,
    fixture_capture,
    fixture_replays,
    managed_response,
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
from tests._master_key import MASTER_KEY

CALL_KEYS: Final = TypeAdapter(tuple[str, ...])
DATETIMES: Final = TypeAdapter(tuple[datetime, datetime])
SPAN_IDENTITY: Final = TypeAdapter(tuple[str, str, str, int])
JSON_FIELDS: Final[TypeAdapter[tuple[Json, Json, Json]]] = TypeAdapter(
    tuple[InstanceOf[Json], InstanceOf[Json], InstanceOf[Json]]
)


@pytest.mark.parametrize("path", sorted(TRACE_FIXTURES.glob("*.json")), ids=lambda path: path.stem)
def test_fixture_replays_preserve_raw_payloads_except_time_and_identity(path: Path) -> None:
    export: Final = JSON.validate_json(path.read_bytes())
    now_ms: Final = max(timestamps(export)) // 1_000_000 + 86_400_000
    replays: Final = fixture_replays(TRACE_FIXTURES, now_ms, "all-fixtures", re.compile(r"(?!)"))
    replay: Final = next(item for item in replays if item.name == path.stem)
    group: Final = tuple(item for item in replays if item.namespace == replay.namespace)
    assert max(max(timestamps(item.export)) for item in group) // 1_000_000 == now_ms - 1000
    assert len(frozenset(item.offset_ms for item in group)) == 1
    assert tuple(timestamps(replay.export)) == tuple(
        timestamp + replay.offset_ms * 1_000_000 for timestamp in timestamps(export)
    )
    assert replay.export != export


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


@pytest.mark.parametrize("name,spends", spend_fixtures())
def test_captured_spend_replay_preserves_real_cost_and_call_identity(
    name: str, spends: tuple[SpendLogRecord, ...]
) -> None:
    export: Final = JSON.validate_json((TRACE_FIXTURES / f"{name}.json").read_bytes())
    pattern: Final = response_pattern(spends)
    offset_ms: Final = 1123
    namespace: Final = f"captured-{name}"
    shifted: Final = rebase(export, offset_ms * 1_000_000, namespace, pattern)
    replayed: Final = rebase_spend(spends, offset_ms, namespace, pattern)
    capture: Final = fixture_capture(name, replayed[0])

    assert capture.trace_id == seed_id(fixture_capture(name, spends[0]).trace_id, namespace, 32)
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


def test_replay_rebases_provider_request_evidence_with_the_spend_row() -> None:
    spend: Final = {**dict(spend_fixtures())["deepagents_swarm"][0], "provider_request_id": "req_replay"}
    pattern: Final = response_pattern((spend,))
    replayed: Final = rebase_spend((spend,), 0, "another-run", pattern)[0]
    export: Final = rebase({"request_id": "req_replay"}, 0, "another-run", pattern)

    assert export == {"request_id": replayed["provider_request_id"]}
    assert replayed["provider_request_id"] != spend["provider_request_id"]
    assert replayed["provider_request_id"].startswith("req_")


@pytest.mark.parametrize("upstream", ("msg_response", "req_response", "chatcmpl-response"))
def test_replay_preserves_identity_between_managed_and_plain_response_ids(upstream: str) -> None:
    payload: Final = f"model:example;response_id:{upstream}"
    managed: Final = "resp_" + base64.b64encode(payload.encode()).decode()
    spend: Final = {**dict(spend_fixtures())["deepagents_swarm"][0], "response_id": managed}
    pattern: Final = response_pattern((spend,))
    replayed: Final = rebase_spend((spend,), 0, "managed-replay", pattern)[0]
    plain: Final = rebase(upstream, 0, "managed-replay", pattern)

    assert plain != upstream
    assert managed_response(replayed["response_id"]) == f"model:example;response_id:{plain}"


@pytest.mark.asyncio
async def test_seed_copy_uses_lens_authenticated_tenant_for_both_spend_stores(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    import httpx

    from litellm.tracing.remote import RemoteTraceStore

    monkeypatch.setenv("LITELLM_MASTER_KEY", "fixture-admin")
    fixture: Final = next(item for item in spend_fixtures() if item[0] == "deepagents_swarm")
    pattern: Final = response_pattern(fixture[1])
    replay: Final = next(
        item
        for item in fixture_replays(TRACE_FIXTURES, 1_800_000_000_000, "remote-seed", pattern)
        if item.name == fixture[0]
    )
    requests: Final = asyncio.Queue[httpx.Request]()

    def accept(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(request)
        if request.url.path == "/internal/read":
            return httpx.Response(
                200,
                json={
                    "meta": [],
                    "data": [{"team_id": "lens-team", "api_key": "lens-key-hash", "user": "lens-user"}],
                    "rows": 1,
                    "statistics": {"elapsed": 0, "rows_read": 1, "bytes_read": 1},
                },
            )
        return httpx.Response(204)

    database: Final = AsyncMock(spec=Prisma, litellm_spendlogs=AsyncMock())
    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(accept)) as client:
        captures: Final = await seed_copy(client, RemoteTraceStore(client), database, (replay,), (fixture,), pattern)
    upload: Final = requests.get_nowait()
    assert upload.url.path == "/v1/traces"
    assert json.loads(upload.content) == replay.export
    read: Final = requests.get_nowait()
    assert json.loads(read.content)["scope"] == {"kind": "all"}
    inserted: Final = requests.get_nowait()
    assert inserted.url.path == "/internal/spend"
    rows: Final = json.loads(inserted.content)
    assert len(rows) == len(fixture[1])
    assert all(
        (row["team_id"], row["api_key"], row["user"]) == ("lens-team", "lens-key-hash", "lens-user") for row in rows
    )
    assert tuple(row["request_id"] for row in captures[0][1]) == tuple(row["request_id"] for row in rows)
    saved: Final = database.litellm_spendlogs.create_many.call_args.kwargs["data"]
    assert tuple((row["request_id"], row["spend"]) for row in saved) == tuple(
        (row["request_id"], row["spend"]) for row in rows
    )
    assert requests.empty()


@pytest.mark.asyncio
async def test_seed_copy_ids_come_from_lens_normalization(monkeypatch: pytest.MonkeyPatch) -> None:

    from litellm.tracing.remote import RemoteTraceStore
    from scripts.seed_tracing_fixtures import seeded_trace_ids

    monkeypatch.setenv("LITELLM_MASTER_KEY", "fixture-admin")

    def accept(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {
            "operation": "sql",
            "scope": {"kind": "all"},
            "sql": "SELECT DISTINCT TraceId AS trace_id FROM otel_traces WHERE ApiKeyHash = 'key-hash' ORDER BY trace_id",
        }
        return httpx.Response(
            200,
            json={
                "meta": [],
                "data": [{"trace_id": "normalized-session"}, {"trace_id": "original-trace"}],
                "rows": 2,
                "statistics": {"elapsed": 0, "rows_read": 2, "bytes_read": 1},
            },
        )

    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(accept)) as client:
        assert await seeded_trace_ids(RemoteTraceStore(client), "key-hash") == ("normalized-session", "original-trace")
