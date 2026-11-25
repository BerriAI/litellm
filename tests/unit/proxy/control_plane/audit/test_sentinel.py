import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from itertools import accumulate, chain as chain_iterables
from typing import Final
from urllib.parse import parse_qs

import httpx
import pytest
from pydantic import SecretStr

from litellm.proxy.control_plane.audit.event import AuditEvent
from litellm.proxy.control_plane.audit.ocsf import AuditRecord, ChainState, Product, chain
from litellm.proxy.control_plane.audit.pipeline import Batch, Delivered, Rejected, Undelivered, build_pipeline
from litellm.proxy.control_plane.audit.sentinel import (
    ClientCredentials,
    SentinelSink,
    TokenRefused,
    chunks,
    sentinel_row,
)
from litellm.proxy.control_plane.settings import SentinelSettings

T0 = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
SETTINGS: Final = SentinelSettings(
    endpoint="https://dce.ingest.monitor.azure.com/",
    dcr_immutable_id="dcr-abc",
    stream_name="Custom-LiteLLMControlPlaneAudit_CL",
    tenant_id="tenant",
    client_id="app",
    client_secret=SecretStr("app-secret"),
    authority_host="https://login.microsoftonline.com",
    token_scope="https://monitor.azure.com/.default",
    timeout_seconds=4.0,
)
INGEST_URL: Final = (
    "https://dce.ingest.monitor.azure.com/dataCollectionRules/dcr-abc/streams/Custom-LiteLLMControlPlaneAudit_CL"
    "?api-version=2023-01-01"
)
TOKEN_URL: Final = "https://login.microsoftonline.com/tenant/oauth2/v2.0/token"


@dataclass
class FakePoster:  # mutable-ok: scripts responses and records posts for assertions
    outcomes: list[httpx.Response | Exception]
    posts: list[tuple[str, bytes, Mapping[str, str], float]] = field(default_factory=list)

    async def post(self, url: str, content: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response:
        self.posts.append((url, content, headers, timeout))
        outcome: Final = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@dataclass
class Clock:  # mutable-ok: the test moves time forward
    now: float = 1_000.0

    def __call__(self) -> float:
        return self.now


def _batch(count: int) -> Batch:
    events: Final = tuple(
        AuditEvent(
            T0, "alice", "a@x.test", "eu", "POST", f"/key/{index}", "denied", 403, None, "no", "api_activity", "ab" * 16
        )
        for index in range(count)
    )
    if not events:
        return ()
    first: Final = chain(events[0], Product("t"), ChainState())
    state_and_records: Final = tuple(
        accumulate(events[1:], lambda previous, event: chain(event, Product("t"), previous[0]), initial=first)
    )
    return tuple(record for _, record in state_and_records)


async def _token() -> str:
    return "bearer-1"


@pytest.mark.asyncio
async def test_a_batch_is_one_json_array_of_flat_rows_posted_to_the_rule_stream_with_the_bearer() -> None:
    poster: Final = FakePoster([httpx.Response(204)])
    batch: Final = _batch(2)
    assert await SentinelSink(SETTINGS, poster, _token).deliver(batch) == Delivered(2)
    ((url, content, headers, timeout),) = poster.posts
    assert url == INGEST_URL and timeout == 4.0
    assert headers == {"Authorization": "Bearer bearer-1", "Content-Type": "application/json"}
    rows: Final = json.loads(content)
    assert rows == [sentinel_row(record) for record in batch]
    assert rows[1]["TimeGenerated"] == "2026-10-13T12:00:00+00:00"
    assert (rows[1]["Sequence"], rows[1]["PrevHash"], rows[1]["Hash"]) == (2, batch[0].hash, batch[1].hash)
    assert (rows[1]["ClassUid"], rows[1]["Decision"], rows[1]["ResponseCode"], rows[1]["ActorEmail"]) == (
        6003,
        "denied",
        403,
        "a@x.test",
    )
    assert json.loads(rows[1]["Ocsf"]) == batch[1].body and rows[1]["TraceId"] == "ab" * 16


def test_rows_are_split_into_arrays_under_the_api_payload_limit() -> None:
    rows: Final = [{"Ocsf": "x" * 300} for _ in range(10)]
    parts: Final = chunks(rows, limit=1_000)
    assert [len(json.loads(part)) for part in parts] == [3, 3, 3, 1]
    assert all(len(part) <= 1_000 for part in parts)
    assert [json.loads(part) for part in chunks([], limit=1_000)] == []


def test_a_row_that_alone_exceeds_the_limit_is_sent_without_its_free_text_instead_of_blocking_the_batch() -> None:
    rows: Final = [{"Hash": "h1", "Detail": "d", "Ocsf": "x" * 2_000}, {"Hash": "h2", "Detail": "d", "Ocsf": "y"}]
    parts: Final = chunks(rows, limit=1_000)
    assert all(len(part) <= 1_000 for part in parts)
    decoded_parts: Final = tuple(json.loads(part) for part in parts)
    sent: Final = tuple(chain_iterables.from_iterable(decoded_parts))
    assert sent == (
        {"Hash": "h1", "Detail": None, "Ocsf": None, "Truncated": True},
        {"Hash": "h2", "Detail": "d", "Ocsf": "y"},
    )


@pytest.mark.asyncio
async def test_the_entra_token_is_fetched_once_and_renewed_only_near_expiry() -> None:
    poster: Final = FakePoster(
        [
            httpx.Response(200, json={"access_token": "t1", "expires_in": 600}),
            httpx.Response(200, json={"access_token": "t2", "expires_in": 600}),
        ]
    )
    clock: Final = Clock()
    credentials: Final = ClientCredentials(SETTINGS, poster, now=clock)
    assert [await credentials() for _ in range(3)] == ["t1", "t1", "t1"]
    ((url, form, headers, _),) = poster.posts
    assert url == TOKEN_URL and headers == {"Content-Type": "application/x-www-form-urlencoded"}
    assert parse_qs(form.decode()) == {
        "client_id": ["app"],
        "client_secret": ["app-secret"],
        "scope": ["https://monitor.azure.com/.default"],
        "grant_type": ["client_credentials"],
    }
    clock.now += 541
    assert await credentials() == "t2" and len(poster.posts) == 2


@pytest.mark.asyncio
async def test_client_credentials_returns_token_refusal_as_a_value() -> None:
    poster: Final = FakePoster([httpx.Response(401, json={"error": "invalid_client"})])

    assert await ClientCredentials(SETTINGS, poster)() == TokenRefused(401)


@pytest.mark.asyncio
async def test_client_credentials_uses_an_explicit_token_scope() -> None:
    scope: Final = "https://monitor.custom.test/.default"
    settings: Final = replace(SETTINGS, token_scope=scope)
    poster: Final = FakePoster([httpx.Response(200, json={"access_token": "token"})])
    credentials: Final = ClientCredentials(settings, poster)

    assert await credentials() == "token"
    ((_, form, _, _),) = poster.posts
    assert parse_qs(form.decode())["scope"] == [scope]


@pytest.mark.asyncio
async def test_a_refused_token_or_ingestion_error_is_retried_and_names_no_secret() -> None:
    poster: Final = FakePoster(
        [httpx.Response(401, json={"error": "invalid_client"}), httpx.Response(429), httpx.ConnectError("x")]
    )
    refused: Final = await SentinelSink(SETTINGS, poster, ClientCredentials(SETTINGS, poster)).deliver(_batch(1))
    assert refused == Undelivered("entra token refused: HTTP 401 from the token endpoint")
    sink: Final = SentinelSink(SETTINGS, poster, _token)
    assert await sink.deliver(_batch(1)) == Undelivered("HTTP 429 from the logs ingestion api")
    assert await sink.deliver(_batch(1)) == Undelivered("ConnectError posting to the logs ingestion api")


@pytest.mark.parametrize("status_code", [401, 403])
@pytest.mark.asyncio
async def test_access_refusals_from_ingestion_are_retryable(status_code: int) -> None:
    poster: Final = FakePoster([httpx.Response(status_code)])
    assert await SentinelSink(SETTINGS, poster, _token).deliver(_batch(1)) == Undelivered(
        f"HTTP {status_code} from the logs ingestion api"
    )


@pytest.mark.asyncio
async def test_permanent_413_rejection_does_not_block_the_next_batch() -> None:
    poster: Final = FakePoster([httpx.Response(413), httpx.Response(204)])
    sink: Final = SentinelSink(SETTINGS, poster, _token, _chunk_limit=1)
    assert await sink.deliver(_batch(1)) == Rejected("chunks were permanently rejected", 1)
    assert await sink.deliver(_batch(1)) == Delivered(1)
    assert len(poster.posts) == 2


@pytest.mark.asyncio
async def test_a_permanently_rejected_chunk_does_not_stop_later_chunks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    poster: Final = FakePoster([httpx.Response(204), httpx.Response(400), httpx.Response(204)])
    sink: Final = SentinelSink(SETTINGS, poster, _token, _chunk_limit=1)
    assert await sink.deliver(_batch(3)) == Rejected("chunks were permanently rejected", 1)
    assert len(poster.posts) == 3
    assert "rejected chunk with HTTP 400" in caplog.text


@pytest.mark.asyncio
async def test_a_retryable_failure_in_a_chunk_makes_the_batch_undelivered() -> None:
    poster: Final = FakePoster([httpx.Response(204), httpx.Response(403)])
    sink: Final = SentinelSink(SETTINGS, poster, _token, _chunk_limit=1)
    assert await sink.deliver(_batch(2)) == Undelivered("HTTP 403 from the logs ingestion api", settled=1)
    assert len(poster.posts) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("statuses, rejected", [((400, 204), 1), ((204, 422), 1), ((400, 422), 2)])
async def test_chunk_rejections_are_counted_alongside_delivered_chunks(
    statuses: tuple[int, int], rejected: int
) -> None:
    poster: Final = FakePoster([httpx.Response(status) for status in statuses])
    sink: Final = SentinelSink(SETTINGS, poster, _token, _chunk_limit=1)
    assert await sink.deliver(_batch(2)) == Rejected("chunks were permanently rejected", rejected)
    assert len(poster.posts) == 2


@pytest.mark.asyncio
async def test_rejected_chunk_is_counted_once_while_only_the_undelivered_suffix_is_retried(
    caplog: pytest.LogCaptureFixture,
) -> None:
    poster: Final = FakePoster([httpx.Response(400), httpx.Response(503), httpx.Response(204), httpx.Response(204)])
    batch: Final = _batch(3)
    sink: Final = SentinelSink(SETTINGS, poster, _token, _chunk_limit=1)
    pipeline: Final = build_pipeline((sink,), queue_size=4, batch_size=3, strict=True)
    for record in batch:
        pipeline.publish(record)
    assert await pipeline.workers[0].flush_once() == 3
    assert pipeline.workers[0].rejected == 1 and not pipeline.accepting()
    assert pipeline.workers[0].queue.empty()
    rows: Final = tuple(json.loads(content)[0] for _, content, _, _ in poster.posts)
    assert [row["Hash"] for row in rows] == [batch[0].hash, batch[1].hash, batch[1].hash, batch[2].hash]
    assert caplog.text.count("sink sentinel rejected 1 records") == 1


def test_a_row_whose_flat_column_alone_exceeds_the_limit_keeps_only_its_fixed_width_columns() -> None:
    rows: Final = [{"Hash": "h1", "Sequence": 7, "UrlPath": "/" + "k" * 2_000, "Detail": "d", "Ocsf": "x"}]
    parts: Final = chunks(rows, limit=1_000)
    assert all(len(part) <= 1_000 for part in parts)
    assert [json.loads(part) for part in parts] == [[{"Hash": "h1", "Sequence": 7, "Truncated": True}]]


def test_oversized_ocsf_and_other_fields_are_bounded_in_utf8_bytes() -> None:
    path: Final = "/" + "é" * 35_000
    detail: Final = "é" * 35_000
    event: Final = AuditEvent(T0, "alice", "a@x.test", "eu", "POST", path, "denied", 403, None, detail)
    record: Final = chain(event, Product("t"), ChainState())[1]
    source: Final = sentinel_row(record)
    assert len(source["Ocsf"].encode("utf-8")) > 64 * 1024

    payload: Final = chunks((source,))[0]
    (row,) = json.loads(payload)
    compact: Final = json.loads(row["Ocsf"])
    assert row["Truncated"] is True and len(payload) <= 1_000_000
    assert len(row["Ocsf"].encode("utf-8")) <= 64 * 1024
    assert len(row["UrlPath"].encode("utf-8")) <= 64 * 1024 and row["UrlPath"].endswith("...[truncated]")
    assert len(row["Detail"].encode("utf-8")) <= 64 * 1024 and row["Detail"].endswith("...[truncated]")
    assert compact["actor"] == record.body["actor"]
    assert (compact["activity_name"], compact["status"]) == ("Create", "Failure")
    assert compact["unmapped"]["litellm"] == {"hash": record.hash, "prev_hash": None, "sequence": 1}
    assert compact["http_request"]["url"]["path"].endswith("...[truncated]")


def test_quote_heavy_paths_keep_the_final_ocsf_string_under_the_field_limit() -> None:
    path: Final = "/" + '"' * (32 * 1024 + 100)
    record: Final = chain(
        AuditEvent(T0, "alice", "a@x.test", "eu", "POST", path, "denied", 403, None, "no"),
        Product("t"),
        ChainState(),
    )[1]
    source: Final = sentinel_row(record)
    assert len(source["Ocsf"].encode("utf-8")) > 64 * 1024

    (row,) = json.loads(chunks((source,))[0])
    ocsf: Final = row["Ocsf"]
    assert isinstance(ocsf, str)
    assert len(ocsf.encode("utf-8")) <= 64 * 1024
    compact: Final = json.loads(ocsf)
    assert compact["Truncated"] is True
    assert compact["activity_id"] == record.body["activity_id"]
    assert compact["activity_name"] == record.body["activity_name"]
    assert compact["unmapped"]["litellm"] == {"hash": record.hash, "prev_hash": None, "sequence": 1}


def test_oversized_actor_values_are_bounded_in_compacted_ocsf() -> None:
    actor: Final = "a" * 70_000
    event: Final = AuditEvent(T0, actor, None, "eu", "POST", "/", "denied", 403, None, None)
    record: Final = chain(event, Product("t"), ChainState())[1]
    source: Final = sentinel_row(record)
    payload: Final = chunks((source,))[0]
    (row,) = json.loads(payload)
    compact: Final = json.loads(row["Ocsf"])
    compact_actor: Final = compact["actor"]["user"]["uid"]

    assert compact_actor != actor
    assert compact_actor.endswith("...[truncated]")
    assert len(compact_actor.encode("utf-8")) <= 4 * 1024


def test_nested_actor_values_are_compacted_without_losing_their_shape() -> None:
    record: Final = chain(
        AuditEvent(T0, "alice", None, "eu", "POST", "/", "denied", 403, None, None),
        Product("t"),
        ChainState(),
    )[1]
    nested_record: Final = replace(
        record,
        body={
            **record.body,
            "actor": {"user": {"uid": "a" * 70_000, "groups": ["b" * 70_000]}},
        },
    )
    source: Final = sentinel_row(nested_record)
    (row,) = json.loads(chunks((source,))[0])
    compact_actor: Final = json.loads(row["Ocsf"])["actor"]

    assert compact_actor["user"]["uid"].endswith("...[truncated]")
    assert compact_actor["user"]["groups"][0].endswith("...[truncated]")
    assert len(compact_actor["user"]["uid"].encode("utf-8")) <= 4 * 1024
    assert len(compact_actor["user"]["groups"][0].encode("utf-8")) <= 4 * 1024
