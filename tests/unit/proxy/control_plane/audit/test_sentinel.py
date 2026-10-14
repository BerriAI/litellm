import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final
from urllib.parse import parse_qs

import httpx
import pytest
from pydantic import SecretStr

from litellm.proxy.control_plane.audit.event import AuditEvent
from litellm.proxy.control_plane.audit.ocsf import AuditRecord, ChainState, Product, chain
from litellm.proxy.control_plane.audit.pipeline import Batch, Delivered, Undelivered
from litellm.proxy.control_plane.audit.sentinel import ClientCredentials, SentinelSink, chunks, sentinel_row
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
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@dataclass
class Clock:  # mutable-ok: the test moves time forward
    now: float = 1_000.0

    def __call__(self) -> float:
        return self.now


def _batch(count: int) -> Batch:
    state: Final[list[ChainState]] = [ChainState()]
    out: Final[list[AuditRecord]] = []
    for index in range(count):
        event = AuditEvent(
            T0, "alice", "a@x.test", "eu", "POST", f"/key/{index}", "denied", 403, None, "no", "api_activity", "ab" * 16
        )
        state[0], record = chain(event, Product("t"), state[0])
        out.append(record)
    return tuple(out)


async def _token() -> str:
    return "bearer-1"


@pytest.mark.asyncio
async def test_a_batch_is_one_json_array_of_flat_rows_posted_to_the_rule_stream_with_the_bearer() -> None:
    poster = FakePoster([httpx.Response(204)])
    batch = _batch(2)
    assert await SentinelSink(SETTINGS, poster, _token).deliver(batch) == Delivered(2)
    ((url, content, headers, timeout),) = poster.posts
    assert url == INGEST_URL and timeout == 4.0
    assert headers == {"Authorization": "Bearer bearer-1", "Content-Type": "application/json"}
    rows = json.loads(content)
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
    rows = [{"Ocsf": "x" * 300} for _ in range(10)]
    parts = chunks(rows, limit=1_000)
    assert [len(json.loads(part)) for part in parts] == [3, 3, 3, 1]
    assert all(len(part) <= 1_000 for part in parts)
    assert [json.loads(part) for part in chunks([], limit=1_000)] == []


def test_a_row_that_alone_exceeds_the_limit_is_sent_without_its_free_text_instead_of_blocking_the_batch() -> None:
    rows = [{"Hash": "h1", "Detail": "d", "Ocsf": "x" * 2_000}, {"Hash": "h2", "Detail": "d", "Ocsf": "y"}]
    parts = chunks(rows, limit=1_000)
    assert all(len(part) <= 1_000 for part in parts)
    sent = [row for part in parts for row in json.loads(part)]
    assert sent == [
        {"Hash": "h1", "Detail": None, "Ocsf": None, "Truncated": True},
        {"Hash": "h2", "Detail": "d", "Ocsf": "y"},
    ]


@pytest.mark.asyncio
async def test_the_entra_token_is_fetched_once_and_renewed_only_near_expiry() -> None:
    poster = FakePoster(
        [
            httpx.Response(200, json={"access_token": "t1", "expires_in": 600}),
            httpx.Response(200, json={"access_token": "t2", "expires_in": 600}),
        ]
    )
    clock = Clock()
    credentials = ClientCredentials(SETTINGS, poster, now=clock)
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
async def test_a_refused_token_or_ingestion_error_is_retried_and_names_no_secret() -> None:
    poster = FakePoster(
        [httpx.Response(401, json={"error": "invalid_client"}), httpx.Response(429), httpx.ConnectError("x")]
    )
    refused = await SentinelSink(SETTINGS, poster, ClientCredentials(SETTINGS, poster)).deliver(_batch(1))
    assert refused == Undelivered("entra token refused: HTTP 401 from the token endpoint")
    sink = SentinelSink(SETTINGS, poster, _token)
    assert await sink.deliver(_batch(1)) == Undelivered("HTTP 429 from the logs ingestion api")
    assert await sink.deliver(_batch(1)) == Undelivered("ConnectError posting to the logs ingestion api")


def test_a_row_whose_flat_column_alone_exceeds_the_limit_keeps_only_its_fixed_width_columns() -> None:
    rows = [{"Hash": "h1", "Sequence": 7, "UrlPath": "/" + "k" * 2_000, "Detail": "d", "Ocsf": "x"}]
    parts = chunks(rows, limit=1_000)
    assert all(len(part) <= 1_000 for part in parts)
    assert [json.loads(part) for part in parts] == [[{"Hash": "h1", "Sequence": 7, "Truncated": True}]]
