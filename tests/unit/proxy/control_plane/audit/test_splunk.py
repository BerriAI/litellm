import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Final

import httpx
import pytest
from pydantic import SecretStr

from litellm.proxy.control_plane.audit.event import AuditEvent
from litellm.proxy.control_plane.audit.ocsf import AuditRecord, ChainState, Product, chain
from litellm.proxy.control_plane.audit.pipeline import Batch, Delivered, Undelivered
from litellm.proxy.control_plane.audit.splunk import SplunkHecSink
from litellm.proxy.control_plane.settings import SplunkHecSettings

T0 = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
SETTINGS: Final = SplunkHecSettings(
    url="https://hec.example.test:8088/", token=SecretStr("hec-token"), index="audit", source="cp", timeout_seconds=4.0
)


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


def _batch() -> Batch:
    state1, call = chain(
        AuditEvent(T0, "alice", None, "eu", "POST", "/key/generate", "allowed", 200, None, None),
        Product("t"),
        ChainState(),
    )
    _, login = chain(
        AuditEvent(
            T0, "admin", None, "control-plane", "POST", "/auth/login", "denied", None, None, "no", "authentication"
        ),
        Product("t"),
        state1,
    )
    return (call, login)


@pytest.mark.asyncio
async def test_a_batch_is_one_post_of_newline_separated_hec_events_carrying_the_ocsf_objects() -> None:
    poster = FakePoster([httpx.Response(200, json={"text": "Success", "code": 0})])
    batch = _batch()
    assert await SplunkHecSink(SETTINGS, poster, "splunk_hec").deliver(batch) == Delivered(2)
    ((url, content, headers, timeout),) = poster.posts
    assert url == "https://hec.example.test:8088/services/collector/event" and timeout == 4.0
    assert headers == {"Authorization": "Splunk hec-token", "Content-Type": "application/json"}
    events = [json.loads(line) for line in content.decode().split("\n")]
    assert events == [
        {
            "time": T0.timestamp(),
            "source": "cp",
            "sourcetype": "ocsf:api_activity",
            "index": "audit",
            "event": batch[0].body,
        },
        {
            "time": T0.timestamp(),
            "source": "cp",
            "sourcetype": "ocsf:authentication",
            "index": "audit",
            "event": batch[1].body,
        },
    ]


@pytest.mark.asyncio
async def test_without_an_index_the_event_leaves_it_to_the_token_default() -> None:
    poster = FakePoster([httpx.Response(200)])
    await SplunkHecSink(replace(SETTINGS, index=None), poster).deliver(_batch()[:1])
    ((_, content, _, _),) = poster.posts
    assert "index" not in json.loads(content)


@pytest.mark.asyncio
async def test_hec_refusals_and_network_errors_are_retried_without_leaking_the_token() -> None:
    poster = FakePoster(
        [
            httpx.Response(403, json={"text": "Invalid token", "code": 4}),
            httpx.Response(200, json={"text": "No data", "code": 5}),
            httpx.Response(200),
            httpx.Response(503),
            httpx.ConnectError("boom"),
        ]
    )
    sink = SplunkHecSink(SETTINGS, poster)
    batch = _batch()
    assert await sink.deliver(batch) == Undelivered("HTTP 403 from splunk hec: Invalid token (code 4)")
    assert await sink.deliver(batch) == Undelivered("HTTP 200 from splunk hec: No data (code 5)")
    assert await sink.deliver(batch) == Undelivered("HTTP 200 from splunk hec: no HEC reply body (code -1)")
    assert await sink.deliver(batch) == Undelivered("HTTP 503 from splunk hec: no HEC reply body (code -1)")
    assert await sink.deliver(batch) == Undelivered("ConnectError posting to splunk hec")
