import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from itertools import accumulate, chain as chain_iterables
from typing import Final

import httpx
import pytest
from pydantic import SecretStr

from litellm.proxy.control_plane.audit.event import AuditEvent
from litellm.proxy.control_plane.audit.ocsf import AuditRecord, ChainState, Product, chain
from litellm.proxy.control_plane.audit.pipeline import Batch, Delivered, Rejected, Undelivered, build_pipeline
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
        outcome: Final = self.outcomes.pop(0)
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


def _four_record_batch() -> Batch:
    def append_record(
        previous: tuple[ChainState, tuple[AuditRecord, ...]], event: AuditEvent
    ) -> tuple[ChainState, tuple[AuditRecord, ...]]:
        state, records = previous
        next_state, record = chain(event, Product("t"), state)
        return next_state, (*records, record)

    events: Final = tuple(
        AuditEvent(T0, "alice", None, "eu", "GET", f"/key/{index}", "allowed", 200, None, None) for index in range(4)
    )
    results: Final = tuple(accumulate(events, append_record, initial=(ChainState(), ())))
    return results[-1][1]


@pytest.mark.asyncio
async def test_a_batch_is_one_post_of_newline_separated_hec_events_carrying_the_ocsf_objects() -> None:
    poster: Final = FakePoster([httpx.Response(200, json={"text": "Success", "code": 0})])
    batch: Final = _batch()
    assert await SplunkHecSink(SETTINGS, poster, "splunk_hec").deliver(batch) == Delivered(2)
    ((url, content, headers, timeout),) = poster.posts
    assert url == "https://hec.example.test:8088/services/collector/event" and timeout == 4.0
    assert headers == {"Authorization": "Splunk hec-token", "Content-Type": "application/json"}
    events: Final = [json.loads(line) for line in content.decode().split("\n")]
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
    poster: Final = FakePoster([httpx.Response(200)])
    await SplunkHecSink(replace(SETTINGS, index=None), poster).deliver(_batch()[:1])
    ((_, content, _, _),) = poster.posts
    assert "index" not in json.loads(content)


@pytest.mark.asyncio
async def test_hec_permanent_refusals_are_rejected_and_transient_errors_are_retryable() -> None:
    poster: Final = FakePoster(
        [
            httpx.Response(401, json={"text": "Unauthorized", "code": 4}),
            httpx.Response(403, json={"text": "Invalid token", "code": 4}),
            httpx.Response(400),
            httpx.Response(422),
            httpx.Response(200, json={"text": "No data", "code": 5}),
            httpx.Response(200),
            httpx.Response(503),
            httpx.ConnectError("boom"),
        ]
    )
    sink: Final = SplunkHecSink(SETTINGS, poster)
    batch: Final = _batch()
    assert await sink.deliver(batch) == Undelivered("HTTP 401 from splunk hec: Unauthorized (code 4)")
    assert await sink.deliver(batch) == Undelivered("HTTP 403 from splunk hec: Invalid token (code 4)")
    assert await sink.deliver(batch) == Rejected("HTTP 400 from splunk hec")
    assert await sink.deliver(batch) == Rejected("HTTP 422 from splunk hec")
    assert await sink.deliver(batch) == Undelivered("HTTP 200 from splunk hec: No data (code 5)")
    assert await sink.deliver(batch) == Undelivered("HTTP 200 from splunk hec: no HEC reply body (code -1)")
    assert await sink.deliver(batch) == Undelivered("HTTP 503 from splunk hec: no HEC reply body (code -1)")
    assert await sink.deliver(batch) == Undelivered("ConnectError posting to splunk hec")


@pytest.mark.parametrize("status_code", [401, 403])
@pytest.mark.asyncio
async def test_access_refusals_are_retryable(status_code: int) -> None:
    poster: Final = FakePoster([httpx.Response(status_code)])
    result: Final = await SplunkHecSink(SETTINGS, poster).deliver(_batch())
    assert isinstance(result, Undelivered)


@pytest.mark.asyncio
async def test_permanent_413_rejection_does_not_block_the_next_batch() -> None:
    poster: Final = FakePoster([httpx.Response(413), httpx.Response(200, json={"code": 0})])
    sink: Final = SplunkHecSink(SETTINGS, poster)
    assert await sink.deliver(_batch()[:1]) == Rejected("HTTP 413 from splunk hec")
    assert await sink.deliver(_batch()[:1]) == Delivered(1)
    assert len(poster.posts) == 2


@pytest.mark.asyncio
async def test_413_splits_a_batch_until_both_halves_are_accepted() -> None:
    poster: Final = FakePoster(
        [
            httpx.Response(413),
            httpx.Response(200, json={"code": 0}),
            httpx.Response(200, json={"code": 0}),
        ]
    )
    batch: Final = _batch() * 2
    result: Final = await SplunkHecSink(SETTINGS, poster).deliver(batch)
    assert result == Delivered(4)
    assert len(poster.posts) == 3


@pytest.mark.asyncio
async def test_413_for_a_single_record_is_rejected() -> None:
    poster: Final = FakePoster([httpx.Response(413)])
    assert await SplunkHecSink(SETTINGS, poster).deliver(_batch()[:1]) == Rejected("HTTP 413 from splunk hec")
    assert len(poster.posts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("statuses, rejected", [((400, 200), 1), ((200, 422), 1), ((400, 422), 2)])
async def test_split_rejections_are_counted_alongside_delivered_halves(
    statuses: tuple[int, int], rejected: int
) -> None:
    poster: Final = FakePoster(
        [httpx.Response(413), *(httpx.Response(status, json={"code": 0}) for status in statuses)]
    )
    assert await SplunkHecSink(SETTINGS, poster).deliver(_batch()) == Rejected(
        "split batches were rejected by splunk hec", rejected
    )
    assert len(poster.posts) == 3


@pytest.mark.asyncio
async def test_nested_split_keeps_the_rejected_record_count() -> None:
    poster: Final = FakePoster(
        [
            httpx.Response(413),
            httpx.Response(413),
            httpx.Response(400),
            httpx.Response(200, json={"code": 0}),
            httpx.Response(200, json={"code": 0}),
        ]
    )
    assert await SplunkHecSink(SETTINGS, poster).deliver(_four_record_batch()) == Rejected(
        "split batches were rejected by splunk hec", 1
    )


@pytest.mark.asyncio
async def test_rejected_split_is_counted_once_while_only_the_undelivered_suffix_is_retried(
    caplog: pytest.LogCaptureFixture,
) -> None:
    poster: Final = FakePoster(
        [
            httpx.Response(413),
            httpx.Response(400),
            httpx.Response(503),
            httpx.Response(200, json={"code": 0}),
        ]
    )
    batch: Final = _batch()
    pipeline: Final = build_pipeline((SplunkHecSink(SETTINGS, poster),), queue_size=4, batch_size=2, strict=True)
    for record in batch:
        pipeline.publish(record)
    assert await pipeline.workers[0].flush_once() == 2
    assert pipeline.workers[0].rejected == 1 and not pipeline.accepting()
    assert pipeline.workers[0].queue.empty()
    posted: Final = tuple(content for _, content, _, _ in poster.posts)
    assert posted[1] != posted[2] and posted[2] == posted[3]
    assert json.loads(posted[3])["event"] == batch[1].body
    assert caplog.text.count("sink splunk_hec rejected 1 records") == 1


@pytest.mark.asyncio
async def test_undelivered_split_makes_the_splunk_batch_undelivered() -> None:
    poster: Final = FakePoster([httpx.Response(413), httpx.Response(200, json={"code": 0}), httpx.Response(403)])
    assert isinstance(await SplunkHecSink(SETTINGS, poster).deliver(_batch()), Undelivered)
    assert len(poster.posts) == 3


@pytest.mark.asyncio
async def test_an_undelivered_first_split_does_not_post_the_second_half() -> None:
    poster: Final = FakePoster([httpx.Response(413), httpx.Response(503), httpx.Response(200, json={"code": 0})])

    result: Final = await SplunkHecSink(SETTINGS, poster).deliver(_batch())

    assert isinstance(result, Undelivered) and result.settled == 0
    assert len(poster.posts) == 2


@pytest.mark.asyncio
async def test_pipeline_retries_only_the_unaccepted_splunk_split() -> None:
    poster: Final = FakePoster(
        [
            httpx.Response(413),
            httpx.Response(200, json={"text": "Success", "code": 0}),
            httpx.Response(503),
            httpx.Response(200, json={"text": "Success", "code": 0}),
        ]
    )
    batch: Final = _four_record_batch()
    sink: Final = SplunkHecSink(SETTINGS, poster)

    async def no_sleep(_: float) -> None:
        return None

    pipeline: Final = build_pipeline((sink,), queue_size=10, batch_size=4, strict=False, sleep=no_sleep)
    for record in batch:
        pipeline.publish(record)

    assert await pipeline.workers[0].flush_once() == 4
    posted_counts: Final = tuple(len(content.decode().split("\n")) for _, content, _, _ in poster.posts)
    assert posted_counts == (4, 2, 2, 2)
    accepted_lines: Final = tuple(
        chain_iterables.from_iterable(poster.posts[index][1].decode().split("\n") for index in (1, 3))
    )
    accepted_events: Final = tuple(json.loads(line)["event"] for line in accepted_lines)
    assert accepted_events == tuple(record.body for record in batch)
