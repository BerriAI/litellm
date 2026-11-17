"""Contract: when a spend log row lands, its time columns, and the ledger delivery row written with it.

Spend Logs V2 phase 1 replaces the Python flusher (`monitor_spend_logs_queue`, 2 s poll, threshold 100 rows;
`update_spend` every `proxy_batch_write_at` seconds) with a Rust queue and sqlx writer that must keep the same
constants and the same transaction: each `LiteLLM_SpendLogs` row is inserted together with one
`LiteLLM_SpendLogDelivery` row (`ON CONFLICT DO NOTHING`), and the Python ledger worker later claims the
delivery, projects it into the entity and daily tables and marks it `applied`. The delivery row and its
`accounting_payload` are the contract between the Rust writer and the Python projection.

This file also pins the time columns that the flusher must carry unchanged: `startTime`, `completionStartTime`
(first token for a stream) and `endTime`, and `request_duration_ms`, and one surprising current behaviour:
`SPEND_LOGS_URL` is read but the HTTP client that would POST to it is never created
(`litellm/proxy/proxy_server.py:3373`, only ever `None`), so rows still go to Postgres.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually, held, object_value, string_value
from tests.integration._support.wire import Reply, Request, wire_server
from tests.integration.spend.dec1_spend_support import (
    ask,
    chat_model,
    hashed,
    number,
    row_by_id,
    rows_of_key,
    scripted_upstream,
    served_id,
    stream,
    stream_chunks,
)
from tests.integration.spend.dec1_spendlogs_support import (
    SLOW_STREAM_CHUNKS,
    SLOW_STREAM_PAUSE,
    deliveries,
    failing_upstream,
    key_rows,
    owned_spend_proxy,
    slow_stream_upstream,
)

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

FLUSH_SECONDS: Final = 20
"""How long a row may take to appear with `proxy_batch_write_at: 1`: two poll cycles of the 2 s queue monitor
with its 1.5x backoff, plus the ledger worker pass. Far below the 70 s every other file allows."""


def _time(value: object) -> datetime:
    return datetime.fromisoformat(str(value))


def test_a_row_lands_within_seconds_of_the_response(gateway: Gateway) -> None:
    """With the suite's `proxy_batch_write_at: 1`, a served chat call's row is readable within 20 seconds of the
    response. A Rust flusher that only writes at the threshold of 100 rows, or on the 10 s job alone with a
    longer interval, would miss this window on a quiet proxy."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        identity: Final = served_id(ask(gateway, model, key))
        rows: Final = eventually(
            lambda: gateway.rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (identity,)),
            lambda found: len(found) == 1,
            seconds=FLUSH_SECONDS,
        )
        assert rows[0]["request_id"] == identity


def test_streamed_row_completion_start_is_the_first_token_between_start_and_end(gateway: Gateway) -> None:
    """A stream of 40 chunks 0.25 s apart: the row's `completionStartTime` is strictly after `startTime` and at
    least 5 seconds before `endTime`, because the first token arrived long before the last."""
    with slow_stream_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        chunks: Final = stream_chunks(
            stream(gateway, "/v1/chat/completions", {"model": model, "stream": True, "messages": [
                {"role": "user", "content": f"dec1 slow {uuid.uuid4().hex}"}]}, key)
        )  # fmt: skip
        row: Final = row_by_id(gateway, string_value(chunks[0]["id"]))
        start, first, end = _time(row["start_time"]), _time(row["completion_start_time"]), _time(row["end_time"])
        assert start < first, row
        assert (end - first).total_seconds() >= 5, row


def test_streamed_row_duration_covers_the_whole_stream(gateway: Gateway) -> None:
    """The same slow stream lasts about 39 * 0.25 = 9.75 s. The row's `request_duration_ms` is at least 9000
    and equals `endTime - startTime` within 1 ms rounding."""
    with slow_stream_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        chunks: Final = stream_chunks(
            stream(gateway, "/v1/chat/completions", {"model": model, "stream": True, "messages": [
                {"role": "user", "content": f"dec1 slow {uuid.uuid4().hex}"}]}, key)
        )  # fmt: skip
        row: Final = row_by_id(gateway, string_value(chunks[0]["id"]))
        minimum: Final = (SLOW_STREAM_CHUNKS - 1) * SLOW_STREAM_PAUSE * 1000 * 0.92
        assert number(row["request_duration_ms"]) >= minimum, row
        assert abs(number(row["request_duration_ms"]) - number(row["measured_ms"])) <= 1, row


def test_streamed_row_bills_the_final_usage_chunk(gateway: Gateway) -> None:
    """The slow stream ends with usage 10 prompt and 40 completion tokens. The row stores exactly those and
    `spend` 10 * 0.001 + 40 * 0.002 = 0.09: a stream is charged once, on the complete response."""
    with slow_stream_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        chunks: Final = stream_chunks(
            stream(gateway, "/v1/chat/completions", {"model": model, "stream": True, "messages": [
                {"role": "user", "content": f"dec1 slow {uuid.uuid4().hex}"}]}, key)
        )  # fmt: skip
        row: Final = row_by_id(gateway, string_value(chunks[0]["id"]))
        assert (row["prompt_tokens"], row["completion_tokens"]) == (10, SLOW_STREAM_CHUNKS), row
        assert number(row["spend"]) == pytest.approx(0.09), row


def test_non_streamed_row_completion_start_equals_end(gateway: Gateway) -> None:
    """A non-streamed call has no first token separate from the answer: `completionStartTime` equals `endTime`
    (the builder defaults it to `end_time`, spend_tracking_utils.py:716).
    UNVERIFIED: the logging object may stamp its own completion start for non-streamed calls; if so this
    should become `startTime <= completionStartTime <= endTime`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        assert row["completion_start_time"] == row["end_time"], row


def test_each_row_gets_exactly_one_delivery_that_ends_applied(gateway: Gateway) -> None:
    """A served chat call has exactly one `LiteLLM_SpendLogDelivery` row with its `request_id`, `is_batch`
    false, and its `state` reaches `applied` once the ledger worker has projected it."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        identity: Final = served_id(ask(gateway, model, key))
        found: Final = eventually(
            lambda: deliveries(gateway, identity, 1),
            lambda rows: rows[0]["state"] == "applied",
            seconds=90,
        )
        assert found[0]["is_batch"] is False, found


def test_delivery_payload_is_a_json_object(gateway: Gateway) -> None:
    """The delivery row's `accounting_payload` is a non-empty JSON object: the shape the Python projection
    reads. A Rust writer that stores a string or an array breaks every rollup.
    UNVERIFIED: the payload's keys (`litellm/accounting/projection.py`); only its type is pinned here, the plan
    calls for goldens on the full shape."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        identity: Final = served_id(ask(gateway, model, key))
        payload: Final = object_value(deliveries(gateway, identity, 1)[0]["accounting_payload"])
        assert payload, payload


def test_failure_rows_also_get_a_delivery(gateway: Gateway) -> None:
    """A provider 500 failure row has exactly one delivery row too, so the daily `failed_requests` counter is
    projected from the ledger like every other counter."""
    with failing_upstream(500) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = ask(gateway, model, key)
        assert response.status_code == 500, response.text
        deliveries(gateway, response.headers["x-litellm-call-id"], 1)


def test_concurrent_calls_each_land_exactly_once(gateway: Gateway) -> None:
    """Thirty chat calls on one key are sent eight at a time. Exactly thirty rows land, one per response id,
    and each has exactly one delivery row: batching and statement splitting never drop or duplicate a row."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        with ThreadPoolExecutor(max_workers=8) as pool:
            ids: Final = sorted(pool.map(lambda _: served_id(ask(gateway, model, key)), range(30)))
        rows: Final = rows_of_key(gateway, key, 30)
        assert sorted(string_value(row["request_id"]) for row in rows) == ids
        counts: Final = eventually(
            lambda: gateway.rows(
                'SELECT request_id, count(*)::int AS n FROM "LiteLLM_SpendLogDelivery" '
                "WHERE request_id = ANY(%s::text[]) GROUP BY request_id",
                ("{" + ",".join(ids) + "}",),
            ),
            lambda found: len(found) == 30,
            seconds=90,
        )
        assert {row["n"] for row in counts} == {1}, counts


def test_flush_threshold_burst_of_rows_lands_without_loss(gateway: Gateway, tmp_path: Path) -> None:
    """An owned proxy with `SPEND_LOG_QUEUE_SIZE_THRESHOLD=5` and `SPEND_LOG_WRITE_BATCH_MAX_ROWS=3` serves
    twelve calls. All twelve rows land exactly once, so a flush triggered by the threshold and split into
    statements of at most three rows (litellm/constants.py:1938-1940) writes every row."""
    with owned_spend_proxy(
        gateway,
        tmp_path,
        environment={"SPEND_LOG_QUEUE_SIZE_THRESHOLD": "5", "SPEND_LOG_WRITE_BATCH_MAX_ROWS": "3"},
    ) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        ids: Final = sorted(served_id(ask(owned, model, key)) for _ in range(12))
        rows: Final = key_rows(owned, key, 12)
        assert sorted(string_value(row["request_id"]) for row in rows) == ids


def test_spend_logs_url_does_not_divert_rows_from_postgres(gateway: Gateway, tmp_path: Path) -> None:
    """With `SPEND_LOGS_URL` pointing at a scripted receiver, a served call still writes its row to Postgres
    and the receiver gets no `POST /spend/update` within five seconds: the export client is never created
    (`db_writer_client` is only ever None, litellm/proxy/proxy_server.py:3373), so the variable is dead today.
    The V2 plan says the port keeps an optional POST; doing so would change behaviour, so this pins today's."""
    received: Final[list[str]] = []

    def receiver(request: Request) -> Reply:
        received.append(request.target)
        return Reply(body=b"{}")

    with (
        wire_server(receiver) as sink,
        owned_spend_proxy(gateway, tmp_path, environment={"SPEND_LOGS_URL": sink.url}) as (owned, scenario, wire),
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        identity: Final = served_id(ask(owned, model, key))
        assert row_by_id(owned, identity)["request_id"] == identity
        seen: Final = held(lambda: list(received), lambda targets: "/spend/update" not in targets, holding=5)
        assert "/spend/update" not in seen, seen


def test_one_request_id_never_produces_two_rows_after_a_repeat_flush(gateway: Gateway) -> None:
    """The writer inserts with `ON CONFLICT DO NOTHING` on `request_id`. After one call lands, the table holds
    that id once and keeps holding it once for ten seconds, across at least two more flush cycles."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        identity: Final = served_id(ask(gateway, model, key))
        row_by_id(gateway, identity)
        settled: Final = held(
            lambda: gateway.rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE api_key = %s', (hashed(key),)),
            lambda found: len(found) == 1,
            holding=10,
            seconds=30,
        )
        assert [row["request_id"] for row in settled] == [identity], settled
