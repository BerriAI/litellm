"""Contract: what the spend log pipeline delivers after Postgres was unreachable while requests were served.

`test_dec1_failures_postgres_outage.py` already pins that every outage request gets exactly one
`LiteLLM_SpendLogs` row and that key spend is raised once. Spend Logs V2 moves the queue and flusher into Rust
(the plan keeps "requeue to the queue head" on transient errors, 3 retries with `1 << i` backoff, and
`ON CONFLICT DO NOTHING`), so this file pins the parts the existing file does not: that the queued rows keep their
columns (call type, model group, spend, status, tags, timing), that each gets exactly one ledger delivery that is
then applied, that the daily tables count each outage request once, and that a streamed request served during
the outage is billed from its final usage.

One proxy is booted per module on a scratch database behind a relay (`dec1_failures_rig`). "Postgres down"
means the relay resets every live connection and refuses new ones. Rows are read around the relay.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import datetime
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import eventually, held, list_value, object_value, string_value
from tests.integration.database.dec1_failures_rig import (
    CHAT,
    EMBED,
    LANDING_SECONDS,
    Rig,
    chat,
    database_down,
    digest,
    embed,
    failures_rig,
    marker,
    number,
    readiness,
    served_ids,
    spend_rows,
    stream_chat,
)

pytestmark: Final = pytest.mark.timeout(900)

FLIP_SECONDS: Final = 45


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with failures_rig(tmp_path_factory.mktemp("dec1-spendlogs-postgres-outage")) as booted:
        yield booted


def _warm_key(rig: Rig) -> str:
    """A fresh key, put into the auth cache by one chat and one embedding call while Postgres is up."""
    key: Final = string_value(rig.proxy.post("/key/generate", {})["key"])
    for response in (chat(rig.proxy, key=key), embed(rig.proxy, key=key)):
        assert response.status_code == 200, response.text
    spend_rows(rig.proxy, key, 2)
    return key


def _down(rig: Rig) -> None:
    eventually(lambda: readiness(rig.proxy), lambda seen: seen[0] == 503, seconds=FLIP_SECONDS)


def _rows_of(rig: Rig, ids: tuple[str, ...]) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: rig.proxy.rows(
            'SELECT request_id, call_type, model_group, spend, status, request_tags, "startTime"::text AS start, '
            '"endTime"::text AS finish, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" '
            "WHERE request_id = ANY(%s::text[]) ORDER BY request_id",
            ("{" + ",".join(ids) + "}",),
        ),
        lambda found: len(found) == len(ids),
        seconds=LANDING_SECONDS,
    )


def test_outage_rows_keep_call_type_model_group_and_status(rig: Rig) -> None:
    """A warm key is served one chat and one embedding call while Postgres is down. After recovery the chat row
    has `call_type` `acompletion` and `model_group` `dec1-failures-chat`, the embedding row `aembedding` and
    `dec1-failures-embed`, both `status` `success`: the queued row is the row that was built, not a stub."""
    with rig.proxy.scenario():
        key: Final = _warm_key(rig)
        with database_down(rig):
            _down(rig)
            chat_id: Final = served_ids([chat(rig.proxy, key=key)])[0]
            embed_response: Final = embed(rig.proxy, key=key)
            assert embed_response.status_code == 200, embed_response.text
        rows: Final = spend_rows(rig.proxy, key, 4)
        by_id: Final = {string_value(row["request_id"]): row for row in rows}
        assert (by_id[chat_id]["call_type"], by_id[chat_id]["model_group"]) == ("acompletion", CHAT), by_id
        embedding: Final = [row for row in rows[2:] if row["call_type"] == "aembedding"]
        assert len(embedding) == 1 and embedding[0]["model_group"] == EMBED, rows
        assert {row["status"] for row in rows} == {"success"}, rows


def test_outage_rows_keep_their_request_tags(rig: Rig) -> None:
    """A tagged call (`metadata.tags: [<tag>]`) served during the outage lands after recovery with
    `request_tags` containing that tag."""
    with rig.proxy.scenario():
        key: Final = _warm_key(rig)
        tag: Final = f"dec1-sl-outage-{uuid.uuid4().hex[:8]}"
        with database_down(rig):
            _down(rig)
            identity: Final = served_ids([chat(rig.proxy, key=key, metadata={"tags": [tag]})])[0]
        row: Final = _rows_of(rig, (identity,))[0]
        assert tag in list_value(row["request_tags"]), row


def test_outage_rows_keep_the_time_they_were_served_not_the_time_they_were_written(rig: Rig) -> None:
    """A call served during the outage keeps the `startTime` of the moment it was served: its `startTime` is
    earlier than the `created_at` of the row (written only after Postgres returned), by at least as long as the
    outage lasted after the call (about 5 seconds here, held with `held` rather than a sleep)."""
    with rig.proxy.scenario():
        key: Final = _warm_key(rig)
        with database_down(rig):
            _down(rig)
            identity: Final = served_ids([chat(rig.proxy, key=key)])[0]
            held(lambda: readiness(rig.proxy)[0], lambda status: status == 503, holding=5, seconds=FLIP_SECONDS)
        rows: Final = eventually(
            lambda: rig.proxy.rows(
                'SELECT "startTime"::text AS start, created_at::text AS created FROM "LiteLLM_SpendLogs" '
                "WHERE request_id = %s",
                (identity,),
            ),
            lambda found: len(found) == 1,
            seconds=LANDING_SECONDS,
        )
        gap: Final = datetime.fromisoformat(str(rows[0]["created"])) - datetime.fromisoformat(str(rows[0]["start"]))
        assert gap.total_seconds() >= 4, rows


def test_each_outage_row_gets_exactly_one_applied_delivery(rig: Rig) -> None:
    """Three calls served during the outage each end with exactly one `LiteLLM_SpendLogDelivery` row in state
    `applied`: a requeued insert batch must not insert the delivery twice, and a delivery that waited out the
    outage must still be projected."""
    with rig.proxy.scenario():
        key: Final = _warm_key(rig)
        with database_down(rig):
            _down(rig)
            ids: Final = served_ids([chat(rig.proxy, key=key) for _ in range(3)])
        _rows_of(rig, ids)
        deliveries: Final = eventually(
            lambda: rig.proxy.rows(
                'SELECT request_id, state::text AS state FROM "LiteLLM_SpendLogDelivery" '
                "WHERE request_id = ANY(%s::text[])",
                ("{" + ",".join(ids) + "}",),
            ),
            lambda found: len(found) == 3 and {row["state"] for row in found} == {"applied"},
            seconds=120,
        )
        assert sorted(string_value(row["request_id"]) for row in deliveries) == sorted(ids), deliveries


def test_daily_row_counts_each_outage_request_once(rig: Rig) -> None:
    """A warm key (one chat and one embedding call before) is served three more chat calls during the outage.
    After recovery its `LiteLLM_DailyUserSpend` chat row reaches `api_requests` 4 and stays at 4 for ten
    seconds, and its `spend` equals the sum of the four chat rows' `spend`."""
    with rig.proxy.scenario():
        key: Final = _warm_key(rig)
        with database_down(rig):
            _down(rig)
            ids: Final = served_ids([chat(rig.proxy, key=key) for _ in range(3)])
        _rows_of(rig, ids)
        chat_rows: Final = [row for row in spend_rows(rig.proxy, key, 5) if row["call_type"] == "acompletion"]
        total: Final = sum(number(row["spend"]) for row in chat_rows)

        def daily() -> list[dict[str, JsonValue]]:
            return rig.proxy.rows(
                'SELECT api_requests, spend FROM "LiteLLM_DailyUserSpend" WHERE api_key = %s AND endpoint = %s',
                (digest(key), "/chat/completions"),
            )

        eventually(daily, lambda found: len(found) == 1 and found[0]["api_requests"] == 4, seconds=120)
        settled: Final = held(daily, lambda found: found[0]["api_requests"] == 4, holding=10, seconds=30)
        assert number(settled[0]["spend"]) == pytest.approx(total), (settled, chat_rows)


def test_a_stream_served_during_the_outage_is_billed_from_its_final_usage(rig: Rig) -> None:
    """A streamed chat completion served during the outage lands after recovery as one `success` row with
    non-zero `completion_tokens` and `spend` above 0: the complete-stream charge is queued like any other."""
    with rig.proxy.scenario():
        key: Final = _warm_key(rig)
        with database_down(rig):
            _down(rig)
            status, text = stream_chat(rig.proxy, key=key, text=marker())
            assert status == 200, text
        rows: Final = spend_rows(rig.proxy, key, 3)
        streamed: Final = [row for row in rows[2:]]
        assert len(streamed) == 1 and streamed[0]["status"] == "success", rows
        assert number(streamed[0]["spend"]) > 0, streamed
        metadata: Final = object_value(streamed[0]["metadata"])
        assert number(object_value(metadata["usage_object"])["completion_tokens"]) > 0, metadata
