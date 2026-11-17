"""Contract: the delivery semantics between the spend log row writer and the ledger worker.

In Spend Logs V2 the Rust flusher's job ends at a committed `LiteLLM_SpendLogs` row plus its
`LiteLLM_SpendLogDelivery` row, and the existing Python ledger worker (`litellm/accounting/worker.py:48`)
projects them into the entity and daily tables (plan, "DB write and batching path"). The writer's transaction
today (`SpendLogsRepository._insert_ledger`, `litellm/repositories/table_repositories.py:185`) is:

1. Insert one delivery per row, keyed by `accounting_delivery_id` (a fresh uuid4 per `update_database` call), or
   by `request_id` for settle-once batch and video cost rows (`_delivery_id`, `:159`), with
   `ON CONFLICT (delivery_id) DO NOTHING`.
2. Insert the rows whose delivery was accepted with `ON CONFLICT DO NOTHING` on `request_id`
   (`spend_log_insert.py`), skipping rows marked `accounting_log_disabled` (the `disable_spend_logs` case).

So `request_id` is the row's identity but not the charge's: two charges that share a response id produce one row
and two deliveries. The projection applies entity and daily contributions in one transaction per batch, so key
spend and daily counters move together. A client may choose `litellm_call_id` with the `x-litellm-call-id`
header (up to 256 characters, `resolve_litellm_call_id`, `common_request_processing.py:433`).

Round 1 (`test_dec1_spendlogs_timing_and_delivery.py`) already pins one applied delivery per row, concurrency
without loss and the repeat flush; these tests pin what is left.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration.spend.dec1_spend_support import (
    LANDING_SECONDS,
    ask,
    chat_model,
    daily_row,
    hashed,
    number,
    row_by_id,
    scripted_upstream,
    served,
    served_id,
    spend_of,
    spend_reaches,
)
from tests.integration.spend.dec1_spendlogs2_support import (
    call_id_of,
    daily_user_counters,
    fixed_id_upstream,
    fresh_call_id,
    rows_with_call_id,
)
from tests.integration.spend.dec1_spendlogs_support import deliveries, owned_spend_proxy

pytestmark: Final = pytest.mark.shares_worker_state


def test_two_charges_with_one_response_id_keep_one_row_but_bill_twice(gateway: Gateway) -> None:
    """A provider answers two chat calls with the same response id. Only one `LiteLLM_SpendLogs` row exists for
    that id (`ON CONFLICT DO NOTHING`), but each call got its own delivery, so the key is charged 0.04.
    UNVERIFIED as end to end behaviour; read from `_insert_ledger` (table_repositories.py:185). If the V2 port
    keys deliveries by `request_id` instead, the second charge disappears."""
    response_id: Final = f"chatcmpl-dec1-sl2-fixed-{uuid.uuid4().hex}"
    with fixed_id_upstream(response_id) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        assert served_id(ask(gateway, model, key)) == response_id
        assert served_id(ask(gateway, model, key)) == response_id
        assert spend_reaches(gateway, "key", hashed(key), 0.04) == pytest.approx(0.04)
        rows: Final = gateway.rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (response_id,))
        assert len(rows) == 1, rows


def test_two_charges_with_one_response_id_have_two_deliveries(gateway: Gateway) -> None:
    """The same two calls leave two `LiteLLM_SpendLogDelivery` rows with that `request_id`, distinct
    `delivery_id`s, both ending `applied`."""
    response_id: Final = f"chatcmpl-dec1-sl2-fixed-{uuid.uuid4().hex}"
    with fixed_id_upstream(response_id) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key))
        served(ask(gateway, model, key))
        found: Final = eventually(
            lambda: deliveries(gateway, response_id, 2),
            lambda rows: all(row["state"] == "applied" for row in rows),
            seconds=LANDING_SECONDS,
        )
        assert len({row["delivery_id"] for row in found}) == 2, found


def test_two_charges_with_one_response_id_count_two_requests_in_the_daily_row(gateway: Gateway) -> None:
    """The daily user row of that key reaches `api_requests` 2 and `spend` 0.04: the daily tables are projected
    from deliveries, not from distinct rows."""
    response_id: Final = f"chatcmpl-dec1-sl2-fixed-{uuid.uuid4().hex}"
    with fixed_id_upstream(response_id) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key))
        served(ask(gateway, model, key))
        counters: Final = eventually(
            lambda: daily_user_counters(gateway, key),
            lambda found: found is not None and found["api_requests"] == 2,
            seconds=LANDING_SECONDS,
        )
        assert counters is not None and number(counters["spend"]) == pytest.approx(0.04), counters


def test_a_client_chosen_call_id_is_stored_as_the_rows_litellm_call_id(gateway: Gateway) -> None:
    """A chat call sent with `x-litellm-call-id: <id>` is answered with the same header value, and its row has
    `litellm_call_id` equal to it while `request_id` stays the provider's `chatcmpl-` id."""
    call_id: Final = fresh_call_id()
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = ask(gateway, model, key, headers={"x-litellm-call-id": call_id})
        assert call_id_of(response) == call_id
        row: Final = row_by_id(gateway, served_id(response))
        assert row["litellm_call_id"] == call_id, row
        assert str(row["request_id"]).startswith("chatcmpl-dec1-"), row


def test_reusing_a_client_call_id_writes_two_rows_that_share_it(gateway: Gateway) -> None:
    """Two calls sent with the same `x-litellm-call-id` produce two rows (distinct provider ids) that both carry
    that `litellm_call_id`: the call id is client settable and is not the row's identity
    (`_find_spend_log_owners` docstring, spend_management_endpoints.py:477)."""
    call_id: Final = fresh_call_id()
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        first: Final = served_id(ask(gateway, model, key, headers={"x-litellm-call-id": call_id}))
        second: Final = served_id(ask(gateway, model, key, headers={"x-litellm-call-id": call_id}))
        rows: Final = rows_with_call_id(gateway, call_id, 2)
        assert {row["request_id"] for row in rows} == {first, second}, rows


def test_a_call_id_longer_than_256_characters_is_replaced_by_a_uuid(gateway: Gateway) -> None:
    """A 257 character `x-litellm-call-id` is ignored (`MAX_LITELLM_CALL_ID_LENGTH` 256, litellm/constants.py:242):
    the response header and the row's `litellm_call_id` carry a fresh 36 character uuid instead."""
    too_long: Final = "x" * 257
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = ask(gateway, model, key, headers={"x-litellm-call-id": too_long})
        assigned: Final = call_id_of(response)
        assert assigned != too_long and len(assigned) == 36 and str(uuid.UUID(assigned)) == assigned
        assert row_by_id(gateway, served_id(response))["litellm_call_id"] == assigned


def test_the_row_and_its_delivery_commit_before_any_projection_is_visible(gateway: Gateway) -> None:
    """Whenever the key's daily user row shows the call (`api_requests` 1), the request row and an `applied`
    delivery for it already exist, read in the same instant: projection only ever reads committed rows and
    deliveries. A writer that projected before committing the row would fail this."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        request_id: Final = served_id(ask(gateway, model, key))
        eventually(
            lambda: daily_user_counters(gateway, key),
            lambda found: found is not None and found["api_requests"] == 1,
            seconds=LANDING_SECONDS,
        )
        assert gateway.rows('SELECT 1 FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (request_id,)) != []
        states: Final = gateway.rows(
            'SELECT state::text AS state FROM "LiteLLM_SpendLogDelivery" WHERE request_id = %s', (request_id,)
        )
        assert [row["state"] for row in states] == ["applied"], states


def test_key_spend_and_the_daily_row_move_in_the_same_projection(gateway: Gateway) -> None:
    """As soon as the key's stored spend reaches 0.02, its daily user row already shows `spend` 0.02 and
    `api_requests` 1, with no extra wait: entity and daily contributions of one delivery are applied in one
    transaction (`flush_accounting_deliveries`, accounting/worker.py:48)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key))
        spend_reaches(gateway, "key", hashed(key), 0.02)
        counters: Final = daily_user_counters(gateway, key)
        assert counters is not None, "key spend moved before the daily row existed"
        assert (number(counters["spend"]), counters["api_requests"]) == (pytest.approx(0.02), 1), counters


def test_the_daily_row_never_runs_ahead_of_key_spend(gateway: Gateway) -> None:
    """The reverse ordering: once the daily user row shows `api_requests` 1, the key's stored spend is already
    0.02, read immediately."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key))
        eventually(
            lambda: daily_user_counters(gateway, key),
            lambda found: found is not None and found["api_requests"] == 1,
            seconds=LANDING_SECONDS,
        )
        assert spend_of(gateway, "key", hashed(key)) == pytest.approx(0.02)


def test_a_delivery_committed_by_a_stopped_proxy_is_projected_by_another(gateway: Gateway, tmp_path: Path) -> None:
    """An owned proxy on the worker's database serves one call and is stopped as soon as the request row exists.
    The key's spend still reaches 0.02 and its daily row `api_requests` 1, projected by the suite proxy's ledger
    worker: committed deliveries outlive the process that wrote them (plan, "Failure semantics today"), which is
    what lets a Rust writer hand off to the Python projector."""
    with owned_spend_proxy(gateway, tmp_path) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        request_id: Final = served_id(ask(owned, model, key))
        row_by_id(owned, request_id)
        token: Final = hashed(key)
    assert spend_reaches(gateway, "key", token, 0.02) == pytest.approx(0.02)
    counters: Final = eventually(
        lambda: gateway.rows(
            'SELECT COALESCE(SUM(api_requests), 0)::int AS n FROM "LiteLLM_DailyUserSpend" WHERE api_key = %s',
            (token,),
        ),
        lambda rows: rows[0]["n"] == 1,
        seconds=LANDING_SECONDS,
    )
    assert counters[0]["n"] == 1


def test_disable_spend_logs_keeps_the_delivery_but_never_the_row(gateway: Gateway, tmp_path: Path) -> None:
    """With `general_settings.disable_spend_logs: true` a served call leaves no `LiteLLM_SpendLogs` row while its
    delivery still reaches `applied` and the daily user row still counts it, because `_insert_ledger` drops only
    rows marked `accounting_log_disabled`. The delivery is looked up by the response id."""
    with owned_spend_proxy(gateway, tmp_path, general_settings={"disable_spend_logs": True}) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        request_id: Final = served_id(ask(owned, model, key))
        eventually(
            lambda: deliveries(owned, request_id, 1),
            lambda rows: rows[0]["state"] == "applied",
            seconds=LANDING_SECONDS,
        )
        assert owned.rows('SELECT 1 FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (request_id,)) == []
        assert number(daily_row(owned, "LiteLLM_DailyUserSpend", key)["api_requests"]) == 1


def test_a_stop_drains_every_queued_row_exactly_once(gateway: Gateway, tmp_path: Path) -> None:
    """Fifteen calls are served by an owned proxy with `proxy_batch_write_at: 60` (so nothing flushes on the
    timer) and the proxy is stopped right after the last answer. After the stop exactly fifteen rows exist for
    the key, each request id once: the shutdown drain (up to 20 passes, proxy_server.py:1422) must not lose or
    duplicate rows, which the V2 plan moves to `Callback::flush` (PR 1)."""
    with owned_spend_proxy(gateway, tmp_path, general_settings={"proxy_batch_write_at": 60}) as (
        owned,
        scenario,
        wire,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        ids: Final = [served_id(ask(owned, model, key)) for _ in range(15)]
        token: Final = hashed(key)
    rows: Final = eventually(
        lambda: gateway.rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE api_key = %s', (token,)),
        lambda found: len(found) >= 15,
        seconds=LANDING_SECONDS,
    )
    assert sorted(str(row["request_id"]) for row in rows) == sorted(ids), rows
