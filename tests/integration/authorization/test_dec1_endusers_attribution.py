"""How spend is attributed to end users, and the `LiteLLM_EndUserTable` row the proxy creates for them.

Contract, read from litellm/proxy/utils.py:9081-9127 `ProxyUpdateSpend.update_end_user_spend`,
litellm/accounting/projection.py:106, litellm/proxy/db/db_spend_update_writer.py:785-960 and
litellm/proxy/spend_tracking/spend_tracking_utils.py:759-794, 1039:

* An end user needs no prior setup. Its first billed request upserts a row with `user_id`, `spend` equal to the
  request cost and `blocked: false`; every other column (alias, budget, region, default model, object permission)
  stays null. Later requests increment `spend`, whichever key or route they come through.
* The spend row's `end_user` column is the resolved end user id, or "" when none was named; the key owner stays in
  the `user` column, so the two never mix.
* `/customer/info` reads the same row, so an auto created end user is visible there with its spend.
* A request refused before it reaches the provider bills nothing and creates no row.
* `/customer/delete` removes the row; the next request starts a new row from zero.

Not repeated: the first-row behaviour of one request (`test_dec1_dbstores_end_users.py`), daily end user tables
(`spend/test_dec1_spendlogs_daily_and_activity.py`), each route spelling with a budget
(`spend/test_end_user_budget_enforcement.py`), and `disable_end_user_cost_tracking`
(`spend/test_disable_end_user_cost_tracking.py`).
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually, object_value
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_dbstores_support import assert_ok, chat_as, forget_end_user
from tests.integration.authorization.dec1_endusers_support import (
    END_USER_HEADER,
    SPEND_SECONDS,
    end_user_row,
    landed_end_user_row,
    new_end_user,
    spend_rows,
)
from tests.integration.authorization.dec1_limits_support import complete, embed, provider


def _total(rows: list[dict[str, object]]) -> float:
    return sum(float(str(row["spend"])) for row in rows)


def test_an_auto_created_end_user_row_has_only_its_id_spend_and_blocked_false(gateway: Gateway) -> None:
    """A never seen end user sends one chat request through a plain key.

    Its new `LiteLLM_EndUserTable` row has `blocked: false`, `spend` equal to the spend row's `spend`, and null
    `alias`, `budget_id`, `allowed_model_region`, `default_model` and `object_permission_id`: the upsert's `create`
    branch writes only `user_id`, `spend` and `blocked` (utils.py:9105-9115).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(gateway, key, model, user=end_user))
        logged: Final = spend_rows(gateway, key)
        row: Final = landed_end_user_row(gateway, end_user)
        assert row["blocked"] is False, row
        assert {name: row[name] for name in ("alias", "budget_id", "allowed_model_region", "default_model")} == {
            "alias": None,
            "budget_id": None,
            "allowed_model_region": None,
            "default_model": None,
        }, row
        assert row["object_permission_id"] is None, row
        assert float(str(row["spend"])) == pytest.approx(_total(logged)), (row, logged)


def test_an_end_users_row_spend_is_the_sum_of_its_requests(gateway: Gateway) -> None:
    """One end user sends three chat requests through one key.

    Once all three spend rows landed, the end user row's `spend` equals the sum of their `spend` values (the update
    branch is `{"increment": cost}`, utils.py:9113), so no request is lost or double counted.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        for _ in range(3):
            assert_ok(chat_as(gateway, key, model, user=end_user))
        logged: Final = spend_rows(gateway, key, at_least=3)
        expected: Final = _total(logged)
        assert expected > 0, logged
        eventually(
            lambda: end_user_row(gateway, end_user),
            lambda row: row is not None and float(str(row["spend"])) == pytest.approx(expected),
            seconds=SPEND_SECONDS,
        )


def test_one_end_user_on_two_keys_has_one_row_with_both_keys_spend(gateway: Gateway) -> None:
    """The same end user id is sent as body `user` through key A and through key B.

    There is one `LiteLLM_EndUserTable` row for the id, and its `spend` is the sum of both keys' spend rows: end
    users are global, not scoped to a key.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        first_key: Final = scenario.key()
        second_key: Final = scenario.key()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(gateway, first_key, model, user=end_user))
        assert_ok(chat_as(gateway, second_key, model, user=end_user))
        expected: Final = _total(spend_rows(gateway, first_key)) + _total(spend_rows(gateway, second_key))
        eventually(
            lambda: gateway.rows('SELECT spend FROM "LiteLLM_EndUserTable" WHERE user_id=%s', (end_user,)),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) == pytest.approx(expected),
            seconds=SPEND_SECONDS,
        )


def test_the_spend_row_keeps_the_key_owner_and_the_end_user_in_separate_columns(gateway: Gateway) -> None:
    """A key owned by an internal user sends a chat request with body `user: <end user>`.

    The spend row's `user` column is the key owner and its `end_user` column is the end user; the end user never
    replaces the owner (spend_tracking_utils.py:759-794 reads only `user_api_key_end_user_id` for `end_user`).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        owner: Final = scenario.user()
        key: Final = scenario.key(user_id=owner)
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(gateway, key, model, user=end_user))
        row: Final = spend_rows(gateway, key)[0]
        assert (row["user"], row["end_user"]) == (owner, end_user), row


def test_an_embedding_with_a_user_is_attributed_to_that_end_user(gateway: Gateway) -> None:
    """A key sends `/v1/embeddings` with `user: <id>` to a scripted provider that answers embeddings.

    The spend row's `end_user` is `<id>` and the end user row is created: attribution is not chat only.
    """
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1", model="openai/text-embedding-3-small")
        key: Final = scenario.key()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(embed(gateway, model, key, user=end_user))
        assert [row["end_user"] for row in spend_rows(gateway, key)] == [end_user]
        landed_end_user_row(gateway, end_user)


def test_a_text_completion_with_a_user_is_attributed_to_that_end_user(gateway: Gateway) -> None:
    """A key sends `/v1/completions` with `user: <id>` to a scripted provider that answers text completions.

    The spend row's `end_user` is `<id>`. UNVERIFIED: that the scripted provider in dec1_limits_support answers
    `/v1/completions` with a usage block the proxy prices above zero.
    """
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(complete(gateway, model, key, user=end_user))
        assert [row["end_user"] for row in spend_rows(gateway, key)] == [end_user]


def test_customer_info_reports_an_auto_created_end_user_with_its_spend(gateway: Gateway) -> None:
    """A never seen end user sends one chat request; then `GET /customer/info?end_user_id=<id>` is called.

    It answers 200 with `user_id` `<id>`, `blocked` false, `budget_id` null, `litellm_budget_table` null and `spend`
    equal to the row's spend (customer_endpoints.py `end_user_info` reads `LiteLLM_EndUserTable`).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(gateway, key, model, user=end_user))
        row: Final = landed_end_user_row(gateway, end_user)
        info: Final = gateway.get("/customer/info", {"end_user_id": end_user})
        assert (info["user_id"], info["blocked"], info["budget_id"], info["litellm_budget_table"]) == (
            end_user,
            False,
            None,
            None,
        ), info
        assert float(str(info["spend"])) == pytest.approx(float(str(row["spend"]))), (info, row)


def test_a_request_refused_at_auth_creates_no_end_user_row(gateway: Gateway) -> None:
    """A key limited to model A asks for model B with `user: <refused id>`; then the same key asks for model A with
    `user: <served id>`.

    The first answer is the 403 `key_model_access_denied` refusal and the second is served. After the served id's
    row has landed, there is still no row for the refused id: nothing was billed, so nothing was upserted.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        refused_id: Final = new_end_user()
        served_id: Final = new_end_user()
        forget_end_user(scenario, refused_id)
        forget_end_user(scenario, served_id)
        refused: Final = chat_as(gateway, key, other, user=refused_id)
        assert refused.status_code == 403, refused.text
        assert_ok(chat_as(gateway, key, allowed, user=served_id))
        landed_end_user_row(gateway, served_id)
        assert end_user_row(gateway, refused_id) is None


def test_a_provider_failure_is_logged_against_the_end_user(gateway: Gateway) -> None:
    """A key sends a chat request with `user: <id>` to a provider that answers 500.

    The caller gets an error, and the failure spend row (status `failure`) still carries `end_user` `<id>` with
    `spend` 0, so failed calls can be traced to the customer. UNVERIFIED: that the failure row is written with the
    end user filled (the failure path builds its payload through the same `get_logging_payload_async`).
    """
    with wire_server(provider(status=500)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        failed: Final = chat_as(gateway, key, model, user=end_user)
        assert failed.status_code >= 400, failed.text
        row: Final = spend_rows(gateway, key)[0]
        assert (row["status"], row["end_user"], float(str(row["spend"]))) == ("failure", end_user, 0.0), row


def test_deleting_an_auto_created_end_user_restarts_its_spend_from_zero(gateway: Gateway) -> None:
    """An end user sends one request and its row lands; `/customer/delete` removes it; the end user sends one more
    request through a second key.

    The recreated row's `spend` equals only the second key's spend, not the sum of both: delete drops the row, and
    the upsert creates it again from the new cost.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        first_key: Final = scenario.key()
        second_key: Final = scenario.key()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(gateway, first_key, model, user=end_user))
        landed_end_user_row(gateway, end_user)
        gateway.post("/customer/delete", {"user_ids": [end_user]})
        assert end_user_row(gateway, end_user) is None
        assert_ok(chat_as(gateway, second_key, model, user=end_user))
        expected: Final = _total(spend_rows(gateway, second_key))
        eventually(
            lambda: end_user_row(gateway, end_user),
            lambda row: row is not None and float(str(row["spend"])) == pytest.approx(expected),
            seconds=SPEND_SECONDS,
        )


def test_a_header_named_end_user_is_billed_like_a_body_user(gateway: Gateway) -> None:
    """One end user sends one request named by body `user` and one named by `x-litellm-end-user-id`, on one key.

    Both spend rows carry the same `end_user`, and the single end user row's `spend` is their sum: the source of the
    id does not split the customer's balance.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(gateway, key, model, user=end_user))
        assert_ok(chat_as(gateway, key, model, headers={END_USER_HEADER: end_user}))
        logged: Final = spend_rows(gateway, key, at_least=2)
        assert [row["end_user"] for row in logged] == [end_user, end_user], logged
        expected: Final = _total(logged)
        eventually(
            lambda: end_user_row(gateway, end_user),
            lambda row: row is not None and float(str(row["spend"])) == pytest.approx(expected),
            seconds=SPEND_SECONDS,
        )


def test_the_customer_info_spend_matches_the_spend_row_metadata_end_user(gateway: Gateway) -> None:
    """An end user named by `metadata.user_id` sends one request.

    The spend row metadata's `user_api_key_end_user_id` is the end user, and `/customer/info` for it answers the
    same `user_id` with `spend` above zero: every view of the attribution agrees.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(gateway, key, model, metadata={"user_id": end_user}))
        row: Final = spend_rows(gateway, key)[0]
        assert object_value(row["metadata"]).get("user_api_key_end_user_id") == end_user, row
        landed_end_user_row(gateway, end_user)
        info: Final = gateway.get("/customer/info", {"end_user_id": end_user})
        assert info["user_id"] == end_user and float(str(info["spend"])) > 0, info
