"""End user budgets and rate limits: when they refuse, with what, and how they combine with key limits.

Contract, read from:

* litellm/proxy/auth/auth_checks.py:2010-2050 `check_end_user_budget` and
  litellm/rust/crates/callbacks-budgets/src/checks.rs:430-452: an end user whose budget row has `max_budget` is
  refused when its spend is strictly above it, with 422 `budget_exceeded` and the message
  `ExceededBudget: End User=<id> over budget. Spend=<spend>, Budget=<max_budget>`. Info routes such as `/v1/models`
  are never refused (`RouteChecks.is_info_route`).
* litellm/proxy/management_endpoints/customer_endpoints.py:271-292, 450-470: `/customer/new` with any budget field
  (`max_budget`, `rpm_limit`, `tpd_limit`, ...) creates a budget row for the customer; `spend` is written on the
  customer row itself.
* litellm/proxy/auth/user_api_key_auth.py:1273-1300, 2862-2899, 3567: the budget row's `rpm_limit` / `tpm_limit`
  become the request's end user limits; with `max_end_user_budget_id` set, a never seen end user gets the default
  budget's limits even before it has a row.
* litellm/proxy/hooks/parallel_request_limiter_v3.py:3364-3376: the `end_user` descriptor is keyed by the end user
  id alone, so its window is shared by every key and every way of naming the end user, and is independent of the
  `api_key` descriptor. `tpd_limit` is read only by the batch limiter (litellm/proxy/hooks/batch_rate_limiter.py:347).

Not repeated: route spellings of a budget refusal (`spend/test_end_user_budget_enforcement.py`), the end user 429
headers and raising the rpm (`test_dec1_ratelim_*`), per-model end user limits
(`management/test_customer_model_limits.py`), the default budget's `max_budget`
(`test_dec1_dbstores_end_user_settings.py`), and auth order against other spent levels (`test_dec1_budgets3_*`).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Final

import pytest

from tests.integration._support.client import Gateway
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_dbstores_support import assert_ok, chat_as, customer, forget_end_user
from tests.integration.authorization.dec1_endusers_support import (
    CUSTOMER_HEADER,
    END_USER_HEADER,
    assert_end_user_over_budget,
    end_user_row,
    new_end_user,
    owned_endusers_proxy,
    text_marker,
    upstream_bodies,
)
from tests.integration.authorization.dec1_keys_support import digest
from tests.integration.authorization.dec1_limits_support import chat, posts, provider, shared_redis
from tests.integration.authorization.dec1_ratelim_support import assert_refused_by, requests_counter

pytestmark: Final = pytest.mark.timeout(300)

_DEFAULT_BUDGET_ID: Final = f"dec1-endusers-default-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def defaulting(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    """A proxy whose `max_end_user_budget_id` names a budget row created by the first test that needs it."""
    yield from owned_endusers_proxy(
        tmp_path_factory.mktemp("dec1-endusers-defaulting"),
        litellm_settings={"max_end_user_budget_id": _DEFAULT_BUDGET_ID},
    )


def _ensure_default_budget(proxy: Gateway) -> None:
    """The default budget allows one request per minute and has no `max_budget`."""
    if not proxy.rows('SELECT budget_id FROM "LiteLLM_BudgetTable" WHERE budget_id=%s', (_DEFAULT_BUDGET_ID,)):
        proxy.post("/budget/new", {"budget_id": _DEFAULT_BUDGET_ID, "rpm_limit": 1})


def test_a_customer_above_its_budget_is_refused_on_its_first_request(gateway: Gateway) -> None:
    """`/customer/new` creates a customer with `spend: 0.5` and `max_budget: 0.1`; it sends its first chat request.

    The answer is 422 with message `ExceededBudget: End User=<id> over budget. Spend=0.5, Budget=0.1`, type
    `budget_exceeded`, param null, code "422"; the provider receives nothing and the row's spend stays 0.5. The
    initial `spend` is the fallback the check reads when no spend counter exists (auth_checks.py:2034-2039).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = customer(scenario, user_id=new_end_user(), spend=0.5, max_budget=0.1)
        text: Final = text_marker()
        assert_end_user_over_budget(chat_as(gateway, key, model, user=end_user, text=text), end_user, "0.5", "0.1")
        assert upstream_bodies(gateway, text) == []
        row: Final = end_user_row(gateway, end_user)
        assert row is not None and float(str(row["spend"])) == pytest.approx(0.5), row


def test_an_over_budget_customer_is_refused_on_every_key(gateway: Gateway) -> None:
    """The same over-budget customer (`spend: 0.5`, `max_budget: 0.1`) calls through two unrelated keys, once by body
    `user` and once by `x-litellm-customer-id`.

    Both are refused with the same end user message: the budget belongs to the end user, not to a key, and the
    source of the id does not matter.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        first_key: Final = scenario.key()
        second_key: Final = scenario.key()
        end_user: Final = customer(scenario, user_id=new_end_user(), spend=0.5, max_budget=0.1)
        assert_end_user_over_budget(chat_as(gateway, first_key, model, user=end_user), end_user, "0.5", "0.1")
        assert_end_user_over_budget(
            chat_as(gateway, second_key, model, headers={CUSTOMER_HEADER: end_user}), end_user, "0.5", "0.1"
        )


def test_an_over_budget_customer_does_not_block_its_key_for_others(gateway: Gateway) -> None:
    """After an over-budget customer is refused on a key, the same key sends one request with no end user and one
    for a different, new end user.

    Both are served with 200: an end user refusal never counts against the key.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = customer(scenario, user_id=new_end_user(), spend=0.5, max_budget=0.1)
        other: Final = new_end_user()
        forget_end_user(scenario, other)
        assert_end_user_over_budget(chat_as(gateway, key, model, user=end_user), end_user, "0.5", "0.1")
        assert_ok(chat_as(gateway, key, model))
        assert_ok(chat_as(gateway, key, model, user=other))


def test_an_over_budget_customer_can_still_list_models(gateway: Gateway) -> None:
    """An over-budget customer is named by `x-litellm-end-user-id` on `GET /v1/models`.

    The answer is 200: `check_end_user_budget` returns early on info routes (auth_checks.py:2024-2025), so a
    customer out of budget can still see what it could call.
    """
    with gateway.scenario() as scenario:
        scenario.model()
        key: Final = scenario.key()
        end_user: Final = customer(scenario, user_id=new_end_user(), spend=0.5, max_budget=0.1)
        response: Final = gateway.request("GET", "/v1/models", key=key, headers={END_USER_HEADER: end_user})
        assert response.status_code == 200, response.text


def test_a_customer_with_a_budget_but_no_spend_yet_is_served(gateway: Gateway) -> None:
    """A customer is created with `max_budget: 0.1` and no spend; it sends one small chat request.

    It is served with 200: only spend strictly above the budget refuses at auth (checks.rs:435).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = customer(scenario, user_id=new_end_user(), max_budget=0.1)
        assert_ok(chat_as(gateway, key, model, user=end_user))


def test_the_master_key_does_not_bypass_an_end_user_budget(gateway: Gateway) -> None:
    """The master key sends a chat request for an over-budget customer (`spend: 0.5`, `max_budget: 0.1`).

    Intended: refused with 422 `budget_exceeded`, because the end user, not the caller, owns the budget.
    UNVERIFIED: the master key returns from the auth builder before `common_checks` (user_api_key_auth.py:3046-3088)
    with the end user params attached, so the refusal may come from the Rust admission hold with its own wording;
    only the status and type are pinned.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        end_user: Final = customer(scenario, user_id=new_end_user(), spend=0.5, max_budget=0.1)
        response: Final = chat_as(gateway, gateway.key, model, user=end_user)
        assert response.status_code == 422, response.text
        assert response.json()["error"]["type"] == "budget_exceeded", response.text


def test_deleting_an_over_budget_customer_lets_its_id_call_again(gateway: Gateway) -> None:
    """An over-budget customer is refused, then `/customer/delete` removes it, then the same id calls again.

    The second call is served with 200: delete evicts the cached row and floors the spend counter to 0
    (customer_endpoints.py:850-856), and the id is now an unknown end user without a budget.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = customer(scenario, user_id=new_end_user(), spend=0.5, max_budget=0.1)
        assert_end_user_over_budget(chat_as(gateway, key, model, user=end_user), end_user, "0.5", "0.1")
        gateway.post("/customer/delete", {"user_ids": [end_user]})
        assert_ok(chat_as(gateway, key, model, user=end_user))


def test_updating_an_over_budget_customers_alias_keeps_it_refused(gateway: Gateway) -> None:
    """An over-budget customer is refused; `/customer/update` then changes only its `alias`.

    The next request is still refused with the same message: update writes only the fields that were sent
    (`fields_set`, customer_endpoints.py:680-683), so neither the budget link nor the spend is reset.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = customer(scenario, user_id=new_end_user(), spend=0.5, max_budget=0.1)
        assert_end_user_over_budget(chat_as(gateway, key, model, user=end_user), end_user, "0.5", "0.1")
        gateway.post("/customer/update", {"user_id": end_user, "alias": "dec1-endusers-renamed"})
        assert_end_user_over_budget(chat_as(gateway, key, model, user=end_user), end_user, "0.5", "0.1")


def test_moving_an_over_budget_customer_to_a_larger_budget_readmits_it(gateway: Gateway) -> None:
    """An over-budget customer (`spend: 0.5`, `max_budget: 0.1`) is refused; `/customer/update` then links it to a
    budget row with `max_budget: 1.0`.

    The next request is served within a few seconds: the update evicts the customer's cached row, and 0.5 is below
    the new budget.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = customer(scenario, user_id=new_end_user(), spend=0.5, max_budget=0.1)
        larger: Final = scenario.budget(max_budget=1.0)
        assert_end_user_over_budget(chat_as(gateway, key, model, user=end_user), end_user, "0.5", "0.1")
        gateway.post("/customer/update", {"user_id": end_user, "budget_id": larger})
        assert_ok(chat_as(gateway, key, model, user=end_user))


def test_an_end_user_rpm_window_is_shared_across_keys(gateway: Gateway) -> None:
    """`/customer/new` creates a customer with `rpm_limit: 1`. It is served once through key A, then calls through
    key B.

    The second call is the v3 limiter's 429 for `end_user: <id>` (requests, limit 1): the descriptor value is the end
    user id alone (parallel_request_limiter_v3.py:3368-3370).
    """
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        first_key: Final = scenario.key(models=[model])
        second_key: Final = scenario.key(models=[model])
        end_user: Final = customer(scenario, user_id=new_end_user(), rpm_limit=1)
        assert_ok(chat(gateway, model, first_key, user=end_user))
        assert_refused_by(chat(gateway, model, second_key, user=end_user), "end_user", end_user, "requests", 1)


def test_end_users_sharing_a_budget_row_have_separate_rpm_windows(gateway: Gateway) -> None:
    """Two customers are linked to the same budget row with `rpm_limit: 1`; each sends one request on one key.

    Both are served with 200: a shared budget row shares the limit value, not the window.
    """
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model])
        budget: Final = scenario.budget(rpm_limit=1)
        first: Final = customer(scenario, user_id=new_end_user(), budget_id=budget)
        second: Final = customer(scenario, user_id=new_end_user(), budget_id=budget)
        assert_ok(chat(gateway, model, key, user=first))
        assert_ok(chat(gateway, model, key, user=second))


def test_a_tighter_key_rpm_refuses_before_the_end_user_rpm(gateway: Gateway) -> None:
    """A key has `rpm_limit: 1`; a customer has `rpm_limit: 5`. The customer sends two requests on that key.

    The second is the 429 for `api_key: <key hash>`, not for the end user: each descriptor is checked against its
    own window, and only the key's is full.
    """
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        end_user: Final = customer(scenario, user_id=new_end_user(), rpm_limit=5)
        assert_ok(chat(gateway, model, key, user=end_user))
        assert_refused_by(chat(gateway, model, key, user=end_user), "api_key", digest(key), "requests", 1)


def test_a_tighter_end_user_rpm_refuses_only_that_end_user_on_the_key(gateway: Gateway) -> None:
    """A key has `rpm_limit: 5`; a customer has `rpm_limit: 1`. The customer sends two requests, then the key sends
    one with no end user.

    The customer's second request is the 429 for `end_user: <id>` and the provider sees only one of its requests;
    the key's own request is served because its window still has room.
    """
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=5)
        end_user: Final = customer(scenario, user_id=new_end_user(), rpm_limit=1)
        assert_ok(chat(gateway, model, key, user=end_user))
        assert_refused_by(chat(gateway, model, key, user=end_user), "end_user", end_user, "requests", 1)
        assert len(posts(wire)) == 1
        assert_ok(chat(gateway, model, key))


def test_a_header_and_a_body_user_share_one_end_user_window(gateway: Gateway) -> None:
    """A customer with `rpm_limit: 1` is served once when named by body `user`, then sends a request naming it only
    by `x-litellm-customer-id`.

    The second is the 429 for `end_user: <id>`: the window is keyed by the resolved id, not by where it came from.
    """
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model])
        end_user: Final = customer(scenario, user_id=new_end_user(), rpm_limit=1)
        assert_ok(chat(gateway, model, key, user=end_user))
        assert_refused_by(
            chat(gateway, model, key, headers={CUSTOMER_HEADER: end_user}), "end_user", end_user, "requests", 1
        )


def test_a_served_end_user_request_counts_in_the_end_user_redis_window(gateway: Gateway) -> None:
    """A customer with `rpm_limit: 10` is served once.

    The shared Redis holds `{end_user:<id>}:requests` with value 1 (key layout in
    litellm-rust/crates/redis-store/src/rate_limits/keys.rs, read through `dec1_limits_support.counter_key`), so a
    second proxy instance sees the same window.
    """
    with wire_server(provider()) as wire, gateway.scenario() as scenario, shared_redis() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model])
        end_user: Final = customer(scenario, user_id=new_end_user(), rpm_limit=10)
        assert_ok(chat(gateway, model, key, user=end_user))
        assert requests_counter(redis, "end_user", end_user) == 1


def test_an_end_user_without_limits_creates_no_end_user_window(gateway: Gateway) -> None:
    """A never seen end user (no row, no default budget on the suite proxy) is served once.

    No `{end_user:<id>}:requests` key exists in Redis: the descriptor is only built when an rpm or tpm limit is set
    (parallel_request_limiter_v3.py:3364-3366), so unlimited end users cost no limiter state.
    """
    with wire_server(provider()) as wire, gateway.scenario() as scenario, shared_redis() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model])
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat(gateway, model, key, user=end_user))
        assert requests_counter(redis, "end_user", end_user) is None


def test_a_customer_tpd_limit_does_not_limit_chat_requests(gateway: Gateway) -> None:
    """A customer is created with `tpd_limit: 1` (one token per day) and sends two chat requests.

    Both are served with 200: `end_user_tpd_limit` is read only by the batch limiter
    (batch_rate_limiter.py:347), never by the v3 request limiter. UNVERIFIED: that no other hook charges chat tokens
    against `tpd_limit`; the BudgetNewRequest field description says "charged by batch submissions".
    """
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model])
        end_user: Final = customer(scenario, user_id=new_end_user(), tpd_limit=1)
        assert_ok(chat(gateway, model, key, user=end_user))
        assert_ok(chat(gateway, model, key, user=end_user))


def test_the_default_end_user_budget_rpm_applies_to_a_never_seen_end_user(defaulting: Gateway) -> None:
    """`max_end_user_budget_id` names a budget with `rpm_limit: 1`. A never seen end user (no row) sends two requests.

    The second is the 429 for `end_user: <id>`: when the end user has no row, the default budget's limits are still
    applied (user_api_key_auth.py:2883-2899 and :3567), unlike its `max_budget`, which needs a row.
    """
    _ensure_default_budget(defaulting)
    with wire_server(provider()) as wire, defaulting.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model])
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat(defaulting, model, key, user=end_user))
        assert_refused_by(chat(defaulting, model, key, user=end_user), "end_user", end_user, "requests", 1)


def test_the_default_end_user_rpm_is_a_window_per_end_user(defaulting: Gateway) -> None:
    """With the same default budget (`rpm_limit: 1`), two never seen end users each send one request on one key.

    Both are served: the default budget gives each end user its own window, it is not one shared pool.
    """
    _ensure_default_budget(defaulting)
    with wire_server(provider()) as wire, defaulting.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model])
        first: Final = new_end_user()
        second: Final = new_end_user()
        forget_end_user(scenario, first)
        forget_end_user(scenario, second)
        assert_ok(chat(defaulting, model, key, user=first))
        assert_ok(chat(defaulting, model, key, user=second))


def test_a_customers_own_rpm_wins_over_the_default_end_user_rpm(defaulting: Gateway) -> None:
    """With the default budget at `rpm_limit: 1`, a customer created with its own `rpm_limit: 5` sends two requests.

    Both are served: a row whose own budget resolved is returned unchanged (`_apply_default_budget_to_end_user`,
    auth_checks.py:1988-1989), so the default never tightens an explicit customer limit.
    """
    _ensure_default_budget(defaulting)
    with wire_server(provider()) as wire, defaulting.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model])
        end_user: Final = customer(scenario, user_id=new_end_user(), rpm_limit=5)
        assert_ok(chat(defaulting, model, key, user=end_user))
        assert_ok(chat(defaulting, model, key, user=end_user))
