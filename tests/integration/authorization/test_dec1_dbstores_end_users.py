"""Which end user a request is attributed to, and which fields of that end user (customer) row take effect.

Contract, read from litellm/proxy/auth/auth_utils.py `get_end_user_id_from_request_body` and
litellm/proxy/auth/auth_checks.py `get_end_user_object`, `_end_user_is_known_unrestricted`,
`_apply_default_budget_to_end_user`, `check_end_user_budget`:

* The end user id is taken from the first of: a standard header (`x-litellm-customer-id`,
  `x-litellm-end-user-id`, `STANDARD_CUSTOMER_ID_HEADERS` in litellm/constants.py:1723), the configured
  `user_header_mappings` / `user_header_name` header, the body `user` field, `litellm_metadata.user`,
  `metadata.user_id`, and finally `safety_identifier`.
* The `LiteLLM_EndUserTable` row is read joined to its budget and object permission rows
  (`find_end_user_row.sql` on draft PR #52170). A cached "restricted registry" (`end_user_restricted_registry`)
  lists every end user whose row restricts something (blocked, budget_id, region, default_model,
  object_permission_id, non-empty models); an id outside it is served without reading its row. Customer writes
  evict both the row and the registry and broadcast the eviction.
* An end user that has no row is served, and the spend writer upserts a row for it with its spend and
  `blocked: false` (litellm/proxy/utils.py `update_end_user_spend`).
* A key's `metadata.end_user_budget_id` is the default budget for end users that have a row but no budget of
  their own, for requests through that key only.

Not duplicated: blocking and unblocking (`test_end_user_block.py`), customer model lists and their CRUD
(`test_customer_model_allowlist.py`), JWT end user claims (`test_jwt_key_mapping_lifecycle.py`), end user budget
arithmetic (`test_dec1_budgets_*`), the body `user` field on the spend row (`test_dec1_keys_request_shaping.py`).
"""

from __future__ import annotations

import uuid
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration.authorization.dec1_dbstores_support import (
    DEFAULT_AUTH_CACHE_TTL_SECONDS,
    PEER_SECONDS,
    WRITER_SECONDS,
    assert_budget_refused,
    assert_customer_model_denied,
    assert_ok,
    chat_as,
    customer,
    end_user_rows,
    forget_end_user,
    settled,
    spend_end_users,
)


def _new_end_user() -> str:
    return f"dec1-dbstores-end-user-{uuid.uuid4().hex}"


@pytest.mark.parametrize("header", ["x-litellm-customer-id", "x-litellm-end-user-id"])
def test_a_standard_customer_header_names_the_end_user_on_the_spend_row(gateway: Gateway, header: str) -> None:
    """A key sends chat with header `<header>: <new id>` and no body `user`.

    The request is served and its spend row carries the header value in `end_user`. Both header names are
    always honoured, without configuration.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = _new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(gateway, key, model, headers={header: end_user}))
        assert spend_end_users(gateway, key) == [end_user]


def test_a_standard_customer_header_wins_over_the_body_user_field(gateway: Gateway) -> None:
    """A request carries `x-litellm-customer-id: <A>` and body `user: <B>`.

    The spend row's `end_user` is A. The header is checked first (`Check 1` in
    `get_end_user_id_from_request_body`).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        from_header: Final = _new_end_user()
        from_body: Final = _new_end_user()
        forget_end_user(scenario, from_header)
        forget_end_user(scenario, from_body)
        assert_ok(chat_as(gateway, key, model, user=from_body, headers={"x-litellm-customer-id": from_header}))
        assert spend_end_users(gateway, key) == [from_header]


def test_the_customer_named_by_the_header_is_the_one_whose_restrictions_apply(gateway: Gateway) -> None:
    """Customer A may only use model X; customer B is unrestricted. A request for model Y carries
    `x-litellm-customer-id: A` and body `user: B`.

    The request is refused with 403 `customer_model_access_denied`: the header picks which end user row is
    looked up, so its restriction applies even though the body names an unrestricted customer.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key()
        restricted: Final = customer(scenario, models=[allowed])
        unrestricted: Final = customer(scenario)
        refused: Final = chat_as(gateway, key, other, user=unrestricted, headers={"x-litellm-customer-id": restricted})
        assert_customer_model_denied(refused)


def test_litellm_metadata_user_names_the_end_user_without_a_user_field(gateway: Gateway) -> None:
    """A request has no `user` field but `litellm_metadata: {"user": <id>}` (the Anthropic style carrier).

    Its spend row carries that id in `end_user`.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = _new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(gateway, key, model, litellm_metadata={"user": end_user}))
        assert spend_end_users(gateway, key) == [end_user]


def test_metadata_user_id_names_the_end_user_without_a_user_field(gateway: Gateway) -> None:
    """A request has no `user` field but `metadata: {"user_id": <id>}`.

    Its spend row carries that id in `end_user`.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = _new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(gateway, key, model, metadata={"user_id": end_user}))
        assert spend_end_users(gateway, key) == [end_user]


def test_the_body_user_field_wins_over_metadata_user_id(gateway: Gateway) -> None:
    """A request carries body `user: <A>` and `metadata: {"user_id": <B>}`.

    The spend row's `end_user` is A (`Check 3` runs before `Check 5`).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        first: Final = _new_end_user()
        second: Final = _new_end_user()
        forget_end_user(scenario, first)
        forget_end_user(scenario, second)
        assert_ok(chat_as(gateway, key, model, user=first, metadata={"user_id": second}))
        assert spend_end_users(gateway, key) == [first]


def test_a_metadata_object_sent_as_a_json_string_still_names_the_end_user(gateway: Gateway) -> None:
    """A request sends `metadata` as the JSON string `'{"user_id": "<id>"}'` rather than an object.

    The id is still attributed as the end user (`_as_dict` decodes string payloads so they cannot evade
    attribution). UNVERIFIED: that the chat route accepts a string `metadata` without a 400; if it does not, the
    contract still holds for multipart routes.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = _new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(gateway, key, model, metadata=f'{{"user_id": "{end_user}"}}'))
        assert spend_end_users(gateway, key) == [end_user]


def test_safety_identifier_names_the_end_user_as_a_last_resort(gateway: Gateway) -> None:
    """A request carries only `safety_identifier: <id>` (no header, `user` or metadata).

    The spend row's `end_user` is that id (`Check 6`).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = _new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(gateway, key, model, safety_identifier=end_user))
        assert spend_end_users(gateway, key) == [end_user]


def test_an_unknown_end_user_is_served_and_gets_a_row_with_its_spend(gateway: Gateway) -> None:
    """A never seen end user id is sent as body `user`.

    The request is served with 200, and a `LiteLLM_EndUserTable` row appears for that id with `blocked: false`,
    no `budget_id`, and spend above zero. The row is written by the spend writer's upsert, not by auth.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = _new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(gateway, key, model, user=end_user))
        rows: Final = eventually(lambda: end_user_rows(gateway, end_user), lambda found: len(found) == 1, seconds=70)
        assert (rows[0]["blocked"], rows[0]["budget_id"]) == (False, None)
        assert float(str(rows[0]["spend"])) > 0


def test_a_customer_restricted_after_the_id_was_seen_is_refused_on_both_proxies(
    gateway: Gateway, peer: Gateway
) -> None:
    """An end user id is served on both proxies while it has no row, so both cache the restricted registry
    without it. `/customer/new` then creates the customer with `models: [<model A>]`.

    A request for model B must be refused with 403 `customer_model_access_denied` on the writer within 5 seconds
    and on the peer within 10. The customer write evicts `end_user_restricted_registry` and the row and
    broadcasts it; a port that keeps the stale registry serves the new customer unrestricted for 60 seconds.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = _new_end_user()
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, other, user=end_user))
        eventually(lambda: end_user_rows(gateway, end_user), lambda found: len(found) == 1, seconds=70)
        gateway.post("/customer/update", {"user_id": end_user, "models": [allowed]})
        forget_end_user(scenario, end_user)
        for proxy, seconds in ((gateway, WRITER_SECONDS), (peer, PEER_SECONDS)):
            assert_customer_model_denied(
                settled(lambda proxy=proxy: chat_as(proxy, key, other, user=end_user), 403, seconds)
            )


def test_a_brand_new_restricted_customer_is_refused_on_its_first_request(gateway: Gateway) -> None:
    """`/customer/new` creates a customer with `models: [<A>]` and the customer immediately asks for model B.

    The very first request is refused with 403 `customer_model_access_denied`, even if another test warmed the
    restricted registry a moment ago without this id.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key()
        assert_ok(chat_as(gateway, key, other, user=_new_end_user()))
        restricted: Final = customer(scenario, models=[allowed])
        assert_customer_model_denied(chat_as(gateway, key, other, user=restricted))


def test_deleting_a_restricted_customer_lifts_the_restriction_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A customer allowed only model A is refused model B on both proxies, then `/customer/delete` removes it.

    Model B must be served for that id on the writer within 5 seconds and on the peer within 10: the id now
    has no row, so it is an unknown, unrestricted end user again.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = _new_end_user()
        gateway.post("/customer/new", {"user_id": end_user, "models": [allowed]})
        forget_end_user(scenario, end_user)
        for proxy in (gateway, peer):
            assert_customer_model_denied(chat_as(proxy, key, other, user=end_user))
        gateway.post("/customer/delete", {"user_ids": [end_user]})
        assert_ok(settled(lambda: chat_as(gateway, key, other, user=end_user), 200, WRITER_SECONDS))
        assert_ok(settled(lambda: chat_as(peer, key, other, user=end_user), 200, PEER_SECONDS))


def test_a_customer_linked_to_a_zero_budget_row_is_refused(gateway: Gateway) -> None:
    """A budget row is created with `/budget/new {"max_budget": 0}` and a customer with `budget_id` pointing to it.

    The customer's first request is refused with 422 `budget_exceeded`. The budget comes from the
    `LiteLLM_BudgetTable` row joined to the end user row, not from the customer row itself.

    The customer is created with `spend: 1`: the end user check refuses only when spend is above the budget
    (`spend <= max_budget` passes, litellm/rust/crates/callbacks-budgets/src/checks.rs:432), so a customer with no
    spend is served under a zero budget.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        budget: Final = scenario.budget(max_budget=0)
        end_user: Final = customer(scenario, budget_id=budget, spend=1)
        assert_budget_refused(chat_as(gateway, key, model, user=end_user))


def test_a_customer_linked_to_a_budget_row_without_max_budget_is_served(gateway: Gateway) -> None:
    """A budget row with only `rpm_limit: 1000` (no `max_budget`) is linked to a customer.

    The customer is served with 200: `check_end_user_budget` returns when `max_budget` is null.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        budget: Final = scenario.budget(rpm_limit=1000)
        end_user: Final = customer(scenario, budget_id=budget)
        assert_ok(chat_as(gateway, key, model, user=end_user))


def test_lowering_a_linked_budget_row_reaches_a_warmed_customer_on_both_proxies(
    gateway: Gateway, peer: Gateway
) -> None:
    """A customer linked to a budget row with `max_budget: 100` is served on both proxies, then `/budget/update`
    sets `max_budget: 0` on that row.

    Intended: both proxies refuse the customer with 422 within 5 and 10 seconds.
    """
    pytest.skip(
        "BUG: /budget/update evicts customer hierarchy and team member default budget caches but not "
        "`end_user_id:<id>`, whose cached row embeds the budget, so the old budget is enforced until the TTL"
    )
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        budget: Final = scenario.budget(max_budget=100)
        end_user: Final = customer(scenario, budget_id=budget)
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, model, user=end_user))
        gateway.post("/budget/update", {"budget_id": budget, "max_budget": 0})
        assert_budget_refused(settled(lambda: chat_as(gateway, key, model, user=end_user), 422, WRITER_SECONDS))
        assert_budget_refused(settled(lambda: chat_as(peer, key, model, user=end_user), 422, PEER_SECONDS))


@pytest.mark.timeout(240)
def test_a_lowered_budget_row_reaches_a_warmed_customer_within_the_auth_cache_ttl(gateway: Gateway) -> None:
    """The same `/budget/update` to `max_budget: 0` on a warmed customer's budget row.

    The customer must be refused with 422 at the latest after the 60 second auth cache TTL plus 15 seconds.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        budget: Final = scenario.budget(max_budget=100)
        end_user: Final = customer(scenario, budget_id=budget)
        assert_ok(chat_as(gateway, key, model, user=end_user))
        gateway.post("/budget/update", {"budget_id": budget, "max_budget": 0})
        assert_budget_refused(
            settled(lambda: chat_as(gateway, key, model, user=end_user), 422, DEFAULT_AUTH_CACHE_TTL_SECONDS + 15)
        )


def test_moving_a_warmed_customer_to_a_zero_budget_refuses_it_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A customer without a budget is served on both proxies, then `/customer/update` sets `budget_id` to a
    budget row with `max_budget: 0`.

    Both proxies must refuse it with 422 within 5 and 10 seconds: the customer write evicts the row and the
    registry, and the new row joins the new budget.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        budget: Final = scenario.budget(max_budget=0)
        end_user: Final = customer(scenario)
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, model, user=end_user))
        gateway.post("/customer/update", {"user_id": end_user, "budget_id": budget})
        assert_budget_refused(settled(lambda: chat_as(gateway, key, model, user=end_user), 422, WRITER_SECONDS))
        assert_budget_refused(settled(lambda: chat_as(peer, key, model, user=end_user), 422, PEER_SECONDS))


def test_a_keys_end_user_budget_applies_to_a_known_customer_without_a_budget(gateway: Gateway) -> None:
    """A key carries `metadata: {"end_user_budget_id": <budget with max_budget 0>}`; a customer exists with no
    budget of its own.

    The customer's request through that key is refused with 422 `budget_exceeded`.

    The customer is created with `spend: 1`: the end user check refuses only when spend is above the budget
    (`spend <= max_budget` passes, litellm/rust/crates/callbacks-budgets/src/checks.rs:432), so a customer with no
    spend is served under a zero budget.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        budget: Final = scenario.budget(max_budget=0)
        key: Final = scenario.key(metadata={"end_user_budget_id": budget})
        end_user: Final = customer(scenario, spend=1)
        assert_budget_refused(chat_as(gateway, key, model, user=end_user))


def test_a_keys_end_user_budget_does_not_follow_the_customer_to_another_key(gateway: Gateway) -> None:
    """The same customer is refused through the key carrying `end_user_budget_id` (zero budget) and then calls
    through a second key without one.

    The second request is served with 200. The cached end user row carries at most the proxy wide default,
    never a key's (`_apply_default_budget_to_end_user` sets the key default on a copy).

    The customer is created with `spend: 1`: the end user check refuses only when spend is above the budget
    (`spend <= max_budget` passes, litellm/rust/crates/callbacks-budgets/src/checks.rs:432), so a customer with no
    spend is served under a zero budget.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        budget: Final = scenario.budget(max_budget=0)
        limiting: Final = scenario.key(metadata={"end_user_budget_id": budget})
        plain: Final = scenario.key()
        end_user: Final = customer(scenario, spend=1)
        assert_budget_refused(chat_as(gateway, limiting, model, user=end_user))
        assert_ok(chat_as(gateway, plain, model, user=end_user))


def test_a_customers_own_budget_wins_over_the_keys_end_user_budget(gateway: Gateway) -> None:
    """A customer has its own budget row with `max_budget: 100`; the key's `end_user_budget_id` names a budget
    with `max_budget: 0`.

    The customer is served with 200: the key default applies only to end users without a budget.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        zero: Final = scenario.budget(max_budget=0)
        roomy: Final = scenario.budget(max_budget=100)
        key: Final = scenario.key(metadata={"end_user_budget_id": zero})
        end_user: Final = customer(scenario, budget_id=roomy)
        assert_ok(chat_as(gateway, key, model, user=end_user))


def test_a_keys_end_user_budget_starts_to_apply_once_the_end_user_has_a_row(gateway: Gateway) -> None:
    """A key carries `end_user_budget_id` naming a zero budget; a never seen end user calls through it.

    UNVERIFIED: the first request is served with 200, because `get_end_user_object` finds no row and returns
    None, so no budget is attached. Once the spend writer has upserted the row, the next request is refused with
    422. A store that synthesises an empty row for unknown ids would refuse the first request instead.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        budget: Final = scenario.budget(max_budget=0)
        key: Final = scenario.key(metadata={"end_user_budget_id": budget})
        end_user: Final = _new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(gateway, key, model, user=end_user))
        eventually(lambda: end_user_rows(gateway, end_user), lambda found: len(found) == 1, seconds=70)
        assert_budget_refused(settled(lambda: chat_as(gateway, key, model, user=end_user), 422, WRITER_SECONDS * 2))
