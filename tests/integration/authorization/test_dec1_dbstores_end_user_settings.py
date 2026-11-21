"""Proxy settings that change how the end user lookup behaves: id validation, the proxy wide default end user
budget, and a configured end user header.

Contract, read from litellm/proxy/auth/auth_checks.py `resolve_and_validate_end_user_id`,
`_end_user_id_exists_in_db`, `_end_user_is_known_unrestricted`, `_apply_default_budget_to_end_user` and
litellm/proxy/auth/auth_utils.py `get_end_user_id_from_request_body`:

* `litellm_settings.validate_end_user_id_in_db: true` keeps a caller supplied end user id only when it matches a
  `LiteLLM_EndUserTable.user_id`, a `LiteLLM_UserTable.user_id`, or (case insensitively) a `user_email`. An
  unmatched id is dropped, so the request is served without an end user and the spend row's `end_user` is "".
  Answers are cached under `end_user_validation:<id>`: "valid" for 300 seconds, "invalid" for 60 seconds, and no
  customer write evicts them.
* `litellm_settings.max_end_user_budget_id: <budget id>` is the budget of every end user whose row has no
  budget. The budget row is read by id at request time, so it may be created after boot.
* `general_settings.user_header_name: <header>` names a header that carries the end user id, matched case
  insensitively, after the two standard customer headers.

Each setting gets its own proxy (module scoped, booted through `owned_proxy` on the suite database), because
validation and a default budget interact (`has_default_budget` keeps unmatched ids).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually, gateway_from_environment
from tests.integration._support.process import owned_proxy
from tests.integration.authorization.dec1_dbstores_support import (
    WRITER_SECONDS,
    assert_budget_refused,
    assert_ok,
    chat_as,
    config_with_settings,
    customer,
    end_user_rows,
    forget_end_user,
    settled,
    spend_end_users,
)

pytestmark: Final = pytest.mark.timeout(300)

_DEFAULT_BUDGET_ID: Final = f"dec1-dbstores-default-end-user-budget-{uuid.uuid4().hex[:8]}"
_CONFIGURED_HEADER: Final = "x-dec1-dbstores-customer"
_NEGATIVE_VALIDATION_SECONDS: Final = 60


def _proxy(
    directory: Path,
    *,
    general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
) -> Iterator[Gateway]:
    config: Final = config_with_settings(
        directory, general_settings=general_settings, litellm_settings=litellm_settings
    )
    with gateway_from_environment() as shared, owned_proxy(shared, directory / "proxy", {}, config=config) as proxy:
        yield proxy


@pytest.fixture(scope="module")
def validating(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    yield from _proxy(
        tmp_path_factory.mktemp("dec1-dbstores-validating"),
        litellm_settings={"validate_end_user_id_in_db": True},
    )


@pytest.fixture(scope="module")
def defaulting(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    yield from _proxy(
        tmp_path_factory.mktemp("dec1-dbstores-defaulting"),
        litellm_settings={"max_end_user_budget_id": _DEFAULT_BUDGET_ID},
    )


@pytest.fixture(scope="module")
def header_configured(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    yield from _proxy(
        tmp_path_factory.mktemp("dec1-dbstores-header"),
        general_settings={"user_header_name": _CONFIGURED_HEADER},
    )


def _new_end_user() -> str:
    return f"dec1-dbstores-end-user-{uuid.uuid4().hex}"


def test_with_validation_an_unknown_end_user_id_is_dropped_from_the_spend_row(validating: Gateway) -> None:
    """With `validate_end_user_id_in_db`, a request names a body `user` that matches no customer, user or email.

    The request is served with 200 and its spend row has `end_user == ""`.
    """
    with validating.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        assert_ok(chat_as(validating, key, model, user=_new_end_user()))
        assert spend_end_users(validating, key) == [""]


def test_with_validation_an_unknown_end_user_gets_no_end_user_row(validating: Gateway) -> None:
    """The same unmatched id is served and, after its spend row landed, still has no `LiteLLM_EndUserTable` row.

    Without validation the spend writer would upsert one; with validation the id never reaches spend tracking.
    """
    with validating.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = _new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(validating, key, model, user=end_user))
        spend_end_users(validating, key)
        assert end_user_rows(validating, end_user) == []


def test_with_validation_a_known_customer_id_is_kept(validating: Gateway) -> None:
    """A customer created through `/customer/new` is named as body `user`.

    Its spend row carries the customer id in `end_user`.
    """
    with validating.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = customer(scenario)
        assert_ok(chat_as(validating, key, model, user=end_user))
        assert spend_end_users(validating, key) == [end_user]


def test_with_validation_an_internal_user_id_is_accepted_as_an_end_user(validating: Gateway) -> None:
    """An internal user (a `LiteLLM_UserTable` row, no customer row) is named as body `user`.

    The id is kept: the spend row's `end_user` is the user id. The user store is consulted as a second source.
    """
    with validating.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        user: Final = scenario.user()
        forget_end_user(scenario, user)
        assert_ok(chat_as(validating, key, model, user=user))
        assert spend_end_users(validating, key) == [user]


def test_with_validation_an_internal_user_email_matches_case_insensitively(validating: Gateway) -> None:
    """An internal user has `user_email: Dec1.<hex>@Example.com`; the request names it in lower case as `user`.

    The id is kept as sent (lower case) on the spend row. Email lookup is case insensitive
    (`get_user_object(..., user_email=...)` through `_get_fuzzy_user_object`).
    """
    with validating.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        email: Final = f"Dec1.{uuid.uuid4().hex[:10]}@Example.com"
        scenario.user(user_email=email)
        forget_end_user(scenario, email.lower())
        assert_ok(chat_as(validating, key, model, user=email.lower()))
        assert spend_end_users(validating, key) == [email.lower()]


def test_with_validation_a_rejected_id_stays_dropped_until_its_negative_answer_expires(validating: Gateway) -> None:
    """An unmatched id is sent once (cached "invalid" for 60 s), then `/customer/new` creates a customer with it.

    A request made at once is still unattributed (spend row `end_user == ""`), because no customer write evicts
    `end_user_validation:<id>`. A request made after the 60 second negative TTL is attributed to the id.
    Pinned current behaviour; a store that drops the negative cache would attribute the second request too,
    which is harmless, but one that caches "invalid" for longer would lose attribution.
    """
    with validating.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = _new_end_user()
        assert_ok(chat_as(validating, key, model, user=end_user))
        customer(scenario, user_id=end_user)
        assert_ok(chat_as(validating, key, model, user=end_user))
        assert spend_end_users(validating, key, at_least=2) == ["", ""]
        eventually(
            lambda: (chat_as(validating, key, model, user=end_user), spend_end_users(validating, key))[1][-1],
            lambda last: last == end_user,
            seconds=_NEGATIVE_VALIDATION_SECONDS + 75,
        )


def test_the_default_end_user_budget_refuses_a_known_customer_without_a_budget(defaulting: Gateway) -> None:
    """`max_end_user_budget_id` names a budget row created after boot with `max_budget: 0`; a customer exists
    without a budget.

    The customer's request is refused with 422 `budget_exceeded`.

    The customer is created with `spend: 1`: the end user check refuses only when spend is above the budget
    (`spend <= max_budget` passes, litellm/rust/crates/callbacks-budgets/src/checks.rs:432), so a customer with no
    spend is served under a zero budget.
    """
    with defaulting.scenario() as scenario:
        _ensure_default_budget(defaulting)
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = customer(scenario, spend=1)
        assert_budget_refused(chat_as(defaulting, key, model, user=end_user))


def test_a_customers_own_budget_wins_over_the_default_end_user_budget(defaulting: Gateway) -> None:
    """A customer linked to its own budget row with `max_budget: 100` calls the proxy whose default end user
    budget is zero.

    The customer is served with 200.
    """
    with defaulting.scenario() as scenario:
        _ensure_default_budget(defaulting)
        model: Final = scenario.model()
        key: Final = scenario.key()
        own: Final = scenario.budget(max_budget=100)
        end_user: Final = customer(scenario, budget_id=own)
        assert_ok(chat_as(defaulting, key, model, user=end_user))


def test_the_default_end_user_budget_applies_once_an_unknown_end_user_has_a_row(defaulting: Gateway) -> None:
    """A never seen id calls the proxy whose default end user budget is zero.

    UNVERIFIED: the first request is served with 200 because there is no row to attach the default to; once the
    spend writer created the row, a later request is refused with 422.
    """
    with defaulting.scenario() as scenario:
        _ensure_default_budget(defaulting)
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = _new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(defaulting, key, model, user=end_user))
        eventually(lambda: end_user_rows(defaulting, end_user), lambda found: len(found) == 1, seconds=70)
        assert_budget_refused(settled(lambda: chat_as(defaulting, key, model, user=end_user), 422, WRITER_SECONDS * 2))


def _ensure_default_budget(proxy: Gateway) -> None:
    if not proxy.rows('SELECT budget_id FROM "LiteLLM_BudgetTable" WHERE budget_id=%s', (_DEFAULT_BUDGET_ID,)):
        proxy.post("/budget/new", {"budget_id": _DEFAULT_BUDGET_ID, "max_budget": 0})


@pytest.mark.parametrize("spelling", [_CONFIGURED_HEADER, _CONFIGURED_HEADER.upper()])
def test_a_configured_user_header_names_the_end_user(header_configured: Gateway, spelling: str) -> None:
    """With `user_header_name: x-dec1-dbstores-customer`, a request carries that header (in either case) and no
    body `user`.

    The spend row's `end_user` is the header value.
    """
    with header_configured.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = _new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(header_configured, key, model, headers={spelling: end_user}))
        assert spend_end_users(header_configured, key) == [end_user]


def test_a_configured_user_header_wins_over_the_body_user_field(header_configured: Gateway) -> None:
    """A request carries the configured header with A and body `user: B`.

    The spend row's `end_user` is A (`Check 2` runs before `Check 3`).
    """
    with header_configured.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        first: Final = _new_end_user()
        second: Final = _new_end_user()
        forget_end_user(scenario, first)
        forget_end_user(scenario, second)
        assert_ok(chat_as(header_configured, key, model, user=second, headers={_CONFIGURED_HEADER: first}))
        assert spend_end_users(header_configured, key) == [first]


def test_the_standard_customer_header_wins_over_the_configured_user_header(header_configured: Gateway) -> None:
    """A request carries `x-litellm-customer-id: A` and the configured header with B.

    The spend row's `end_user` is A: the standard headers are always checked first.
    """
    with header_configured.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        first: Final = _new_end_user()
        second: Final = _new_end_user()
        forget_end_user(scenario, first)
        forget_end_user(scenario, second)
        assert_ok(
            chat_as(header_configured, key, model, headers={"x-litellm-customer-id": first, _CONFIGURED_HEADER: second})
        )
        assert spend_end_users(header_configured, key) == [first]
