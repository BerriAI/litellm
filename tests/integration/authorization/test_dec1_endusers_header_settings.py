"""Proxy settings that name the end user from a header, or rewrite the `user` the provider sees.

Contract, read from litellm/proxy/auth/auth_utils.py:2273-2292 `get_customer_user_header_from_mapping`,
:2357-2398 `get_end_user_id_from_request_body` and litellm/proxy/litellm_pre_call_utils.py:1930-1968 and
:3034-3051 (`add_internal_user_from_user_mapping`, `get_user_from_headers`, `overwrite_user_with_key_hash`):

* `general_settings.user_header_mappings` maps headers to roles. Every mapping with role `customer` is an end user
  header, tried in the order listed, after the two standard customer headers and before the body `user`. When at
  least one `customer` mapping exists, auth ignores the deprecated `user_header_name`.
* A mapping with role `internal_user` replaces the request's `user_id` (the spend row's `user` column) with the
  header value. It never names the end user.
* After auth, the request setup reads `user_header_name` on its own, whatever the mappings say: when auth found no
  end user it adopts the header value as the end user, and when the body has no `user` it forwards the header value
  to the provider as `user`. So the provider can see a different `user` than the one spend is attributed to.
* `litellm_settings.overwrite_user_with_key_hash: true` replaces the forwarded `user` with the key's SHA-256 (or the
  `litellm_proxy_master_key` alias), after the end user was resolved, so attribution is unchanged.
* `litellm_settings.max_end_user_budget` (the float, not `max_end_user_budget_id`) is read in exactly one place,
  litellm/proxy/utils.py:9103, where its branch is `pass`: it limits nothing.

Not repeated: one configured header naming and budget checking an end user (`test_dec1_dbstores_end_user_settings.py`,
`spend/test_end_user_spend_without_proxy_user.py`), `user_header_name` filling and not replacing the forwarded
`user` (`configuration/test_dec1_config_general_settings_matrix.py`). Three module scoped proxies share the suite
database and upstream.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest

from tests.integration._support.client import Gateway
from tests.integration.authorization.dec1_dbstores_support import assert_ok, chat_as, customer, forget_end_user
from tests.integration.authorization.dec1_endusers_support import (
    CUSTOMER_HEADER,
    MASTER_KEY_ALIAS,
    end_user_row,
    landed_end_user_row,
    new_end_user,
    owned_endusers_proxy,
    spend_rows,
    text_marker,
    upstream_bodies,
)
from tests.integration.authorization.dec1_keys_support import digest

pytestmark: Final = pytest.mark.timeout(300)

_FIRST: Final = "x-dec1-endusers-first"
_SECOND: Final = "x-dec1-endusers-second"
_INTERNAL: Final = "x-dec1-endusers-internal"
_LEGACY: Final = "x-dec1-endusers-legacy"


@pytest.fixture(scope="module")
def mapped(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    """Two `customer` mappings, one `internal_user` mapping, the deprecated header, and the dead float budget."""
    yield from owned_endusers_proxy(
        tmp_path_factory.mktemp("dec1-endusers-mapped"),
        general_settings={
            "user_header_mappings": [
                {"header_name": _INTERNAL, "litellm_user_role": "internal_user"},
                {"header_name": _FIRST, "litellm_user_role": "customer"},
                {"header_name": _SECOND, "litellm_user_role": "customer"},
            ],
            "user_header_name": _LEGACY,
        },
        litellm_settings={"max_end_user_budget": 0.000001},
    )


@pytest.fixture(scope="module")
def legacy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    """Only an `internal_user` mapping, so auth falls back to the deprecated `user_header_name`."""
    yield from owned_endusers_proxy(
        tmp_path_factory.mktemp("dec1-endusers-legacy"),
        general_settings={
            "user_header_mappings": [{"header_name": _INTERNAL, "litellm_user_role": "internal_user"}],
            "user_header_name": _LEGACY,
        },
    )


@pytest.fixture(scope="module")
def hashed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    yield from owned_endusers_proxy(
        tmp_path_factory.mktemp("dec1-endusers-hashed"), litellm_settings={"overwrite_user_with_key_hash": True}
    )


def _only_end_user(gateway: Gateway, key: str) -> object:
    rows: Final = spend_rows(gateway, key)
    assert len(rows) == 1, rows
    return rows[0]["end_user"]


def test_the_first_listed_customer_mapping_wins_when_both_mapped_headers_are_sent(mapped: Gateway) -> None:
    """Mappings list `x-dec1-endusers-first` before `x-dec1-endusers-second`, both with role `customer`. A request
    carries the second header with B and the first with A.

    The mapped headers are tried in configuration order (auth_utils.py:2388-2393), so the spend row's `end_user`
    is A whatever order the client sent them in.
    """
    with mapped.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        first: Final = new_end_user()
        second: Final = new_end_user()
        forget_end_user(scenario, first)
        forget_end_user(scenario, second)
        assert_ok(chat_as(mapped, key, model, headers={_SECOND: second, _FIRST: first}))
        assert _only_end_user(mapped, key) == first


def test_the_second_customer_mapping_is_used_when_the_first_header_is_absent(mapped: Gateway) -> None:
    """A request carries only `x-dec1-endusers-second: <id>`.

    Every `customer` mapping counts, not just the first (auth_utils.py:2286-2290 collects all of them), so the spend
    row's `end_user` is `<id>`.
    """
    with mapped.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(mapped, key, model, headers={_SECOND: end_user}))
        assert _only_end_user(mapped, key) == end_user


def test_a_blank_first_mapped_header_falls_through_to_the_second(mapped: Gateway) -> None:
    """A request carries `x-dec1-endusers-first: ""` and `x-dec1-endusers-second: <id>`.

    A blank value is not usable (`if user_id_str` at auth_utils.py:2392), so the next mapping names the end user.
    """
    with mapped.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(mapped, key, model, headers={_FIRST: "", _SECOND: end_user}))
        assert _only_end_user(mapped, key) == end_user


def test_a_mapped_customer_header_wins_over_the_body_user_but_the_body_user_is_forwarded(mapped: Gateway) -> None:
    """A request carries `x-dec1-endusers-first: A` and body `user: B`.

    The spend row's `end_user` is A (the mapping is check 2, the body is check 3), while the provider receives
    `user: B` unchanged: a `customer` mapping never writes the forwarded `user`.
    """
    with mapped.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        first: Final = new_end_user()
        second: Final = new_end_user()
        forget_end_user(scenario, first)
        forget_end_user(scenario, second)
        text: Final = text_marker()
        assert_ok(chat_as(mapped, key, model, user=second, headers={_FIRST: first}, text=text))
        assert _only_end_user(mapped, key) == first
        assert [body.get("user") for body in upstream_bodies(mapped, text)] == [second]


def test_a_standard_customer_header_wins_over_a_mapped_customer_header(mapped: Gateway) -> None:
    """A request carries `x-litellm-customer-id: A` and `x-dec1-endusers-first: B`.

    The standard headers are check 1 and need no configuration (auth_utils.py:2365-2367), so the spend row's
    `end_user` is A.
    """
    with mapped.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        first: Final = new_end_user()
        second: Final = new_end_user()
        forget_end_user(scenario, first)
        forget_end_user(scenario, second)
        assert_ok(chat_as(mapped, key, model, headers={CUSTOMER_HEADER: first, _FIRST: second}))
        assert _only_end_user(mapped, key) == first


def test_with_customer_mappings_the_legacy_header_is_still_billed_and_forwarded_after_auth(mapped: Gateway) -> None:
    """With `customer` mappings configured, a request carries only the deprecated `x-dec1-endusers-legacy: <id>`.

    Auth ignores the legacy header (the mapping takes precedence, auth_utils.py:2375-2385), but the request setup
    reads `user_header_name` afterwards and adopts it (litellm_pre_call_utils.py:3042-3047): the spend row's
    `end_user` is `<id>` and the provider receives `user: <id>`. Pinned because a port that only follows the auth
    side would drop this attribution.
    """
    with mapped.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        text: Final = text_marker()
        assert_ok(chat_as(mapped, key, model, headers={_LEGACY: end_user}, text=text))
        assert _only_end_user(mapped, key) == end_user
        assert [body.get("user") for body in upstream_bodies(mapped, text)] == [end_user]


def test_with_customer_mappings_an_over_budget_customer_named_by_the_legacy_header_is_refused(mapped: Gateway) -> None:
    """With `customer` mappings configured, a customer created with `spend: 0.5` and `max_budget: 0.1` is named only
    by the deprecated `x-dec1-endusers-legacy` header.

    Intended: the request is refused with 422 `budget_exceeded`, as it is when the same header is the only
    configured one, because the request is billed to that customer (see the test above).
    """
    pytest.skip(
        "BUG: with customer user_header_mappings, auth ignores user_header_name but the request setup still adopts "
        "it as the end user, so the customer is billed without its budget being checked"
    )
    with mapped.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = customer(scenario, user_id=new_end_user(), spend=0.5, max_budget=0.1)
        response: Final = chat_as(mapped, key, model, headers={_LEGACY: end_user})
        assert response.status_code == 422, response.text


def test_an_internal_user_mapping_sets_the_spend_rows_user_and_not_the_end_user(mapped: Gateway) -> None:
    """A key without a user sends a request carrying `x-dec1-endusers-internal: <internal id>` and body `user: <id>`.

    The `internal_user` mapping replaces the request's `user_id` (litellm_pre_call_utils.py:1932-1949) before the
    spend metadata is stamped (:3201), so the spend row's `user` column is `<internal id>` while `end_user` stays the
    body `user`.
    """
    with mapped.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        internal: Final = f"dec1-endusers-internal-{new_end_user()}"
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(mapped, key, model, user=end_user, headers={_INTERNAL: internal}))
        row: Final = spend_rows(mapped, key)[0]
        assert (row["user"], row["end_user"]) == (internal, end_user), row


def test_an_internal_user_mapping_header_alone_names_no_end_user(mapped: Gateway) -> None:
    """A request carries only `x-dec1-endusers-internal: <id>` (no customer header, no body `user`).

    An `internal_user` mapping is never an end user source (auth_utils.py:2286 keeps only `customer` roles), so the
    spend row's `end_user` is "" and no `LiteLLM_EndUserTable` row is created for `<id>`.
    """
    with mapped.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        internal: Final = new_end_user()
        forget_end_user(scenario, internal)
        assert_ok(chat_as(mapped, key, model, headers={_INTERNAL: internal}))
        assert _only_end_user(mapped, key) == ""
        assert end_user_row(mapped, internal) is None


def test_the_float_max_end_user_budget_setting_limits_nothing(mapped: Gateway) -> None:
    """The mapped proxy sets `litellm_settings.max_end_user_budget: 0.000001`. A never seen end user sends two
    requests, the second after its first spend has landed on its row.

    Both are served with 200 and the auto-created row has no `budget_id`: the only reader of the setting is a `pass`
    branch (litellm/proxy/utils.py:9103-9104). Only `max_end_user_budget_id` gives end users a default budget.
    """
    with mapped.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(mapped, key, model, user=end_user))
        row: Final = landed_end_user_row(mapped, end_user)
        assert row["budget_id"] is None, row
        assert_ok(chat_as(mapped, key, model, user=end_user))


def test_without_customer_mappings_the_legacy_header_names_the_end_user(legacy: Gateway) -> None:
    """The proxy has only an `internal_user` mapping plus `user_header_name: x-dec1-endusers-legacy`. A request
    carries the legacy header with A and body `user: B`.

    With no `customer` mapping, auth falls back to `user_header_name` (auth_utils.py:2381-2385), which is checked
    before the body: the spend row's `end_user` is A.
    """
    with legacy.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        first: Final = new_end_user()
        second: Final = new_end_user()
        forget_end_user(scenario, first)
        forget_end_user(scenario, second)
        assert_ok(chat_as(legacy, key, model, user=second, headers={_LEGACY: first}))
        assert _only_end_user(legacy, key) == first


def test_the_provider_gets_the_legacy_header_user_while_spend_goes_to_the_standard_header(legacy: Gateway) -> None:
    """A request has no body `user` and carries `x-litellm-customer-id: A` and `x-dec1-endusers-legacy: B`.

    Auth attributes the request to A (standard header first), and the request setup still copies the legacy header
    into the forwarded body because `"user" not in data` (litellm_pre_call_utils.py:3046-3047): the spend row's
    `end_user` is A while the provider receives `user: B`.
    """
    with legacy.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        first: Final = new_end_user()
        second: Final = new_end_user()
        forget_end_user(scenario, first)
        forget_end_user(scenario, second)
        text: Final = text_marker()
        assert_ok(chat_as(legacy, key, model, headers={CUSTOMER_HEADER: first, _LEGACY: second}, text=text))
        assert _only_end_user(legacy, key) == first
        assert [body.get("user") for body in upstream_bodies(legacy, text)] == [second]


def test_an_empty_legacy_header_is_forwarded_as_an_empty_user(legacy: Gateway) -> None:
    """A request has no body `user` and carries `x-dec1-endusers-legacy: ""`.

    Auth finds no usable end user, but `get_user_from_headers` only checks for `None`
    (litellm_pre_call_utils.py:1962-1966, :3043), so the provider receives `user: ""` and the spend row's `end_user`
    is "". UNVERIFIED: that the HTTP server keeps an empty header value rather than dropping the header.
    """
    with legacy.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        text: Final = text_marker()
        assert_ok(chat_as(legacy, key, model, headers={_LEGACY: ""}, text=text))
        assert [body.get("user") for body in upstream_bodies(legacy, text)] == [""]
        assert _only_end_user(legacy, key) == ""


def test_overwrite_user_with_key_hash_forwards_the_key_hash_but_bills_the_body_user(hashed: Gateway) -> None:
    """With `overwrite_user_with_key_hash: true`, a virtual key sends body `user: <id>`.

    The provider receives `user` equal to the SHA-256 hex of the key (`_stampable_key_hash`,
    litellm_pre_call_utils.py:371-382, :3049-3052), while the spend row's `end_user` stays `<id>` because the end user
    was resolved at auth before the overwrite.
    """
    with hashed.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        text: Final = text_marker()
        assert_ok(chat_as(hashed, key, model, user=end_user, text=text))
        assert [body.get("user") for body in upstream_bodies(hashed, text)] == [digest(key)]
        assert _only_end_user(hashed, key) == end_user


def test_overwrite_user_with_key_hash_stamps_a_user_even_when_none_was_sent(hashed: Gateway) -> None:
    """With `overwrite_user_with_key_hash: true`, a virtual key sends a request naming no end user at all.

    The provider still receives `user` equal to the key's SHA-256 hex, and the spend row's `end_user` is "": the
    stamp is for the provider only and never creates an end user.
    """
    with hashed.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        text: Final = text_marker()
        assert_ok(chat_as(hashed, key, model, text=text))
        assert [body.get("user") for body in upstream_bodies(hashed, text)] == [digest(key)]
        assert _only_end_user(hashed, key) == ""
        assert end_user_row(hashed, digest(key)) is None


def test_overwrite_user_with_key_hash_stamps_the_alias_for_the_master_key(hashed: Gateway) -> None:
    """With `overwrite_user_with_key_hash: true`, the master key sends a request with body `user: <id>`.

    The master key is never forwarded or hashed: the provider receives `user: "litellm_proxy_master_key"`
    (litellm/constants.py:1820, litellm_pre_call_utils.py:380).
    """
    with hashed.scenario() as scenario:
        model: Final = scenario.model()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        text: Final = text_marker()
        assert_ok(chat_as(hashed, hashed.key, model, user=end_user, text=text))
        assert [body.get("user") for body in upstream_bodies(hashed, text)] == [MASTER_KEY_ALIAS]
