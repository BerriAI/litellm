"""Which end user a request is attributed to when the caller names it in odd shapes or in several places.

Contract, read from litellm/proxy/auth/auth_utils.py:2357-2440 `get_end_user_id_from_request_body` and
:2322-2354 `_coerce_user_id_to_str`:

* The sources are tried in this order and the first usable value wins: the standard headers
  `x-litellm-customer-id` then `x-litellm-end-user-id` (litellm/constants.py:1723, in list order), the configured
  header, the body `user`, `litellm_metadata.user`, `metadata.user_id`, then `safety_identifier`.
* A value is usable when it is a non-empty string after stripping, or a number (rendered with `str()`). Booleans,
  objects, lists and blank strings are not usable, so the next source is tried instead of giving up.
* With `validate_end_user_id_in_db` off (the suite default) a string that looks like JSON is kept verbatim.
* The resolved id is what the spend row's `end_user` column and `metadata.user_api_key_end_user_id` carry
  (litellm/proxy/spend_tracking/spend_tracking_utils.py:759-794). The body sent upstream is not rewritten, so a
  padded `user` is forwarded padded.

Already pinned elsewhere and not repeated: each source alone and the basic precedence pairs
(`test_dec1_dbstores_end_users.py`, `spend/test_end_user_spend_without_proxy_user.py`), and malformed values with
`disable_end_user_cost_tracking` on (`spend/test_disable_end_user_cost_tracking.py`). A Rust port of the extraction
chain has to keep the strip, the number rendering and the fall through, or spend lands on a different customer.
"""

from __future__ import annotations

import json
import random
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, object_value
from tests.integration.authorization.dec1_dbstores_support import assert_ok, chat_as, forget_end_user
from tests.integration.authorization.dec1_endusers_support import (
    CUSTOMER_HEADER,
    END_USER_HEADER,
    new_end_user,
    spend_rows,
    text_marker,
    upstream_bodies,
)


def _raw_chat(gateway: Gateway, key: str, model: str, **fields: JsonValue) -> httpx.Response:
    """A chat request whose extra top-level fields are sent exactly as given, including non-string `user` values."""
    body: Final[dict[str, JsonValue]] = {"model": model, "messages": [{"role": "user", "content": text_marker()}]}
    return gateway.request("POST", "/v1/chat/completions", {**body, **fields}, key=key)


def _recorded_end_user(gateway: Gateway, key: str) -> JsonValue:
    rows: Final = spend_rows(gateway, key)
    assert len(rows) == 1, rows
    return rows[0]["end_user"]


def test_a_padded_body_user_is_recorded_stripped_and_forwarded_unchanged(gateway: Gateway) -> None:
    """A chat request carries `user: "  <id>  "` (two spaces on each side).

    The spend row's `end_user` must be `<id>` without the padding (`value.strip()` at auth_utils.py:2341), while the
    body the scripted upstream receives still has the padded `user`, because only the attribution is normalized.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        text: Final = text_marker()
        assert_ok(chat_as(gateway, key, model, user=f"  {end_user}  ", text=text))
        assert _recorded_end_user(gateway, key) == end_user
        assert [body.get("user") for body in upstream_bodies(gateway, text)] == [f"  {end_user}  "]


@pytest.mark.parametrize("kind", ["int", "float"])
def test_a_numeric_body_user_is_recorded_as_its_decimal_text(gateway: Gateway, kind: str) -> None:
    """A chat request carries a JSON number as `user` (an integer, or a float with a fractional part).

    It must be served with 200 and the spend row's `end_user` must be Python's `str()` of the number, for example
    `"812345678"` or `"812345678.25"` (`isinstance(value, (int, float))` branch at auth_utils.py:2338).
    UNVERIFIED: that no request model rejects a non-string `user` on /v1/chat/completions before auth; the flag-on
    test in spend/test_disable_end_user_cost_tracking.py sees 200 for an integer.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        whole: Final = random.randint(100_000_000, 999_999_999)
        value: Final[JsonValue] = whole if kind == "int" else whole + 0.25
        expected: Final = str(whole) if kind == "int" else f"{whole}.25"
        forget_end_user(scenario, expected)
        assert_ok(_raw_chat(gateway, key, model, user=value))
        assert _recorded_end_user(gateway, key) == expected


def test_a_boolean_body_user_is_ignored_and_metadata_user_id_names_the_end_user(gateway: Gateway) -> None:
    """A chat request carries `user: true` and `metadata: {"user_id": <id>}`.

    A boolean is not an id (auth_utils.py:2335 checks `bool` before `int`), so the chain falls through and the spend
    row's `end_user` is `<id>`, not `"True"`.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(_raw_chat(gateway, key, model, user=True, metadata={"user_id": end_user}))
        assert _recorded_end_user(gateway, key) == end_user


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_body_user_falls_through_to_metadata_user_id(gateway: Gateway, blank: str) -> None:
    """A chat request carries an empty or all-space `user` and `metadata: {"user_id": <id>}`.

    A blank string strips to nothing and is dropped (auth_utils.py:2341-2343), so the spend row's `end_user` is
    `<id>` from the metadata, not an empty attribution.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(gateway, key, model, user=blank, metadata={"user_id": end_user}))
        assert _recorded_end_user(gateway, key) == end_user


@pytest.mark.parametrize("shape", ["object", "list"])
def test_a_structured_body_user_falls_through_to_safety_identifier(gateway: Gateway, shape: str) -> None:
    """A chat request carries `user` as a JSON object or list, plus `safety_identifier: <id>`.

    Structured values are always dropped (auth_utils.py:2353), so the last source in the chain names the end user:
    the spend row's `end_user` is `<id>`. The object's text must never become an id.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        structured: Final[JsonValue] = {"device_id": "dec1-endusers"} if shape == "object" else ["dec1-endusers"]
        assert_ok(_raw_chat(gateway, key, model, user=structured, safety_identifier=end_user))
        assert _recorded_end_user(gateway, key) == end_user


def test_the_customer_id_header_wins_over_the_end_user_id_header(gateway: Gateway) -> None:
    """A chat request carries both standard headers: `x-litellm-customer-id: A` and `x-litellm-end-user-id: B`.

    `STANDARD_CUSTOMER_ID_HEADERS` is walked in list order (litellm/constants.py:1723, auth_utils.py:2313), so the
    spend row's `end_user` is A.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        first: Final = new_end_user()
        second: Final = new_end_user()
        forget_end_user(scenario, first)
        forget_end_user(scenario, second)
        assert_ok(chat_as(gateway, key, model, headers={END_USER_HEADER: second, CUSTOMER_HEADER: first}))
        assert _recorded_end_user(gateway, key) == first


def test_a_blank_customer_id_header_falls_through_to_the_end_user_id_header(gateway: Gateway) -> None:
    """A chat request carries `x-litellm-customer-id: ""` and `x-litellm-end-user-id: <id>`.

    The blank header is not usable (auth_utils.py:2316-2318 keep looking), so the spend row's `end_user` is `<id>`.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(gateway, key, model, headers={CUSTOMER_HEADER: "", END_USER_HEADER: end_user}))
        assert _recorded_end_user(gateway, key) == end_user


def test_a_blank_standard_header_falls_through_to_the_body_user(gateway: Gateway) -> None:
    """A chat request carries `x-litellm-end-user-id: ""` and body `user: <id>`.

    No standard header is usable, so the body `user` names the end user: the spend row's `end_user` is `<id>`.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(gateway, key, model, user=end_user, headers={END_USER_HEADER: ""}))
        assert _recorded_end_user(gateway, key) == end_user


def test_litellm_metadata_user_wins_over_metadata_user_id(gateway: Gateway) -> None:
    """A chat request has no `user` but carries `litellm_metadata: {"user": A}` and `metadata: {"user_id": B}`.

    Check 4 (`litellm_metadata.user`) runs before check 5 (`metadata.user_id`) at auth_utils.py:2415-2425, so the
    spend row's `end_user` is A.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        first: Final = new_end_user()
        second: Final = new_end_user()
        forget_end_user(scenario, first)
        forget_end_user(scenario, second)
        assert_ok(chat_as(gateway, key, model, litellm_metadata={"user": first}, metadata={"user_id": second}))
        assert _recorded_end_user(gateway, key) == first


def test_metadata_user_id_wins_over_safety_identifier(gateway: Gateway) -> None:
    """A chat request has no `user` but carries `metadata: {"user_id": A}` and `safety_identifier: B`.

    `safety_identifier` is the last resort (auth_utils.py:2427-2435), so the spend row's `end_user` is A.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        first: Final = new_end_user()
        second: Final = new_end_user()
        forget_end_user(scenario, first)
        forget_end_user(scenario, second)
        assert_ok(chat_as(gateway, key, model, metadata={"user_id": first}, safety_identifier=second))
        assert _recorded_end_user(gateway, key) == first


def test_a_blank_litellm_metadata_user_falls_through_to_metadata_user_id(gateway: Gateway) -> None:
    """A chat request carries `litellm_metadata: {"user": "  "}` and `metadata: {"user_id": <id>}`.

    The blank value is dropped like a blank body `user`, so the spend row's `end_user` is `<id>`.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(gateway, key, model, litellm_metadata={"user": "  "}, metadata={"user_id": end_user}))
        assert _recorded_end_user(gateway, key) == end_user


def test_a_numeric_metadata_user_id_is_recorded_as_text(gateway: Gateway) -> None:
    """A chat request carries `metadata: {"user_id": <integer>}` and nothing else that names an end user.

    The integer is rendered with `str()` (auth_utils.py:2338), so the spend row's `end_user` is its decimal text.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        whole: Final = random.randint(100_000_000, 999_999_999)
        forget_end_user(scenario, str(whole))
        assert_ok(chat_as(gateway, key, model, metadata={"user_id": whole}))
        assert _recorded_end_user(gateway, key) == str(whole)


def test_a_json_looking_body_user_is_kept_verbatim_without_validation(gateway: Gateway) -> None:
    """On the suite proxy (`validate_end_user_id_in_db` off) a chat request carries `user: '{"id": "<marker>"}'`.

    JSON-looking strings are only rejected when validation is on (auth_utils.py:2347-2351), so the spend row's
    `end_user` is exactly that string. A port that always rejects them would silently drop attribution for
    deployments that pass encoded ids.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        encoded: Final = json.dumps({"id": new_end_user()})
        forget_end_user(scenario, encoded)
        assert_ok(chat_as(gateway, key, model, user=encoded))
        assert _recorded_end_user(gateway, key) == encoded


def test_a_metadata_json_string_that_decodes_to_a_list_names_no_end_user(gateway: Gateway) -> None:
    """A chat request carries `metadata` as the JSON string `'["<id>"]'` and nothing else that names an end user.

    `_as_dict` only accepts a string that decodes to an object (auth_utils.py:2403-2412), so no end user is found:
    the request is served and its spend row's `end_user` is the empty string (spend_tracking_utils.py:1039).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        assert_ok(chat_as(gateway, key, model, metadata=json.dumps([new_end_user()])))
        assert _recorded_end_user(gateway, key) == ""


def test_the_spend_row_metadata_carries_the_resolved_end_user(gateway: Gateway) -> None:
    """A chat request names its end user through `x-litellm-end-user-id` only.

    The spend row's `end_user` column is the end user.

    Pinned current behaviour: the column is filled from the logging payload's `user_api_key_end_user_id`
    (spend_tracking_utils.py:793-794), but the stored `metadata` is a `SpendLogsMetadata`
    (spend_tracking_utils.py:347), which declares no end user field, so the row's metadata does not carry it.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = new_end_user()
        forget_end_user(scenario, end_user)
        assert_ok(chat_as(gateway, key, model, headers={END_USER_HEADER: end_user}))
        row: Final = spend_rows(gateway, key)[0]
        assert row["end_user"] == end_user, row
        assert "user_api_key_end_user_id" not in object_value(row["metadata"]), row
