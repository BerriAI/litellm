"""What each state of a virtual key answers on an LLM route: valid, hashed, malformed, blocked, expired, master.

Contract (litellm/proxy/auth/user_api_key_auth.py, `user_api_key_auth_builder` and
`validate_resolved_virtual_key`; litellm/proxy/auth/auth_exception_handler.py):

* Only the raw `sk-` key authenticates. The database stores its SHA-256, and sending that hash (or any value
  that does not start with `sk-`) is a 401 `auth_error` "LiteLLM Virtual Key expected" whose message shows
  only the first and last four characters of what was received.
* A blocked key is a 401 `auth_error` on every route, checked before expiry and before model access.
* An expired key is a 401 with type `expired_key`. Expiry is evaluated on every request, cached key or not.
* The master key is a proxy admin on every route. Three routes are master key only and refuse every other
  credential before the key is even looked up.

Every denial uses the OpenAI error envelope `{"error": {"message", "type", "param", "code"}}` with `code`
as a string equal to the HTTP status. The Rust `gateway-auth` crate currently answers a plain text body
("missing or invalid bearer token"), so these shapes are the gap to close.
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually, string_value
from tests.integration.authorization.dec1_keys_support import (
    BLOCKED_KEY_MESSAGE,
    UNKNOWN_KEY_MESSAGE,
    assert_denied,
    assert_served,
    chat,
    digest,
    key_row,
    marker,
    upstream_requests,
)

_JWT_HINT: Final = (
    " This key has the structure of a JWT, but JWT auth is not enabled on this proxy, so it was treated as a"
    " virtual key. Set `enable_jwt_auth: true` under `general_settings` in your proxy config to authenticate"
    " with JWTs."
)


def _expired(gateway: Gateway, key: str) -> bool:
    return (
        gateway.rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE token=%s AND expires < now()', (digest(key),))
        != []
    )


def test_a_generated_key_is_stored_only_as_its_sha256(gateway: Gateway) -> None:
    """A key is generated and its database row is read.

    The `token` column must equal the SHA-256 hex of the raw key, and `key_name` must be `sk-...` followed
    by the last four characters. The raw key must not appear in the row, since auth looks keys up by hash.
    """
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        row: Final = key_row(gateway, key)
        assert (row["token"], row["key_name"]) == (digest(key), f"sk-...{key[-4:]}"), row
        assert key not in str(row), row


def test_an_admin_chosen_key_value_authenticates(gateway: Gateway) -> None:
    """An admin creates a key with an explicit `key` value that starts with `sk-`.

    That exact value must authenticate a chat completion with 200, and its row must be stored under the
    SHA-256 of the chosen value. Migrated deployments carry many such imported keys.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        chosen: Final = f"sk-{marker()}"
        assert scenario.key(key=chosen, models=[model]) == chosen
        assert_served(chat(gateway, chosen, model))
        assert key_row(gateway, chosen)["token"] == digest(chosen)


def test_the_hash_of_a_valid_key_is_not_accepted_as_a_credential(gateway: Gateway) -> None:
    """The caller sends the SHA-256 hex of a valid key as the bearer token.

    The proxy must answer 401 with exactly `{"error": {"message": "LiteLLM Virtual Key expected.
    Received=<first4>****<last4>, expected to start with 'sk-'.", "type": "auth_error", "param": null,
    "code": "401"}}` and must not call the upstream. Accepting the hash would turn a database read into a
    credential leak.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        hashed: Final = digest(key)
        text: Final = marker()
        refused: Final = chat(gateway, hashed, model, text)
        assert refused.status_code == 401, refused.text
        assert refused.json() == {
            "error": {
                "message": (
                    f"LiteLLM Virtual Key expected. Received={hashed[:4]}****{hashed[-4:]}, "
                    "expected to start with 'sk-'."
                ),
                "type": "auth_error",
                "param": None,
                "code": "401",
            }
        }, refused.text
        assert upstream_requests(gateway, text) == []


def test_a_short_non_sk_credential_is_fully_masked_in_the_refusal(gateway: Gateway) -> None:
    """The caller sends the eight character bearer token `hunter22`.

    Values of eight characters or fewer are masked completely, so the 401 message must be exactly
    "LiteLLM Virtual Key expected. Received=****, expected to start with 'sk-'." and must not echo it.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        refused: Final = chat(gateway, "hunter22", model)
        assert_denied(
            refused,
            status=401,
            error_type="auth_error",
            message="LiteLLM Virtual Key expected. Received=****, expected to start with 'sk-'.",
        )
        assert "hunter22" not in refused.text


def test_a_jwt_shaped_credential_gets_the_enable_jwt_auth_hint(gateway: Gateway) -> None:
    """The caller sends a three segment JWT shaped token to a proxy that has JWT auth disabled.

    The proxy treats it as a virtual key and must answer 401 `auth_error` with the "LiteLLM Virtual Key
    expected" message followed by the hint that names `enable_jwt_auth: true`.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        token: Final = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJkZWMxIn0.c2lnbmF0dXJl"
        refused: Final = chat(gateway, token, model)
        assert_denied(
            refused,
            status=401,
            error_type="auth_error",
            message=(
                f"LiteLLM Virtual Key expected. Received={token[:4]}****{token[-4:]}, "
                f"expected to start with 'sk-'.{_JWT_HINT}"
            ),
        )


def test_a_bearer_value_with_two_spaces_is_not_a_virtual_key(gateway: Gateway) -> None:
    """The caller sends `Authorization: Bearer  <key>` with two spaces before a valid key.

    Only one `Bearer ` prefix is stripped, the remainder starts with a space and not `sk-`, so the proxy
    must answer 401 `auth_error` with a message that starts with "LiteLLM Virtual Key expected.".
    UNVERIFIED: an HTTP server that collapses header whitespace would make this a 200; see `get_bearer_token`.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        refused: Final = gateway.client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "two spaces"}]},
            headers={"Authorization": f"Bearer  {key}"},
        )
        error: Final = assert_denied(refused, status=401, error_type="auth_error")
        assert string_value(error["message"]).startswith("LiteLLM Virtual Key expected."), refused.text


def test_a_blocked_key_gets_the_exact_blocked_body_and_no_upstream_call(gateway: Gateway) -> None:
    """A key that served once is blocked through `/key/block` and then sends a chat completion.

    The proxy must answer 401 with exactly `{"error": {"message": "Authentication Error, Key is blocked.
    Update via `/key/unblock` if you're an admin.", "type": "auth_error", "param": null, "code": "401"}}`
    and the upstream must not receive the request.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        assert_served(chat(gateway, key, model))
        gateway.post("/key/block", {"key": key})
        text: Final = marker()
        refused: Final = chat(gateway, key, model, text)
        assert refused.status_code == 401, refused.text
        assert refused.json() == {
            "error": {"message": BLOCKED_KEY_MESSAGE, "type": "auth_error", "param": None, "code": "401"}
        }, refused.text
        assert upstream_requests(gateway, text) == []


@pytest.mark.parametrize("path", ["/v1/models", "/key/info", "/model/info"])
def test_a_blocked_key_is_refused_on_read_only_routes_too(gateway: Gateway, path: str) -> None:
    """A blocked key calls a read only route that an unblocked key may call.

    The blocked check runs for every route, so `/v1/models`, `/key/info` and `/model/info` must each answer
    401 `auth_error` with the blocked key message.
    """
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        gateway.post("/key/block", {"key": key})
        refused: Final = gateway.request("GET", path, key=key)
        assert_denied(refused, status=401, error_type="auth_error", message=BLOCKED_KEY_MESSAGE)


def test_a_key_created_blocked_never_serves(gateway: Gateway) -> None:
    """An admin generates a key with `blocked: true` and the key is used immediately.

    The very first request must answer the blocked key 401, with no window in which the new key serves.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model], blocked=True)
        assert_denied(chat(gateway, key, model), status=401, error_type="auth_error", message=BLOCKED_KEY_MESSAGE)


def test_a_blocked_key_asking_for_a_model_it_may_not_use_still_gets_the_blocked_401(gateway: Gateway) -> None:
    """A blocked key requests a model outside its `models` list.

    The blocked check runs before model access, so the answer must be the 401 blocked body and not the
    403 `key_model_access_denied`. A reordered pipeline would leak which models exist to a blocked key.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        gateway.post("/key/block", {"key": key})
        assert_denied(chat(gateway, key, other), status=401, error_type="auth_error", message=BLOCKED_KEY_MESSAGE)


def test_a_key_past_its_duration_answers_401_expired_key(gateway: Gateway) -> None:
    """A key is generated with `duration: 1s`, and the test waits until the row's `expires` is in the past.

    The proxy must answer 401 with type `expired_key`, code "401" and a message that starts with
    "Authentication Error - Expired Key. Key Expiry time ", and the upstream must not be called.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model], duration="1s")
        eventually(lambda: _expired(gateway, key), bool)
        text: Final = marker()
        refused: Final = chat(gateway, key, model, text)
        assert refused.status_code == 401, refused.text
        error: Final = refused.json()["error"]
        assert (error["type"], error["code"]) == ("expired_key", "401"), refused.text
        assert error["message"].startswith("Authentication Error - Expired Key. Key Expiry time "), refused.text
        assert upstream_requests(gateway, text) == []


def test_the_expired_key_refusal_never_echoes_the_key_or_its_hash(gateway: Gateway) -> None:
    """An expired key sends a chat completion and the whole response is searched.

    `error.param` is an abbreviation of the form `sk-...XXXX`. Neither the raw key nor its full SHA-256
    may appear anywhere in the body or headers.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model], duration="1s")
        eventually(lambda: _expired(gateway, key), bool)
        refused: Final = chat(gateway, key, model)
        assert refused.status_code == 401, refused.text
        assert string_value(refused.json()["error"]["param"]).startswith("sk-..."), refused.text
        seen: Final = f"{refused.text} {refused.headers}"
        assert [secret for secret in (key, digest(key)) if secret in seen] == []


def test_the_expired_key_param_is_the_same_on_the_first_and_on_a_cached_lookup(gateway: Gateway) -> None:
    """An expired key is sent twice, so the second request finds the key object in the auth cache.

    Both refusals must carry the same `error.param`. The first lookup abbreviates the hashed token and the
    cached lookup abbreviates the raw key, so the last four characters differ between the two.
    """
    pytest.skip(
        "BUG: expired_key error.param is sk-...<last 4 of the hash> on a database lookup and "
        "sk-...<last 4 of the raw key> on a cache hit"
    )
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model], duration="1s")
        eventually(lambda: _expired(gateway, key), bool)
        first: Final = chat(gateway, key, model)
        second: Final = chat(gateway, key, model)
        assert (first.status_code, second.status_code) == (401, 401), (first.text, second.text)
        assert first.json()["error"]["param"] == second.json()["error"]["param"]


def test_a_key_that_served_while_valid_stops_at_its_expiry_without_any_management_call(gateway: Gateway) -> None:
    """A key with `duration: 3s` serves a request, which caches the key object, and is then polled.

    Within a few seconds the same key must flip to 401 `expired_key` although nothing invalidated the
    cache. Expiry is compared against the clock on every request, not only when the row is loaded.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model], duration="3s")
        first: Final = chat(gateway, key, model)
        assert first.status_code in (200, 401), first.text
        refused: Final = eventually(lambda: chat(gateway, key, model), lambda response: response.status_code == 401)
        assert refused.json()["error"]["type"] == "expired_key", refused.text


def test_an_expired_key_is_refused_on_key_info_as_well(gateway: Gateway) -> None:
    """An expired key calls `GET /key/info` for itself.

    Expiry applies to management reads too, so the proxy must answer 401 with type `expired_key`.
    """
    with gateway.scenario() as scenario:
        key: Final = scenario.key(duration="1s")
        eventually(lambda: _expired(gateway, key), bool)
        refused: Final = gateway.request("GET", "/key/info", key=key)
        assert refused.status_code == 401, refused.text
        assert refused.json()["error"]["type"] == "expired_key", refused.text


def test_a_blocked_and_expired_key_answers_blocked(gateway: Gateway) -> None:
    """A key is both past its expiry and blocked.

    The blocked check runs first, so the proxy must answer the blocked key message with type `auth_error`
    and not `expired_key`.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model], duration="1s")
        gateway.post("/key/block", {"key": key})
        eventually(lambda: _expired(gateway, key), bool)
        assert_denied(chat(gateway, key, model), status=401, error_type="auth_error", message=BLOCKED_KEY_MESSAGE)


def test_the_master_key_reaches_any_model_without_a_key_row(gateway: Gateway) -> None:
    """The master key sends a chat completion for a model that was just added.

    The proxy must answer 200. The master key has no `models` list, team or budget and no row in the
    verification token table, so a rewrite that resolves every credential through the database would 401.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        assert_served(chat(gateway, gateway.key, model))
        assert (
            gateway.rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest(gateway.key),)) == []
        )


def test_the_master_key_spend_row_does_not_store_the_master_key_or_its_hash(gateway: Gateway) -> None:
    """The master key serves one chat completion and its spend log row is read.

    The row's `api_key` must be neither the raw master key nor its SHA-256, because the proxy substitutes
    the alias `litellm_proxy_master_key` before logging. The row is found by the unique message text.
    UNVERIFIED: the stored value is expected to be the alias or the alias's hash; see
    `LITELLM_PROXY_MASTER_KEY_ALIAS` and `get_logged_api_key`.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        text: Final = marker()
        body: Final = assert_served(chat(gateway, gateway.key, model, text))
        rows: Final = eventually(
            lambda: gateway.rows(
                'SELECT api_key FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (string_value(body["id"]),)
            ),
            lambda found: len(found) == 1,
            seconds=70,
        )
        assert rows[0]["api_key"] not in (gateway.key, digest(gateway.key)), rows


@pytest.mark.parametrize("route", ["/global/spend/reset"])
def test_a_master_key_only_route_refuses_a_valid_virtual_key(gateway: Gateway, route: str) -> None:
    """A valid virtual key calls `POST /global/spend/reset`.

    The proxy must answer 401 `auth_error` with "Authentication Error, Tried to access route=/global/spend/reset,
    which is only for MASTER KEY". No role or `allowed_routes` entry can open this route.
    """
    with gateway.scenario() as scenario:
        key: Final = scenario.key(allowed_routes=[route])
        refused: Final = gateway.request("POST", route, {}, key=key)
        assert_denied(
            refused,
            status=401,
            error_type="auth_error",
            message=f"Authentication Error, Tried to access route={route}, which is only for MASTER KEY",
        )


def test_a_master_key_only_route_refuses_an_unknown_key_before_looking_it_up(gateway: Gateway) -> None:
    """A never issued `sk-` key calls `GET /memory-usage-in-mem-cache`.

    The master key only check runs before the database lookup, so the answer must be the "only for MASTER
    KEY" 401 with type `auth_error`, not the generic `token_not_found_in_db` 401.
    """
    refused: Final = gateway.request("GET", "/memory-usage-in-mem-cache", key="sk-dec1-keys-never-issued-master-only")
    error: Final = assert_denied(
        refused,
        status=401,
        error_type="auth_error",
        message=(
            "Authentication Error, Tried to access route=/memory-usage-in-mem-cache, which is only for MASTER KEY"
        ),
    )
    assert error["message"] != UNKNOWN_KEY_MESSAGE


def test_a_virtual_key_equal_to_the_master_key_with_a_suffix_is_not_the_master_key(gateway: Gateway) -> None:
    """The caller sends the master key with one extra character appended.

    The comparison is exact, so the proxy must not grant admin access. The answer must be a 401 and the
    body must not contain the master key.
    """
    refused: Final = gateway.request("GET", "/key/list", key=f"{gateway.key}x")
    assert refused.status_code == 401, refused.text
    assert gateway.key not in refused.text
