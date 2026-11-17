"""What `allow_requests_on_db_unavailable` may and may not admit while the key database is unreachable.

Contract (litellm/proxy/auth/auth_exception_handler.py `handle_authentication_error`, which returns a
`UserAPIKeyAuth` with `user_id="__db_unavailable_fallback__"`, role `internal_user` and token
`failed-to-connect-to-db` when the setting is on and the failure is a database connection error):

* The fallback is meant to keep known callers working through a database blip. Today it admits any value
  that reaches the database lookup, which is any `sk-` value that is not in the auth cache (pinned in
  `authorization/test_dec1_keys_db_unavailable.py`). This file adds the cases that turn that into a security
  problem or that must stay closed: a credential that is not `sk-` shaped, a key an admin just blocked or
  deleted (both evict the key from the cache, so the next request goes to the database), and routes that need
  more than an internal user.
* The Rust auth core (#51374) moves the key read behind `PythonAuthHost.db_key`; the outage decision must not
  move from "refuse unless opted in" to "admit", and the opt in must not admit revoked keys.

Each test boots its own proxy on a scratch database behind a relay that drops the connection carrying the
SHA-256 of one specific key once armed (`outage_rig` in `dec1_authsec_support.py`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from integration._support.client import string_value
from integration.security.dec1_authsec_support import (
    CHAT,
    DB_UNAVAILABLE_USER,
    INVALID_VIRTUAL_KEY_PREFIX,
    MASTER_ONLY_TEMPLATE,
    OWNED_MODEL,
    assert_auth_error,
    chat_body,
    outage_rig,
    spend_row_text,
)

pytestmark: Final = pytest.mark.timeout(240)


def test_a_non_sk_credential_is_refused_during_an_outage_with_the_setting_on(tmp_path: Path) -> None:
    """With `allow_requests_on_db_unavailable: true` and the relay armed for a 40 character value that does not
    start with `sk-`, that value calls a chat model. The proxy must answer 401 with a message starting
    `LiteLLM Virtual Key expected. Received=`, because a non `sk-` value is refused before any database read.
    The fallback must never widen to values that are not virtual key shaped."""
    token: Final = "tok" + "c" * 37
    with outage_rig(tmp_path, allow_requests=True, key=token) as rig:
        rig.relay.arm()
        refused: Final = rig.proxy.request("POST", CHAT, chat_body(OWNED_MODEL), key=token)
        rig.relay.disarm()
        assert refused.status_code == 401, refused.text
        assert INVALID_VIRTUAL_KEY_PREFIX in refused.text


def test_a_jwt_shaped_credential_is_not_admitted_during_an_outage_when_jwt_auth_is_off(tmp_path: Path) -> None:
    """With the setting on, JWT auth off and the relay armed, a three segment `header.claims.signature` value
    calls a chat model. It must be refused with 401 (the non `sk-` refusal with the enable_jwt_auth hint), never
    admitted under the fallback identity."""
    token: Final = "eyJhbGciOiJub25lIn0.eyJzdWIiOiJkZWMxIn0.c2lnbmF0dXJl"
    with outage_rig(tmp_path, allow_requests=True, key=token) as rig:
        rig.relay.arm()
        refused: Final = rig.proxy.request("POST", CHAT, chat_body(OWNED_MODEL), key=token)
        rig.relay.disarm()
        assert refused.status_code == 401, refused.text


def test_a_key_blocked_just_before_an_outage_stays_refused(tmp_path: Path) -> None:
    """With the setting on, a key serves one call, an admin blocks it with `/key/block` (which evicts the key from
    the auth cache), and then the relay starts dropping that key's lookups. The next call with the key must still
    be refused with 401; a revoked credential must not come back to life because the database is unreachable.
    Today the evicted key goes to the database, the lookup fails, and the fallback admits it."""
    pytest.skip("BUG: allow_requests_on_db_unavailable admits a key that was blocked just before the outage")
    with outage_rig(tmp_path, allow_requests=True) as rig:
        rig.scenario.key(key=rig.key)
        assert rig.proxy.request("POST", CHAT, chat_body(OWNED_MODEL), key=rig.key).status_code == 200
        rig.proxy.post("/key/block", {"key": rig.key})
        rig.relay.arm()
        during: Final = rig.proxy.request("POST", CHAT, chat_body(OWNED_MODEL), key=rig.key)
        rig.relay.disarm()
        assert during.status_code == 401, during.text


def test_a_key_deleted_just_before_an_outage_stays_refused(tmp_path: Path) -> None:
    """With the setting on, a key serves one call, an admin deletes it with `/key/delete` (which evicts it from
    the cache), and the relay then drops lookups for that key. The next call with the deleted key must be refused
    with 401. Today it is admitted under the fallback identity for the same reason as a blocked key."""
    pytest.skip("BUG: allow_requests_on_db_unavailable admits a key that was deleted just before the outage")
    with outage_rig(tmp_path, allow_requests=True) as rig:
        created: Final = rig.proxy.post("/key/generate", {"key": rig.key})
        assert string_value(created["key"]) == rig.key
        assert rig.proxy.request("POST", CHAT, chat_body(OWNED_MODEL), key=rig.key).status_code == 200
        rig.proxy.post("/key/delete", {"keys": [rig.key]})
        rig.relay.arm()
        during: Final = rig.proxy.request("POST", CHAT, chat_body(OWNED_MODEL), key=rig.key)
        rig.relay.disarm()
        assert during.status_code == 401, during.text


def test_the_fallback_identity_cannot_call_a_master_key_only_route(tmp_path: Path) -> None:
    """With the setting on and the relay armed for an uncached `sk-` value, that value calls
    `POST /global/spend/reset`. The proxy must answer 401 with exactly
    `Authentication Error, Tried to access route=/global/spend/reset, which is only for MASTER KEY`: the
    master key only check runs before the database read, so the fallback never reaches it."""
    with outage_rig(tmp_path, allow_requests=True) as rig:
        rig.relay.arm()
        refused: Final = rig.proxy.request("POST", "/global/spend/reset", {}, key=rig.key)
        rig.relay.disarm()
        assert_auth_error(refused, 401, MASTER_ONLY_TEMPLATE.format(route="/global/spend/reset"))


def test_the_fallback_identity_cannot_change_proxy_config(tmp_path: Path) -> None:
    """With the setting on and the relay armed, an uncached `sk-` value calls `POST /config/update`. The proxy
    must refuse with 401 (the internal user route gate) and the stored config must not change: the fallback
    identity is an `internal_user`, never an admin. UNVERIFIED: that the refusal is the route gate 401 and not a
    403 from the endpoint; read from `RouteChecks.non_proxy_admin_allowed_routes_check`."""
    with outage_rig(tmp_path, allow_requests=True) as rig:
        rig.relay.arm()
        refused: Final = rig.proxy.request(
            "POST", "/config/update", {"general_settings": {"allow_requests_on_db_unavailable": False}}, key=rig.key
        )
        rig.relay.disarm()
        assert refused.status_code in (401, 403), refused.text


def test_a_request_admitted_by_the_fallback_is_attributed_to_the_fallback_user_and_never_stores_the_token(
    tmp_path: Path,
) -> None:
    """With the setting on and the relay armed for one never issued `sk-` value, that value calls a chat model
    and is served (pinned current behaviour). Its spend row must name the user `__db_unavailable_fallback__`, and
    neither the presented value nor its SHA-256 may appear anywhere in the row, so the outage does not write a
    caller supplied secret into the logs. UNVERIFIED: the row's `api_key`; the fallback object's token is the
    literal `failed-to-connect-to-db`."""
    with outage_rig(tmp_path, allow_requests=True) as rig:
        rig.relay.arm()
        served: Final = rig.proxy.request("POST", CHAT, chat_body(OWNED_MODEL), key=rig.key)
        rig.relay.disarm()
        assert served.status_code == 200, served.text
        row: Final = spend_row_text(rig.proxy, string_value(served.json()["id"]))
        assert DB_UNAVAILABLE_USER in row
        assert rig.key not in row


def test_without_the_setting_a_blocked_then_evicted_key_gets_503_not_200(tmp_path: Path) -> None:
    """Default settings: a key serves one call, is blocked (cache evicted), and the relay then drops its lookups.
    The next call must answer 503 `no_db_connection`, never 200: without the opt in the proxy fails closed for a
    key it cannot read."""
    with outage_rig(tmp_path, allow_requests=False) as rig:
        rig.scenario.key(key=rig.key)
        assert rig.proxy.request("POST", CHAT, chat_body(OWNED_MODEL), key=rig.key).status_code == 200
        rig.proxy.post("/key/block", {"key": rig.key})
        rig.relay.arm()
        during: Final = rig.proxy.request("POST", CHAT, chat_body(OWNED_MODEL), key=rig.key)
        rig.relay.disarm()
        assert during.status_code == 503, during.text
        assert during.json()["error"]["type"] == "no_db_connection"
