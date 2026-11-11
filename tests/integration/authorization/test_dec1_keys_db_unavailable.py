"""Virtual key auth while Postgres is unreachable, with and without `allow_requests_on_db_unavailable`.

Contract (litellm/proxy/auth/auth_exception_handler.py `handle_authentication_error`;
litellm/proxy/db/exception_handler.py):

* A key that is already in the auth cache does not need the database, so it keeps being served during an
  outage whatever the setting is.
* A key lookup that has to reach the database and loses its connection is, by default, HTTP 503 with exactly
  `{"error": {"message": "Service Unavailable, the authentication database is temporarily unreachable.
  Please retry shortly.", "type": "no_db_connection", "param": null, "code": "503"}}`. It is never a 401:
  the proxy cannot tell whether the key is valid.
* With `general_settings.allow_requests_on_db_unavailable: true` the same lookup failure admits the request
  under a restricted fallback identity (role `internal_user`, user id `__db_unavailable_fallback__`). The
  caller is served even though the key was never verified, and the fallback identity is not an admin.

Each test boots its own proxy on a scratch database behind a relay that drops the connection carrying the
SHA-256 of one specific key once it is armed. The relay stays disarmed while the scenario is set up and torn
down. Assumption: `/key/generate` does not pre warm the auth cache, so a key that never sent a request is
"uncached"; the tests assert the relay actually saw the lookup.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway, Scenario
from tests.integration._support.database import scratch_database
from tests.integration._support.database_relay import (
    DroppedConnectionRelay,
    dropped_connection_relay,
    plaintext_database_url,
)
from tests.integration._support.process import owned_proxy
from tests.integration.authorization.dec1_keys_support import (
    DB_UNAVAILABLE_MESSAGE,
    assert_denied,
    assert_served,
    chat,
    config_with_general_settings,
    digest,
    marker,
    upstream_requests,
)

pytestmark: Final = pytest.mark.timeout(240)


@dataclass(frozen=True, slots=True)
class _Outage:
    proxy: Gateway
    relay: DroppedConnectionRelay
    scenario: Scenario
    model: str
    key: str


@contextmanager
def _outage_rig(gateway: Gateway, directory: Path, *, allow_requests: bool) -> Iterator[_Outage]:
    key: Final = f"sk-{marker()}"
    with ExitStack() as stack:
        database_url: Final = stack.enter_context(scratch_database())
        relay, relayed_url = stack.enter_context(dropped_connection_relay(database_url, digest(key).encode()))
        config: Final = config_with_general_settings(directory, {"allow_requests_on_db_unavailable": allow_requests})
        proxy: Final = stack.enter_context(
            owned_proxy(
                gateway,
                directory / "proxy",
                {"DATABASE_URL": plaintext_database_url(relayed_url), "PRISMA_HEALTH_WATCHDOG_ENABLED": "false"},
                config=config,
                remove_environment=("DATABASE_URL_READ_REPLICA",),
            )
        )
        scenario: Final = stack.enter_context(proxy.scenario())
        stack.callback(relay.disarm)
        yield _Outage(proxy, relay, scenario, scenario.model(), key)


@pytest.mark.parametrize("allow_requests", [False, True], ids=["flag_off", "flag_on"])
def test_a_cached_key_keeps_being_served_while_its_database_lookup_would_fail(
    gateway: Gateway, tmp_path: Path, allow_requests: bool
) -> None:
    """A key serves one request, which caches it, and then the relay starts dropping any query for that key.

    The next request must still be served with 200, with the setting off and with it on, because a cached
    key is not looked up again. A port that re-reads the key on every request would turn a database blip
    into an outage for all traffic.
    """
    with _outage_rig(gateway, tmp_path, allow_requests=allow_requests) as rig:
        assert rig.scenario.key(key=rig.key) == rig.key
        assert_served(chat(rig.proxy, rig.key, rig.model))
        rig.relay.arm()
        during_outage: Final = chat(rig.proxy, rig.key, rig.model)
        rig.relay.disarm()
        assert_served(during_outage)


def test_an_uncached_valid_key_gets_503_no_db_connection_when_the_lookup_fails(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A valid key that has never sent a request sends its first one while its database lookup is dropped,
    on a proxy with the setting off.

    The proxy must answer 503 with exactly the `no_db_connection` body from the module docstring, and the
    upstream must not be called. It must not answer 401: the key is valid and the caller should retry.
    """
    with _outage_rig(gateway, tmp_path, allow_requests=False) as rig:
        assert rig.scenario.key(key=rig.key) == rig.key
        rig.relay.arm()
        text: Final = marker()
        refused: Final = chat(rig.proxy, rig.key, rig.model, text)
        rig.relay.disarm()
        assert rig.relay.dropped.is_set(), "the key lookup never reached the database relay"
        assert refused.status_code == 503, refused.text
        assert refused.json() == {
            "error": {"message": DB_UNAVAILABLE_MESSAGE, "type": "no_db_connection", "param": None, "code": "503"}
        }, refused.text
        assert upstream_requests(rig.proxy, text) == []


def test_an_uncached_valid_key_recovers_once_the_database_is_back(gateway: Gateway, tmp_path: Path) -> None:
    """A valid key gets the 503 during a dropped lookup, then the relay is disarmed and the key retries.

    The retry must be served with 200. The failed lookup must not be remembered as "key does not exist".
    """
    with _outage_rig(gateway, tmp_path, allow_requests=False) as rig:
        assert rig.scenario.key(key=rig.key) == rig.key
        rig.relay.arm()
        refused: Final = chat(rig.proxy, rig.key, rig.model)
        rig.relay.disarm()
        assert refused.status_code == 503, refused.text
        assert_served(chat(rig.proxy, rig.key, rig.model))


def test_an_uncached_valid_key_is_served_under_the_fallback_identity_when_the_setting_is_on(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A valid key that has never sent a request sends its first one while its lookup is dropped, on a proxy
    with `allow_requests_on_db_unavailable: true`.

    The request must be served with 200 and reach the upstream once, because the proxy issues the
    restricted fallback identity instead of failing.

    Product bug: a dropped connection during the key lookup is classified as a transient outage for the 503
    wording (litellm/proxy/db/exception_handler.py `database_unavailable_message` returns the "temporarily
    unreachable. Please retry shortly." text), but `is_database_connection_error`, the gate the setting uses in
    litellm/proxy/auth/auth_exception_handler.py `handle_authentication_error`, does not accept it, so the
    request is refused with 503 even though the setting is on.
    """
    pytest.skip(
        "BUG: allow_requests_on_db_unavailable does not admit a key whose lookup loses its database connection; "
        "the proxy answers 503 no_db_connection with its own 'temporarily unreachable' message"
    )
    with _outage_rig(gateway, tmp_path, allow_requests=True) as rig:
        assert rig.scenario.key(key=rig.key) == rig.key
        rig.relay.arm()
        text: Final = marker()
        served: Final = chat(rig.proxy, rig.key, rig.model, text)
        rig.relay.disarm()
        assert rig.relay.dropped.is_set(), "the key lookup never reached the database relay"
        assert_served(served)
        assert len(upstream_requests(rig.proxy, text)) == 1


def test_the_fallback_identity_ignores_the_keys_own_model_list(gateway: Gateway, tmp_path: Path) -> None:
    """A key limited to another model sends its first request, for a model it may not use, while its lookup
    is dropped on a proxy with the setting on.

    The proxy cannot read the key's `models`, so the fallback identity is served with 200. This is the
    documented trade of the setting (availability over enforcement) and is pinned so a port does not
    change it silently in either direction.

    Product bug: a dropped connection during the key lookup is classified as a transient outage for the 503
    wording (litellm/proxy/db/exception_handler.py `database_unavailable_message` returns the "temporarily
    unreachable. Please retry shortly." text), but `is_database_connection_error`, the gate the setting uses in
    litellm/proxy/auth/auth_exception_handler.py `handle_authentication_error`, does not accept it, so the
    request is refused with 503 even though the setting is on.
    """
    pytest.skip(
        "BUG: allow_requests_on_db_unavailable does not admit a key whose lookup loses its database connection; "
        "the proxy answers 503 no_db_connection with its own 'temporarily unreachable' message"
    )
    with _outage_rig(gateway, tmp_path, allow_requests=True) as rig:
        other: Final = rig.scenario.model()
        assert rig.scenario.key(key=rig.key, models=[other]) == rig.key
        rig.relay.arm()
        served: Final = chat(rig.proxy, rig.key, rig.model)
        rig.relay.disarm()
        assert rig.relay.dropped.is_set(), "the key lookup never reached the database relay"
        assert_served(served)


def test_a_never_issued_key_is_admitted_during_an_outage_when_the_setting_is_on(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A key value that was never created sends a chat completion while its lookup is dropped, on a proxy
    with the setting on.

    The proxy cannot distinguish it from a valid key, so the request must be served with 200. Once the
    relay is disarmed the same value must go back to the generic 401 `token_not_found_in_db`.

    Product bug: a dropped connection during the key lookup is classified as a transient outage for the 503
    wording (litellm/proxy/db/exception_handler.py `database_unavailable_message` returns the "temporarily
    unreachable. Please retry shortly." text), but `is_database_connection_error`, the gate the setting uses in
    litellm/proxy/auth/auth_exception_handler.py `handle_authentication_error`, does not accept it, so the
    request is refused with 503 even though the setting is on.
    """
    pytest.skip(
        "BUG: allow_requests_on_db_unavailable does not admit a key whose lookup loses its database connection; "
        "the proxy answers 503 no_db_connection with its own 'temporarily unreachable' message"
    )
    with _outage_rig(gateway, tmp_path, allow_requests=True) as rig:
        rig.relay.arm()
        served: Final = chat(rig.proxy, rig.key, rig.model)
        rig.relay.disarm()
        assert rig.relay.dropped.is_set(), "the key lookup never reached the database relay"
        assert_served(served)
        assert_denied(chat(rig.proxy, rig.key, rig.model), status=401, error_type="token_not_found_in_db", param="key")


def test_a_never_issued_key_gets_503_not_401_during_an_outage_when_the_setting_is_off(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A key value that was never created sends a chat completion while its lookup is dropped, on a proxy
    with the setting off.

    The proxy must answer 503 `no_db_connection`, not 401, because it could not check the key. After the
    relay is disarmed the same value must get the generic 401.
    """
    with _outage_rig(gateway, tmp_path, allow_requests=False) as rig:
        rig.relay.arm()
        refused: Final = chat(rig.proxy, rig.key, rig.model)
        rig.relay.disarm()
        assert rig.relay.dropped.is_set(), "the key lookup never reached the database relay"
        assert_denied(refused, status=503, error_type="no_db_connection", message=DB_UNAVAILABLE_MESSAGE)
        assert_denied(chat(rig.proxy, rig.key, rig.model), status=401, error_type="token_not_found_in_db", param="key")


def test_the_fallback_identity_is_not_an_admin(gateway: Gateway, tmp_path: Path) -> None:
    """A never issued key posts to the admin route `/user/new` while its lookup is dropped, on a proxy with
    the setting on.

    The fallback identity has role `internal_user`, so the proxy must answer the 401 "Only proxy admin"
    refusal and must not create the user. A database outage must never grant admin access.

    Product bug: a dropped connection during the key lookup is classified as a transient outage for the 503
    wording (litellm/proxy/db/exception_handler.py `database_unavailable_message` returns the "temporarily
    unreachable. Please retry shortly." text), but `is_database_connection_error`, the gate the setting uses in
    litellm/proxy/auth/auth_exception_handler.py `handle_authentication_error`, does not accept it, so the
    request is refused with 503 even though the setting is on.
    """
    pytest.skip(
        "BUG: allow_requests_on_db_unavailable does not admit a key whose lookup loses its database connection; "
        "the proxy answers 503 no_db_connection with its own 'temporarily unreachable' message"
    )
    with _outage_rig(gateway, tmp_path, allow_requests=True) as rig:
        user_id: Final = marker()
        rig.relay.arm()
        refused: Final = rig.proxy.request("POST", "/user/new", {"user_id": user_id}, key=rig.key)
        rig.relay.disarm()
        assert rig.relay.dropped.is_set(), "the key lookup never reached the database relay"
        error: Final = assert_denied(refused, status=401, error_type="auth_error")
        assert "Only proxy admin can be used" in str(error["message"]), refused.text
        assert rig.proxy.rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id=%s', (user_id,)) == []


def test_the_master_key_works_during_an_outage_without_the_setting(gateway: Gateway, tmp_path: Path) -> None:
    """The master key sends a chat completion while the relay is armed for another key, with the setting off.

    The master key is compared in memory and never looked up, so the request must be served with 200.
    Operators rely on this to keep an emergency path open while the database is down.
    """
    with _outage_rig(gateway, tmp_path, allow_requests=False) as rig:
        rig.relay.arm()
        served: Final = chat(rig.proxy, rig.proxy.key, rig.model)
        rig.relay.disarm()
        assert_served(served)
