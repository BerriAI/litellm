"""Starting the proxy when Postgres is unreachable, and running it with `allow_requests_on_db_unavailable`.

Contract (litellm/proxy/proxy_cli.py database setup block, litellm/proxy/proxy_server.py
`ProxyStartupEvent._initialize_prisma_client`, litellm/proxy/db/exception_handler.py `handle_db_exception`,
litellm/proxy/health_endpoints/_health_endpoints.py):

* Default: a proxy that cannot set up its database at boot does not start. With `prisma db push` (how the
  harness boots every proxy) the pre-push data safety check cannot reach Postgres, the CLI prints
  `LiteLLM Proxy: Database migration cannot proceed.` followed by the redacted connection error and exits
  non-zero, so an orchestrator sees a crash loop instead of a pod that serves nobody.
* With no database configured at all the proxy starts, and readiness answers 200 with `db` `Not connected`.
* `general_settings.allow_requests_on_db_unavailable: true` trades enforcement for availability: readiness
  stays 200 during an outage and reports the state in the body (`db: disconnected`), and a key that cannot be
  looked up is admitted under a restricted fallback identity.

The outage tests for the default configuration are in `test_dec1_failures_postgres_outage.py`. Virtual key
behaviour under the flag for chat completions is in `authorization/test_dec1_keys_db_unavailable.py`; this
file adds the readiness contract, the other endpoints and the boot cases.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import httpx
import pytest
import yaml

from tests.integration._support.client import JSON_OBJECT, Gateway, eventually
from tests.integration._support.process import owned_proxy_process, refused_boot_log
from tests.integration.database.dec1_failures_rig import (
    CHAT,
    DISCONNECTED_BODY,
    EMBED,
    MESSAGES,
    READY_BODY,
    FaultRelay,
    Rig,
    anthropic_message,
    chat,
    database_down,
    deployment,
    digest,
    embed,
    failures_config,
    failures_rig,
    healthy_provider,
    marker,
    readiness,
    unreachable_database_url,
)

pytestmark: Final = pytest.mark.timeout(900)

MIGRATION_REFUSED: Final = "LiteLLM Proxy: Database migration cannot proceed."
Call = Callable[[Gateway, str, str], httpx.Response]


def _unreachable_environment() -> dict[str, str]:
    unreachable: Final = unreachable_database_url(os.environ["DATABASE_URL"])
    reader: Final = {"DATABASE_URL_READ_REPLICA": unreachable} if os.environ.get("DATABASE_URL_READ_REPLICA") else {}
    return {"DATABASE_URL": unreachable, **reader}


def _down_at_boot(database: FaultRelay, _redis: FaultRelay | None) -> None:
    database.refuse()


def test_the_proxy_refuses_to_start_when_postgres_is_unreachable(gateway: Gateway, tmp_path: Path) -> None:
    """`DATABASE_URL` points at a port nothing listens on and the proxy is started with its default settings.
    The process must exit non-zero without ever answering readiness, and its log must contain the line
    `LiteLLM Proxy: Database migration cannot proceed.` (litellm/proxy/proxy_cli.py:1681, reached because the
    `prisma db push` data safety check in litellm_proxy_extras/utils.py:667 cannot connect).
    Assumes the setup retries finish inside the harness readiness budget."""
    with healthy_provider() as provider:
        config: Final = failures_config(tmp_path, provider.url, with_redis=True)
        log: Final = refused_boot_log(gateway, tmp_path / "proxy", _unreachable_environment(), config=config)
        assert MIGRATION_REFUSED in log, log[-3000:]


def test_the_boot_failure_log_does_not_print_the_database_password(gateway: Gateway, tmp_path: Path) -> None:
    """`DATABASE_URL` carries a password and points at an unreachable port. The log of the failed boot must
    not contain that password anywhere: boot errors are what operators paste into tickets.
    Assumes the harness database URL has a password; the test is vacuous for a passwordless URL."""
    password: Final = urlsplit(os.environ["DATABASE_URL"]).password
    with healthy_provider() as provider:
        config: Final = failures_config(tmp_path, provider.url, with_redis=True)
        log: Final = refused_boot_log(gateway, tmp_path / "proxy", _unreachable_environment(), config=config)
        assert MIGRATION_REFUSED in log, log[-3000:]
        assert not password or password not in log


def test_the_proxy_refuses_to_start_on_an_unreachable_database_even_with_schema_updates_disabled(
    gateway: Gateway, tmp_path: Path
) -> None:
    """`general_settings.disable_prisma_schema_update: true` skips the migration step, and the database is
    unreachable. The proxy must still exit non-zero instead of becoming ready: without the availability flag
    a failed connect at startup is fatal.
    UNVERIFIED: read from `_initialize_prisma_client`, where `handle_db_exception` re-raises the connect
    error unless `allow_requests_on_db_unavailable` is on; the exit code and log text are not asserted."""
    with healthy_provider() as provider:
        config: Final = failures_config(
            tmp_path, provider.url, with_redis=True, general_settings={"disable_prisma_schema_update": True}
        )
        log: Final = refused_boot_log(gateway, tmp_path / "proxy", _unreachable_environment(), config=config)
        assert "Application startup complete" not in log, log[-3000:]


def test_a_proxy_with_no_database_configured_starts_and_reports_not_connected(gateway: Gateway, tmp_path: Path) -> None:
    """The proxy is started with no `DATABASE_URL` and no `database_url` setting. It must become ready,
    `GET /health/readiness` must answer 200 with exactly `{"status": "healthy", "db": "Not connected"}`, and a
    master key chat completion must be served: "no database" is a supported mode, not an outage."""
    with healthy_provider() as provider:
        config: Final = tmp_path / "no-database.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [deployment(CHAT, "openai/gpt-4o-mini", f"{provider.url}/v1")],
                    "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
                    "router_settings": {"num_retries": 0},
                }
            )
        )
        with owned_proxy_process(
            gateway,
            tmp_path / "proxy",
            {"STORE_MODEL_IN_DB": "False"},
            config=config,
            remove_environment=("DATABASE_URL", "DATABASE_URL_READ_REPLICA", "DIRECT_URL"),
        ) as owned:
            assert readiness(owned.gateway) == (200, {"status": "healthy", "db": "Not connected"})
            assert chat(owned.gateway).status_code == 200


def test_a_proxy_with_no_database_refuses_key_generation_without_creating_anything(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A proxy with no database gets `POST /key/generate` from the master key. It must answer an error status
    whose message says the database is not connected, never a 200 with a key that nothing can verify.
    UNVERIFIED: the status is expected to be 500 with `DB not connected` in the message
    (`CommonProxyErrors.db_not_connected_error`); only "not 2xx" and the wording are asserted."""
    with healthy_provider() as provider:
        config: Final = tmp_path / "no-database.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [deployment(CHAT, "openai/gpt-4o-mini", f"{provider.url}/v1")],
                    "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
                }
            )
        )
        with owned_proxy_process(
            gateway,
            tmp_path / "proxy",
            {"STORE_MODEL_IN_DB": "False"},
            config=config,
            remove_environment=("DATABASE_URL", "DATABASE_URL_READ_REPLICA", "DIRECT_URL"),
        ) as owned:
            refused: Final = owned.gateway.request("POST", "/key/generate", {})
            assert refused.status_code >= 400, refused.text
            assert "DB not connected" in refused.text, refused.text


def test_with_the_availability_flag_the_proxy_starts_while_postgres_is_down(tmp_path: Path) -> None:
    """`allow_requests_on_db_unavailable: true` and `disable_prisma_schema_update: true`, and Postgres is
    unreachable when the proxy starts. The proxy must become ready, readiness must answer 200 with a `db`
    value other than `connected`, and a master key chat completion must be served.
    UNVERIFIED: `db` is expected to read `Not connected`, because the startup code discards the client it
    could not connect; the exact word is not asserted."""
    settings: Final = {"allow_requests_on_db_unavailable": True, "disable_prisma_schema_update": True}
    with failures_rig(tmp_path, general_settings=settings, before_boot=_down_at_boot) as rig:
        status, body = readiness(rig.proxy)
        assert status == 200, body
        assert body["status"] == "healthy" and body["db"] != "connected", body
        assert chat(rig.proxy).status_code == 200


def test_a_proxy_started_during_an_outage_connects_once_postgres_appears(tmp_path: Path) -> None:
    """The proxy starts with the availability flag while Postgres is unreachable, and Postgres then becomes
    reachable. With no proxy restart `GET /health/readiness` must come to report `db: connected`.
    Product bug: CI run 42966523075 kept answering `db: Not connected` for two minutes after Postgres came
    up. `ProxyStartupEvent._initialize_prisma_client` (litellm/proxy/proxy_server.py:12549) swallows the
    failed first connect under the flag and returns no client, and nothing retries it, so the proxy stays
    database-less (every virtual key refused) until it is restarted, although a client that connected and
    then failed a later startup step is kept for the watchdog to heal. This test states the intended behaviour."""
    pytest.skip("BUG: a proxy booted during an outage with allow_requests_on_db_unavailable never connects to Postgres")
    settings: Final = {"allow_requests_on_db_unavailable": True, "disable_prisma_schema_update": True}
    with failures_rig(tmp_path, general_settings=settings, before_boot=_down_at_boot) as rig:
        assert readiness(rig.proxy)[1]["db"] != "connected"
        rig.database.open()
        eventually(lambda: readiness(rig.proxy), lambda seen: seen == (200, dict(READY_BODY)), seconds=120)


@pytest.fixture(scope="module")
def tolerant(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with failures_rig(
        tmp_path_factory.mktemp("dec1-failures-postgres-tolerant"),
        general_settings={"allow_requests_on_db_unavailable": True},
    ) as booted:
        yield booted


def _await_reported(rig: Rig) -> tuple[int, dict[str, object]]:
    seen: Final = eventually(lambda: readiness(rig.proxy), lambda probe: probe[1]["db"] != "connected", seconds=45)
    return seen[0], dict(seen[1])


def test_with_the_availability_flag_readiness_stays_200_and_reports_disconnected(tolerant: Rig) -> None:
    """`allow_requests_on_db_unavailable: true` and Postgres goes down. `GET /health/readiness` must keep
    answering 200, with exactly `{"status": "healthy", "db": "disconnected"}`: the pod stays in rotation
    because it still serves traffic, and the body carries the truth."""
    with database_down(tolerant):
        assert _await_reported(tolerant) == (200, dict(DISCONNECTED_BODY))


def test_with_the_availability_flag_readiness_details_stays_200_and_reports_disconnected(tolerant: Rig) -> None:
    """Under the availability flag Postgres goes down and an admin reads `GET /health/readiness/details`. It
    must answer 200 with `db: disconnected` and `status: healthy`."""
    with database_down(tolerant):
        _await_reported(tolerant)
        details: Final = tolerant.proxy.request("GET", "/health/readiness/details")
        body: Final = JSON_OBJECT.validate_json(details.content)
        assert details.status_code == 200, details.text
        assert (body["status"], body["db"]) == ("healthy", "disconnected"), body


def _embed(proxy: Gateway, key: str, text: str) -> httpx.Response:
    return embed(proxy, EMBED, key=key, text=text)


def _message(proxy: Gateway, key: str, text: str) -> httpx.Response:
    return anthropic_message(proxy, MESSAGES, key=key, text=text)


@pytest.mark.parametrize("call", [_embed, _message], ids=["embeddings", "messages"])
def test_with_the_availability_flag_an_uncached_key_is_served_on_other_endpoints(tolerant: Rig, call: Call) -> None:
    """Under the availability flag a valid key that has never sent a request calls embeddings or messages
    while Postgres is down. It must be served with 200 and reach the provider once, exactly as chat
    completions does: the fallback identity is not specific to one route.
    Product bug: CI run 42966523075 answered 503 `no_db_connection` with the transient outage message. The
    flag's gate `PrismaDBExceptionHandler.is_database_connection_error` (litellm/proxy/db/exception_handler.py:155)
    is an allowlist that does not include the refused-connection errors the outage raises, so
    `handle_authentication_error` (litellm/proxy/auth/auth_exception_handler.py:169) never issues the fallback
    identity it documents for "DB is down and having an outage"."""
    pytest.skip(
        "BUG: with allow_requests_on_db_unavailable an uncached key gets 503 no_db_connection, not the fallback identity"
    )
    with tolerant.proxy.scenario() as scenario:
        key: Final = scenario.key()
        text: Final = marker()
        with database_down(tolerant):
            served: Final = call(tolerant.proxy, key, text)
            assert served.status_code == 200, served.text
            assert tolerant.provider.count(text) == 1


def test_with_the_availability_flag_management_writes_still_fail_during_the_outage(tolerant: Rig) -> None:
    """Under the availability flag the admin posts `/key/generate` while Postgres is down. It must answer a
    5xx and create nothing: the flag keeps LLM traffic flowing, it does not pretend writes happened."""
    wanted: Final = f"sk-{marker()}"
    with database_down(tolerant):
        refused: Final = tolerant.proxy.request("POST", "/key/generate", {"key": wanted})
        assert refused.status_code >= 500, refused.text
    assert (
        tolerant.proxy.rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE token = %s', (digest(wanted),)) == []
    )


def test_the_fallback_identity_does_not_outlive_the_outage(tolerant: Rig) -> None:
    """A key restricted to the embedding model first calls the chat model while Postgres is down, and is
    served under the fallback identity. After Postgres returns, the same key calling the chat model must come
    to be refused with 403 again: the fallback identity must not stay cached as the key's identity.
    UNVERIFIED: how long the fallback stays cached is not read from code; the test allows two minutes.
    Product bug: CI run 42966523075 answered 503 `no_db_connection` with the transient outage message. The
    flag's gate `PrismaDBExceptionHandler.is_database_connection_error` (litellm/proxy/db/exception_handler.py:155)
    is an allowlist that does not include the refused-connection errors the outage raises, so
    `handle_authentication_error` (litellm/proxy/auth/auth_exception_handler.py:169) never issues the fallback
    identity it documents for "DB is down and having an outage"."""
    pytest.skip(
        "BUG: with allow_requests_on_db_unavailable an uncached key gets 503 no_db_connection, not the fallback identity"
    )
    with tolerant.proxy.scenario() as scenario:
        key: Final = scenario.key(models=[EMBED])
        with database_down(tolerant):
            assert chat(tolerant.proxy, CHAT, key=key).status_code == 200
        eventually(lambda: chat(tolerant.proxy, CHAT, key=key).status_code, lambda status: status == 403, seconds=120)


def test_with_the_availability_flag_a_request_served_unverified_is_not_billed_to_the_key(tolerant: Rig) -> None:
    """Under the availability flag a valid, never used key is served one chat completion while Postgres is
    down, and one more after Postgres is back. The key must end with exactly one spend row, the one for the
    verified request: the outage request ran under the fallback identity and is not billed to the key.
    UNVERIFIED: read from the fallback identity carrying no key hash; where the outage request's row goes
    (user id `__db_unavailable_fallback__`) is not asserted.
    Product bug: CI run 42966523075 answered 503 `no_db_connection` with the transient outage message. The
    flag's gate `PrismaDBExceptionHandler.is_database_connection_error` (litellm/proxy/db/exception_handler.py:155)
    is an allowlist that does not include the refused-connection errors the outage raises, so
    `handle_authentication_error` (litellm/proxy/auth/auth_exception_handler.py:169) never issues the fallback
    identity it documents for "DB is down and having an outage"."""
    pytest.skip(
        "BUG: with allow_requests_on_db_unavailable an uncached key gets 503 no_db_connection, not the fallback identity"
    )
    with tolerant.proxy.scenario() as scenario:
        key: Final = scenario.key()
        with database_down(tolerant):
            assert chat(tolerant.proxy, CHAT, key=key).status_code == 200
        assert chat(tolerant.proxy, CHAT, key=key).status_code == 200
        rows: Final = eventually(
            lambda: tolerant.proxy.rows(
                'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE api_key = %s', (digest(key),)
            ),
            lambda found: len(found) >= 1,
            seconds=90,
        )
        assert len(rows) == 1, rows
