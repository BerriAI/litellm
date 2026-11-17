"""Proxy startup against Postgres: the order of boot steps, schema updates, and the connection budget line.

Contract (litellm/proxy/proxy_cli.py `run_server` database block from line 1725, litellm/proxy/db/db_url_settings.py
`postgres_connection_budget_message`, litellm/proxy/db/prisma_client.py `should_update_prisma_schema`):

* Boot order on one pod: the CLI resolves the database URL and prints one line stating the pod's worst-case
  Postgres connection demand, then runs the schema step (migrations or `prisma db push`) to completion, and only
  then starts the HTTP server. The port is never open while the schema is being changed.
* The budget line is `LiteLLM Proxy: Postgres connection budget per pod = <workers> worker(s) x <engines> = up to
  <n> connections; keep pods x that figure under max_connections minus superuser_reserved_connections`, where
  `<engines>` adds a reader term when `DATABASE_URL_READ_REPLICA` is set. The writer's `connection_limit` is the
  configured pool limit even if the URL pins another; a reader that pins its own limit keeps it.
* `DISABLE_SCHEMA_UPDATE=true` or `general_settings.disable_prisma_schema_update: true` skips the schema step, so
  a separate migration job (the CLI with `--skip_server_startup`) must have run first.
* `--skip_server_startup` runs the whole setup, prints `LiteLLM: Setup complete. Skipping server startup as
  requested.` and exits 0 without opening a port.

Operators size Postgres from the budget line and split migrations into an init job. Whatever process model
replaces the Python CLI has to keep both. Unreachable Postgres at boot is in `test_dec1_failures_postgres_boot.py`,
migration error logging in `test_v1_migration_error_logging.py`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import pytest

from tests.integration._support.client import Gateway
from tests.integration._support.database import read_rows, scratch_database
from tests.integration._support.process import (
    LEGACY_MIGRATE_DEPLOY,
    MIGRATE_DEPLOY,
    owned_proxy_process,
    setup_only_proxy_run,
)
from tests.integration.configuration.dec1_deploy_support import (
    BUDGET_PREFIX,
    BUDGET_SUFFIX,
    READY,
    SETUP_COMPLETE,
    boot_outcome,
    budget_line,
    chat,
    deploy_config,
    digest,
    log_text,
    plain,
    provider,
    readiness,
    write_config,
)

pytestmark: Final = pytest.mark.timeout(900)

LEGACY_WARNING: Final = "LiteLLM Proxy: Using the legacy (v1) migration resolver."
V2_DEPRECATED: Final = "LiteLLM Proxy: --use_v2_migration_resolver is deprecated and has no effect"
OUT_OF_SYNC: Final = "prisma schema out of sync with db"
TABLE_EXISTS: Final = "SELECT to_regclass('public.\"LiteLLM_VerificationToken\"')::text AS name"


def _with_query(url: str, query: str) -> str:
    return urlunsplit(urlsplit(url)._replace(query=query))


def _table_exists(database_url: str) -> bool:
    return read_rows(TABLE_EXISTS, (), database_url=database_url)[0]["name"] is not None


def _budget(workers: int, engines: str, total: int) -> str:
    return f"{BUDGET_PREFIX}{workers} worker(s) x {engines} = up to {total}{BUDGET_SUFFIX}"


def test_two_workers_and_a_pool_limit_of_seven_print_a_budget_of_fourteen(gateway: Gateway, tmp_path: Path) -> None:
    """`--num_workers 2`, `database_connection_pool_limit: 7`, no read replica. The log must contain exactly
    `LiteLLM Proxy: Postgres connection budget per pod = 2 worker(s) x writer connection_limit 7 = up to 14
    connections; keep pods x that figure under max_connections minus superuser_reserved_connections`
    (db_url_settings.py:306). Every worker owns its own Prisma engine, so the per-pod figure multiplies."""
    with provider() as upstream:
        config: Final = write_config(
            tmp_path, deploy_config(upstream.api_base, general_settings={"database_connection_pool_limit": 7})
        )
        with owned_proxy_process(
            gateway, tmp_path / "proxy", {}, config=config, workers=2, remove_environment=("DATABASE_URL_READ_REPLICA",)
        ) as owned:
            line: Final = budget_line(log_text(owned))
    assert line == _budget(2, "writer connection_limit 7", 14)


def test_a_read_replica_adds_a_reader_term_to_the_budget(gateway: Gateway, tmp_path: Path) -> None:
    """One worker, `database_connection_pool_limit: 7`, and `DATABASE_URL_READ_REPLICA` set to the same scratch
    database as the writer. The budget line must read `1 worker(s) x (writer connection_limit 7 + reader
    connection_limit 7) = up to 14`: the reader engine is a second pool against the same server cap."""
    with scratch_database("dec1_deploy") as database, provider() as upstream:
        config: Final = write_config(
            tmp_path, deploy_config(upstream.api_base, general_settings={"database_connection_pool_limit": 7})
        )
        with owned_proxy_process(
            gateway,
            tmp_path / "proxy",
            {"DATABASE_URL": database, "DATABASE_URL_READ_REPLICA": database},
            config=config,
        ) as owned:
            line: Final = budget_line(log_text(owned))
    assert line == _budget(1, "(writer connection_limit 7 + reader connection_limit 7)", 14)


def test_a_reader_that_pins_its_own_connection_limit_is_counted_at_that_limit(gateway: Gateway, tmp_path: Path) -> None:
    """The read replica URL carries `?connection_limit=3` while `database_connection_pool_limit` is 7. The line
    must read `(writer connection_limit 7 + reader connection_limit 3) = up to 10`: a limit pinned on the
    replica URL wins there (proxy_cli.py:1809)."""
    with scratch_database("dec1_deploy") as database, provider() as upstream:
        config: Final = write_config(
            tmp_path, deploy_config(upstream.api_base, general_settings={"database_connection_pool_limit": 7})
        )
        with owned_proxy_process(
            gateway,
            tmp_path / "proxy",
            {"DATABASE_URL": database, "DATABASE_URL_READ_REPLICA": _with_query(database, "connection_limit=3")},
            config=config,
        ) as owned:
            line: Final = budget_line(log_text(owned))
    assert line == _budget(1, "(writer connection_limit 7 + reader connection_limit 3)", 10)


def test_the_configured_pool_limit_replaces_a_limit_pinned_on_the_writer_url(gateway: Gateway, tmp_path: Path) -> None:
    """`DATABASE_URL` carries `?connection_limit=50` and the config says `database_connection_pool_limit: 7`.
    The line must read `writer connection_limit 7`: on the writer the config is applied on top of the URL
    (`append_query_params`, proxy_cli.py:181). A rewrite that lets the URL win would multiply connections."""
    with scratch_database("dec1_deploy") as database, provider() as upstream:
        config: Final = write_config(
            tmp_path, deploy_config(upstream.api_base, general_settings={"database_connection_pool_limit": 7})
        )
        with owned_proxy_process(
            gateway,
            tmp_path / "proxy",
            {"DATABASE_URL": _with_query(database, "connection_limit=50")},
            config=config,
            remove_environment=("DATABASE_URL_READ_REPLICA",),
        ) as owned:
            line: Final = budget_line(log_text(owned))
    assert line == _budget(1, "writer connection_limit 7", 7)


def test_extra_connection_params_override_the_pool_limit_in_the_budget(gateway: Gateway, tmp_path: Path) -> None:
    """`database_connection_pool_limit: 7` and `database_extra_connection_params: {connection_limit: 9}`. The
    line must read `writer connection_limit 9 = up to 9`: extra params win over the named settings
    (`_build_db_connection_url_params`, proxy_cli.py:108) and the line is read off the final URL."""
    with provider() as upstream:
        config: Final = write_config(
            tmp_path,
            deploy_config(
                upstream.api_base,
                general_settings={
                    "database_connection_pool_limit": 7,
                    "database_extra_connection_params": {"connection_limit": 9},
                },
            ),
        )
        with owned_proxy_process(
            gateway, tmp_path / "proxy", {}, config=config, remove_environment=("DATABASE_URL_READ_REPLICA",)
        ) as owned:
            line: Final = budget_line(log_text(owned))
    assert line == _budget(1, "writer connection_limit 9", 9)


def test_without_a_pool_limit_setting_the_budget_uses_the_default_of_ten(gateway: Gateway, tmp_path: Path) -> None:
    """The config has `general_settings` but no `database_connection_pool_limit`. The line must read
    `writer connection_limit 10 = up to 10` (`LiteLLMDatabaseConnectionPool.database_connection_pool_limit`,
    proxy_cli.py:73). Under worker isolation the harness injects a limit of 2 into every config
    (tests/integration/_support/process.py:741), so the default cannot be observed there."""
    if os.environ.get("INTEGRATION_WORKER_ISOLATION") == "1":
        pytest.skip("worker isolation injects database_connection_pool_limit: 2 into the config")
    with provider() as upstream:
        config: Final = write_config(tmp_path, deploy_config(upstream.api_base))
        with owned_proxy_process(
            gateway, tmp_path / "proxy", {}, config=config, remove_environment=("DATABASE_URL_READ_REPLICA",)
        ) as owned:
            line: Final = budget_line(log_text(owned))
    assert line == _budget(1, "writer connection_limit 10", 10)


def test_the_schema_step_finishes_before_the_port_opens(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy boots on a fresh private database with `prisma db push`. The harness records, from outside, when a
    `prisma` child process was first and last seen and when the port first accepted a connection (the
    `.startup.json` written next to the log, tests/integration/_support/process.py:684). Prisma must have been
    seen, and it must have finished no later than the port started listening: a balancer never reaches a pod
    whose schema is half changed.
    UNVERIFIED: a very fast `db push` may finish between two 0.1 second probes; then `prisma_started` is null and
    the test skips rather than fails."""
    with provider() as upstream:
        config: Final = write_config(tmp_path, deploy_config(upstream.api_base))
        with owned_proxy_process(gateway, tmp_path / "proxy", {}, config=config) as owned:
            phases: Final = json.loads(owned.log.with_suffix(".startup.json").read_text())["phases"]
    if phases["prisma_started"] is None:
        pytest.skip("prisma finished between two probes; the order could not be observed")
    assert phases["listening"] is not None, phases
    assert phases["prisma_finished"] <= phases["listening"], phases


def test_the_budget_line_is_printed_before_the_server_reports_startup(gateway: Gateway, tmp_path: Path) -> None:
    """In the boot log of a normal proxy, the connection budget line must come before uvicorn's
    `Application startup complete` line: the database is configured before any worker serves.
    UNVERIFIED: that uvicorn's startup line reaches the log at the harness's default log level."""
    with provider() as upstream:
        config: Final = write_config(tmp_path, deploy_config(upstream.api_base))
        with owned_proxy_process(gateway, tmp_path / "proxy", {}, config=config) as owned:
            text: Final = log_text(owned)
    assert BUDGET_PREFIX in text and "Application startup complete" in text, text[-3000:]
    assert text.index(BUDGET_PREFIX) < text.index("Application startup complete")


def test_a_setup_only_run_creates_the_schema_prints_setup_complete_and_exits_zero(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The CLI runs with `--skip_server_startup` against an empty scratch database, the way an init container or
    migration job does. It must exit 0, print `LiteLLM: Setup complete. Skipping server startup as requested.`,
    and leave the `LiteLLM_VerificationToken` table in that database (proxy_cli.py:1957)."""
    with scratch_database("dec1_deploy") as database, provider() as upstream:
        config: Final = write_config(tmp_path, deploy_config(upstream.api_base))
        completed: Final = setup_only_proxy_run(gateway, {"DATABASE_URL": database}, config=config, workers=1)
        output: Final = plain(completed.stdout + completed.stderr)
        assert completed.returncode == 0, output[-3000:]
        assert SETUP_COMPLETE in output, output[-3000:]
        assert _table_exists(database)


def test_after_a_setup_job_a_proxy_with_schema_updates_disabled_serves_and_stores_keys(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A setup-only run prepares an empty scratch database, then a proxy boots on it with
    `DISABLE_SCHEMA_UPDATE=true`. It must answer readiness 200 `connected`, `POST /key/generate` must answer 200,
    and the key's hash must be a row of `LiteLLM_VerificationToken` in that database: the split of migration job
    and serving pods works end to end."""
    with scratch_database("dec1_deploy") as database, provider() as upstream:
        config: Final = write_config(tmp_path, deploy_config(upstream.api_base))
        assert setup_only_proxy_run(gateway, {"DATABASE_URL": database}, config=config, workers=1).returncode == 0
        with owned_proxy_process(
            gateway,
            tmp_path / "proxy",
            {"DATABASE_URL": database, "DISABLE_SCHEMA_UPDATE": "true"},
            config=config,
            remove_environment=("DATABASE_URL_READ_REPLICA",),
        ) as owned:
            assert readiness(owned.gateway) == (200, dict(READY))
            key: Final = str(owned.gateway.post("/key/generate", {})["key"])
            assert chat(owned.gateway, key=key).status_code == 200
        rows: Final = read_rows(
            'SELECT token FROM "LiteLLM_VerificationToken" WHERE token = %s', (digest(key),), database_url=database
        )
        assert len(rows) == 1


@pytest.mark.parametrize(
    ("environment", "general_settings"),
    [({"DISABLE_SCHEMA_UPDATE": "true"}, {}), ({}, {"disable_prisma_schema_update": True})],
    ids=["environment", "config"],
)
def test_with_schema_updates_disabled_an_empty_database_gets_no_tables(
    gateway: Gateway, tmp_path: Path, environment: dict[str, str], general_settings: dict[str, bool]
) -> None:
    """The proxy starts on an empty scratch database with schema updates disabled, by environment or by config.
    Whatever the boot outcome, `LiteLLM_VerificationToken` must not exist afterwards and the log must contain
    `prisma schema out of sync with db` (litellm/proxy/db/check_migration.py:119): disabling updates means the
    serving pod never changes the schema, it only reports the drift.
    UNVERIFIED: whether such a proxy then refuses to boot or boots and fails on first use is not asserted."""
    with scratch_database("dec1_deploy") as database, provider() as upstream:
        config: Final = write_config(tmp_path, deploy_config(upstream.api_base, general_settings=general_settings))
        outcome: Final = boot_outcome(
            gateway, tmp_path / "proxy", {"DATABASE_URL": database, **environment}, config=config
        )
        assert not _table_exists(database), outcome.log[-3000:]
        assert OUT_OF_SYNC in outcome.log, outcome.log[-3000:]


def test_disable_schema_update_false_as_a_string_still_creates_the_schema(gateway: Gateway, tmp_path: Path) -> None:
    """`DISABLE_SCHEMA_UPDATE=false` (a string, as every manifest sets it) on an empty scratch database. The
    proxy must boot, answer readiness 200 `connected`, and `LiteLLM_VerificationToken` must exist: the string is
    parsed as a boolean (`should_update_prisma_schema`, prisma_client.py:1360), not treated as truthy."""
    with scratch_database("dec1_deploy") as database, provider() as upstream:
        config: Final = write_config(tmp_path, deploy_config(upstream.api_base))
        with owned_proxy_process(
            gateway,
            tmp_path / "proxy",
            {"DATABASE_URL": database, "DISABLE_SCHEMA_UPDATE": "false"},
            config=config,
            remove_environment=("DATABASE_URL_READ_REPLICA",),
        ) as owned:
            assert readiness(owned.gateway) == (200, dict(READY))
        assert _table_exists(database)


def test_the_legacy_migration_resolver_prints_its_warning(gateway: Gateway, tmp_path: Path) -> None:
    """The proxy boots with `prisma migrate deploy` and `--use_legacy_migration_resolver`. The log must contain
    `LiteLLM Proxy: Using the legacy (v1) migration resolver.` (proxy_cli.py:1881): operators running two
    versions against one database need to see that the thrash-prone resolver is on."""
    with provider() as upstream:
        config: Final = write_config(tmp_path, deploy_config(upstream.api_base))
        with owned_proxy_process(
            gateway, tmp_path / "proxy", {}, config=config, database_setup=LEGACY_MIGRATE_DEPLOY
        ) as owned:
            assert LEGACY_WARNING in log_text(owned)


def test_use_v2_migration_resolver_false_in_the_environment_selects_the_legacy_resolver(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The proxy boots with `prisma migrate deploy` and `USE_V2_MIGRATION_RESOLVER=false`. The log must contain
    the legacy resolver warning: the environment variable is the documented way back to v1
    (`resolve_v2_migration_resolver`, proxy_cli.py:199)."""
    with provider() as upstream:
        config: Final = write_config(tmp_path, deploy_config(upstream.api_base))
        with owned_proxy_process(
            gateway,
            tmp_path / "proxy",
            {"USE_V2_MIGRATION_RESOLVER": "false"},
            config=config,
            database_setup=MIGRATE_DEPLOY,
        ) as owned:
            assert LEGACY_WARNING in log_text(owned)


def test_the_default_migrate_deploy_boot_does_not_warn_about_the_legacy_resolver(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The proxy boots with `prisma migrate deploy` and no resolver flag or variable. It must serve a chat
    completion and its log must not contain the legacy resolver warning: v2 is the default."""
    with provider() as upstream:
        config: Final = write_config(tmp_path, deploy_config(upstream.api_base))
        with owned_proxy_process(
            gateway,
            tmp_path / "proxy",
            {},
            config=config,
            database_setup=MIGRATE_DEPLOY,
            remove_environment=("USE_V2_MIGRATION_RESOLVER",),
        ) as owned:
            assert chat(owned.gateway).status_code == 200
            assert LEGACY_WARNING not in log_text(owned)


def test_the_deprecated_v2_resolver_flag_is_accepted_and_announced(gateway: Gateway, tmp_path: Path) -> None:
    """The proxy boots with `prisma migrate deploy` and the deprecated `--use_v2_migration_resolver` flag. It
    must still boot and serve, and its log must contain `LiteLLM Proxy: --use_v2_migration_resolver is
    deprecated and has no effect` (proxy_cli.py:1873): old manifests keep working and learn they can drop it."""
    with provider() as upstream:
        config: Final = write_config(tmp_path, deploy_config(upstream.api_base))
        with owned_proxy_process(
            gateway,
            tmp_path / "proxy",
            {},
            config=config,
            database_setup=MIGRATE_DEPLOY,
            extra_arguments=("--use_v2_migration_resolver",),
            remove_environment=("USE_V2_MIGRATION_RESOLVER",),
        ) as owned:
            assert chat(owned.gateway).status_code == 200
            assert V2_DEPRECATED in log_text(owned)
