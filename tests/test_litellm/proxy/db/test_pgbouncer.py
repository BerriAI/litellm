import configparser
import logging
import os
import signal
import socket
import stat
import sys
import textwrap
import time
import urllib.parse
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pytest

from litellm._logging import verbose_proxy_logger
from litellm.proxy.db.pgbouncer import (
    PgBouncerError,
    PgBouncerPlan,
    PgBouncerProcess,
    PgBouncerSettings,
    plan_pgbouncer,
    start_in_container_pgbouncer,
    write_pgbouncer_files,
)

UPSTREAM: Final = (
    "postgresql://app:p%40ss%27w@db.internal:5433/litellm"
    "?schema=public&connection_limit=10&pool_timeout=20"
    "&sslmode=require&sslaccept=strict&sslcert=/certs/ca.pem"
    "&options=-c%20statement_timeout%3D7000%20-c%20lock_timeout%3D3000"
)
SETTINGS: Final = PgBouncerSettings(enabled=True, port=6543, max_db_connections=8, max_client_conn=400)


def _plan(url: str = UPSTREAM, run_as_user: str | None = None) -> PgBouncerPlan:
    plan: Final = plan_pgbouncer(url, SETTINGS, Path("/run/pgb"), run_as_user)
    assert isinstance(plan, PgBouncerPlan), plan
    return plan


def _ini(plan: PgBouncerPlan) -> configparser.ConfigParser:
    parser: Final = configparser.ConfigParser(interpolation=None)
    parser.read_string(plan.ini)
    return parser


def _query(url: str) -> dict[str, str]:
    return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query, keep_blank_values=True))


class TestPlanPgBouncer:
    def test_upstream_credentials_and_timeouts_move_into_the_pgbouncer_config(self):
        ini: Final = _ini(_plan())
        assert ini["databases"]["litellm"] == (
            "host='db.internal' port=5433 dbname='litellm' user='app' password='p@ss''w' "
            "connect_query='SET statement_timeout TO ''7000''; SET lock_timeout TO ''3000'''"
        )
        assert _plan().userlist == '"app" "p@ss\'w"\n'

    def test_an_upstream_without_a_port_is_reached_on_the_postgres_default(self):
        ini: Final = _ini(_plan("postgresql://app:pw@db/litellm"))
        assert ini["databases"]["litellm"] == "host='db' port=5432 dbname='litellm' user='app' password='pw'"

    def test_pool_is_sized_from_settings_in_transaction_mode(self):
        pgb: Final = _ini(_plan())["pgbouncer"]
        assert pgb["pool_mode"] == "transaction"
        assert pgb["max_db_connections"] == "8"
        assert pgb["default_pool_size"] == "8"
        assert pgb["max_client_conn"] == "400"
        assert pgb["auth_type"] == "scram-sha-256"
        assert pgb["listen_addr"] == "127.0.0.1"
        assert pgb["listen_port"] == "6543"
        assert pgb["auth_file"] == "/run/pgb/userlist.txt"
        assert pgb["unix_socket_dir"] == "/run/pgb"

    def test_pooled_url_points_prisma_at_loopback_without_prepared_statements(self):
        pooled: Final = urllib.parse.urlsplit(_plan().pooled_url)
        assert (pooled.hostname, pooled.port, pooled.path) == ("127.0.0.1", 6543, "/litellm")
        assert (pooled.username, pooled.password) == ("app", "p%40ss%27w")
        assert _query(_plan().pooled_url) == {
            "schema": "public",
            "connection_limit": "10",
            "pool_timeout": "20",
            "pgbouncer": "true",
        }

    def test_verified_tls_becomes_server_side_verify_full_with_the_ca_bundle(self):
        pgb: Final = _ini(_plan())["pgbouncer"]
        assert pgb["server_tls_sslmode"] == "verify-full"
        assert pgb["server_tls_ca_file"] == "/certs/ca.pem"

    def test_unverified_require_stays_require_without_a_ca_file(self):
        pgb: Final = _ini(_plan("postgresql://app:pw@db/litellm?sslmode=require"))["pgbouncer"]
        assert pgb["server_tls_sslmode"] == "require"
        assert "server_tls_ca_file" not in pgb

    def test_no_tls_params_default_to_prefer(self):
        assert _ini(_plan("postgresql://app:pw@db/litellm"))["pgbouncer"]["server_tls_sslmode"] == "prefer"

    def test_verification_without_a_ca_bundle_is_refused(self):
        outcome: Final = plan_pgbouncer(
            "postgresql://app:pw@db/litellm?sslmode=require&sslaccept=strict", SETTINGS, Path("/run/pgb"), None
        )
        assert isinstance(outcome, PgBouncerError)
        assert "sslcert" in outcome.reason

    def test_client_certificates_are_refused(self):
        outcome: Final = plan_pgbouncer(
            "postgresql://app:pw@db/litellm?sslidentity=/certs/client.p12", SETTINGS, Path("/run/pgb"), None
        )
        assert isinstance(outcome, PgBouncerError)
        assert "sslidentity" in outcome.reason

    @pytest.mark.parametrize(
        "url",
        [
            "postgresql://app@db/litellm",
            "postgresql://app:pw@db",
            "postgresql://:pw@db/litellm",
        ],
    )
    def test_urls_missing_forwardable_credentials_are_refused(self, url: str):
        outcome: Final = plan_pgbouncer(url, SETTINGS, Path("/run/pgb"), None)
        assert isinstance(outcome, PgBouncerError)

    def test_every_options_spelling_becomes_a_set_statement(self):
        options: Final = urllib.parse.quote("-c a=1 -cb=2 --c=3")
        ini: Final = _ini(_plan(f"postgresql://app:pw@db/litellm?options={options}"))
        assert ini["databases"]["litellm"].endswith("connect_query='SET a TO ''1''; SET b TO ''2''; SET c TO ''3'''")

    def test_options_that_are_not_settings_are_refused(self):
        outcome: Final = plan_pgbouncer(
            "postgresql://app:pw@db/litellm?options=-c%20search_path", SETTINGS, Path("/run/pgb"), None
        )
        assert isinstance(outcome, PgBouncerError)
        assert "options" in outcome.reason

    def test_run_as_user_is_only_written_when_given(self):
        assert _ini(_plan(run_as_user="nobody"))["pgbouncer"]["user"] == "nobody"
        assert "user" not in _ini(_plan())["pgbouncer"]


class TestWritePgBouncerFiles:
    def test_files_hold_the_plan_and_are_private_to_the_owner(self, tmp_path: Path):
        ini_path: Final = write_pgbouncer_files(_plan(), tmp_path, None)
        assert ini_path == tmp_path / "pgbouncer.ini"
        assert ini_path.read_text() == _plan().ini
        assert (tmp_path / "userlist.txt").read_text() == _plan().userlist
        for path in (ini_path, tmp_path / "userlist.txt"):
            assert stat.S_IMODE(path.stat().st_mode) == 0o600


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _fake_pooler(tmp_path: Path, port: int, exit_immediately: bool = False) -> Path:
    """An executable that listens on ``port`` like PgBouncer would (or exits at once), ignoring its ini argument."""
    script: Final = tmp_path / "fake-pgbouncer"
    script.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import socket, sys, time
            if {exit_immediately!r}:
                sys.exit(3)
            listener = socket.socket()
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", {port}))
            listener.listen()
            while True:
                conn, _ = listener.accept()
                conn.close()
            """
        )
    )
    script.chmod(0o700)
    return script


def _listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _wait_until(condition: Callable[[], bool], timeout_seconds: float = 5.0) -> bool:
    deadline: Final = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


class TestPgBouncerProcess:
    def test_start_waits_for_the_listener_and_stop_ends_it(self, tmp_path: Path):
        port: Final = _free_port()
        pooler: Final = PgBouncerProcess(argv=(str(_fake_pooler(tmp_path, port)),), port=port)
        assert pooler.start() is None
        assert _listening(port)
        pid: Final = pooler.pid
        assert pid is not None
        pooler.stop()
        assert _wait_until(lambda: not _listening(port))
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)

    def test_a_crashed_pooler_is_restarted_with_a_new_pid(self, tmp_path: Path):
        port: Final = _free_port()
        pooler: Final = PgBouncerProcess(
            argv=(str(_fake_pooler(tmp_path, port)),), port=port, restart_delay_seconds=0.1
        )
        assert pooler.start() is None
        first_pid: Final = pooler.pid
        assert first_pid is not None
        os.kill(first_pid, signal.SIGKILL)
        assert _wait_until(lambda: pooler.pid not in (None, first_pid) and _listening(port))
        pooler.stop()
        assert _wait_until(lambda: not _listening(port))

    def test_a_stopped_pooler_is_not_restarted(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        port: Final = _free_port()
        pooler: Final = PgBouncerProcess(
            argv=(str(_fake_pooler(tmp_path, port)),), port=port, restart_delay_seconds=0.1
        )
        assert pooler.start() is None
        with caplog.at_level(logging.ERROR, logger=verbose_proxy_logger.name):
            pooler.stop()
            time.sleep(0.5)
        assert not _listening(port)
        assert caplog.records == []

    def test_a_pooler_that_exits_during_startup_is_reported(self, tmp_path: Path):
        port: Final = _free_port()
        pooler: Final = PgBouncerProcess(argv=(str(_fake_pooler(tmp_path, port, exit_immediately=True)),), port=port)
        outcome: Final = pooler.start()
        assert isinstance(outcome, PgBouncerError)
        assert "status 3" in outcome.reason

    def test_a_missing_binary_is_reported(self):
        outcome: Final = PgBouncerProcess(argv=("/nonexistent/pgbouncer",), port=_free_port()).start()
        assert isinstance(outcome, PgBouncerError)
        assert "/nonexistent/pgbouncer" in outcome.reason

    def test_a_pooler_that_never_listens_times_out(self, tmp_path: Path):
        port: Final = _free_port()
        pooler: Final = PgBouncerProcess(argv=(str(_fake_pooler(tmp_path, _free_port())),), port=port)
        outcome: Final = pooler.start(ready_timeout_seconds=0.5)
        assert isinstance(outcome, PgBouncerError)
        assert "did not start listening" in outcome.reason
        pid: Final = pooler.pid
        assert pid is not None
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


class TestStartInContainerPgBouncer:
    def test_returns_the_loopback_url_once_the_pooler_listens(self, tmp_path: Path):
        port: Final = _free_port()
        settings: Final = PgBouncerSettings(enabled=True, port=port, binary=str(_fake_pooler(tmp_path, port)))
        pooled: Final = start_in_container_pgbouncer(settings, "postgresql://app:pw@db/litellm?connection_limit=5")
        assert pooled == f"postgresql://app:pw@127.0.0.1:{port}/litellm?connection_limit=5&pgbouncer=true"
        assert _listening(port)

    def test_a_bad_upstream_url_is_reported_without_starting_anything(self, tmp_path: Path):
        port: Final = _free_port()
        settings: Final = PgBouncerSettings(enabled=True, port=port, binary=str(_fake_pooler(tmp_path, port)))
        outcome: Final = start_in_container_pgbouncer(settings, "postgresql://app@db/litellm")
        assert isinstance(outcome, PgBouncerError)
        assert not _listening(port)


class TestPgBouncerSettings:
    def test_reads_the_litellm_pgbouncer_env_vars(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LITELLM_PGBOUNCER_ENABLED", "true")
        monkeypatch.setenv("LITELLM_PGBOUNCER_PORT", "7000")
        monkeypatch.setenv("LITELLM_PGBOUNCER_MAX_DB_CONNECTIONS", "12")
        settings: Final = PgBouncerSettings()
        assert (settings.enabled, settings.port, settings.max_db_connections) == (True, 7000, 12)

    def test_defaults_are_off(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LITELLM_PGBOUNCER_ENABLED", raising=False)
        assert PgBouncerSettings().enabled is False
