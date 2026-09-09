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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final, cast

import pytest

from litellm._logging import verbose_proxy_logger
from litellm.proxy.db.pgbouncer import (
    PgBouncerError,
    PgBouncerPlan,
    PgBouncerProcess,
    PgBouncerSettings,
    pgbouncer_version,
    plan_pgbouncer,
    start_in_container_pgbouncer,
    unix_socket_path,
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


def _bound_port(sock: socket.socket) -> int:
    return cast(tuple[str, int], sock.getsockname())[1]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return _bound_port(probe)


def _fake_pooler(
    tmp_path: Path,
    port: int,
    exit_immediately: bool = False,
    port_file: Path | None = None,
    bind_delay_seconds: float = 0.0,
    version_banner: str = "PgBouncer 1.25.2\nlibevent 2.1.13-stable",
) -> Path:
    """An executable that listens like PgBouncer: on the TCP port first, then on ``.s.PGSQL.<port>`` in the socket dir.

    Port and socket dir come from the ini it is given, else from ``port`` and
    ``tmp_path``. With ``port_file`` each start reads the port from that file
    instead. ``bind_delay_seconds`` holds the bind back, like a slow start.
    ``--version`` prints ``version_banner``.
    """
    script: Final = tmp_path / "fake-pgbouncer"
    script.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import configparser, os, pathlib, select, socket, sys, time
            if sys.argv[1:] == ["--version"]:
                print({version_banner!r})
                sys.exit(0)
            if {exit_immediately!r}:
                sys.exit(3)
            ini = configparser.ConfigParser()
            ini.read(sys.argv[1:2])
            port = ini.getint("pgbouncer", "listen_port", fallback={port})
            if not {port_file is None!r}:
                port = int(pathlib.Path({str(port_file)!r}).read_text())
            socket_dir = ini.get("pgbouncer", "unix_socket_dir", fallback={str(tmp_path)!r})
            socket_path = f"{{socket_dir}}/.s.PGSQL.{{port}}"
            time.sleep({bind_delay_seconds!r})
            listener = socket.socket()
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", port))
            listener.listen()
            if os.path.exists(socket_path):
                os.unlink(socket_path)
            unix_listener = socket.socket(socket.AF_UNIX)
            unix_listener.bind(socket_path)
            unix_listener.listen()
            while True:
                for ready in select.select([listener, unix_listener], [], [])[0]:
                    conn, _ = ready.accept()
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
        pooler: Final = PgBouncerProcess(
            argv=(str(_fake_pooler(tmp_path, port)),), port=port, socket_path=unix_socket_path(tmp_path, port)
        )
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
            argv=(str(_fake_pooler(tmp_path, port)),),
            port=port,
            socket_path=unix_socket_path(tmp_path, port),
            restart_delay_seconds=0.1,
        )
        assert pooler.start() is None
        first_pid: Final = pooler.pid
        assert first_pid is not None
        os.kill(first_pid, signal.SIGKILL)
        assert _wait_until(lambda: pooler.pid not in (None, first_pid) and _listening(port))
        pooler.stop()
        assert _wait_until(lambda: not _listening(port))

    def test_a_failed_restart_is_retried_until_the_pooler_is_back(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        port: Final = _free_port()
        script: Final = _fake_pooler(tmp_path, port)
        pooler: Final = PgBouncerProcess(
            argv=(str(script),), port=port, socket_path=unix_socket_path(tmp_path, port), restart_delay_seconds=0.1
        )
        assert pooler.start() is None
        first_pid: Final = pooler.pid
        assert first_pid is not None
        hidden: Final = script.rename(tmp_path / "hidden")
        with caplog.at_level(logging.ERROR, logger=verbose_proxy_logger.name):
            os.kill(first_pid, signal.SIGKILL)
            assert _wait_until(lambda: any("could not be restarted" in record.message for record in caplog.records))
            assert not _listening(port)
            hidden.rename(script)
            assert _wait_until(lambda: pooler.pid not in (None, first_pid) and _listening(port))
        pooler.stop()
        assert _wait_until(lambda: not _listening(port))

    def test_a_replacement_that_never_listens_is_replaced_again(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        port: Final = _free_port()
        port_file: Final = tmp_path / "port"
        port_file.write_text(str(port))
        script: Final = _fake_pooler(tmp_path, port, port_file=port_file)
        pooler: Final = PgBouncerProcess(
            argv=(str(script),),
            port=port,
            socket_path=unix_socket_path(tmp_path, port),
            restart_delay_seconds=0.1,
            ready_timeout_seconds=0.3,
        )
        assert pooler.start() is None
        first_pid: Final = pooler.pid
        assert first_pid is not None
        wrong_port: Final = _free_port()
        port_file.write_text(str(wrong_port))
        with caplog.at_level(logging.ERROR, logger=verbose_proxy_logger.name):
            os.kill(first_pid, signal.SIGKILL)
            assert _wait_until(lambda: _listening(wrong_port))
            assert _wait_until(lambda: any("did not start listening" in record.message for record in caplog.records))
            port_file.write_text(str(port))
            assert _wait_until(lambda: _listening(port))
            assert _wait_until(lambda: not _listening(wrong_port))
        pooler.stop()
        assert _wait_until(lambda: not _listening(port))

    def test_stopping_during_the_restart_delay_leaves_no_pooler_behind(self, tmp_path: Path):
        port: Final = _free_port()
        pooler: Final = PgBouncerProcess(
            argv=(str(_fake_pooler(tmp_path, port)),),
            port=port,
            socket_path=unix_socket_path(tmp_path, port),
            restart_delay_seconds=0.3,
        )
        assert pooler.start() is None
        first_pid: Final = pooler.pid
        assert first_pid is not None
        os.kill(first_pid, signal.SIGKILL)
        assert _wait_until(lambda: not _listening(port))
        pooler.stop()
        time.sleep(1.0)
        assert not _listening(port)
        assert pooler.pid == first_pid

    def test_a_stopped_pooler_is_not_restarted(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        port: Final = _free_port()
        pooler: Final = PgBouncerProcess(
            argv=(str(_fake_pooler(tmp_path, port)),),
            port=port,
            socket_path=unix_socket_path(tmp_path, port),
            restart_delay_seconds=0.1,
        )
        assert pooler.start() is None
        with caplog.at_level(logging.ERROR, logger=verbose_proxy_logger.name):
            pooler.stop()
            time.sleep(0.5)
        assert not _listening(port)
        assert caplog.records == []

    def test_a_pooler_that_exits_during_startup_is_reported(self, tmp_path: Path):
        port: Final = _free_port()
        pooler: Final = PgBouncerProcess(
            argv=(str(_fake_pooler(tmp_path, port, exit_immediately=True)),),
            port=port,
            socket_path=unix_socket_path(tmp_path, port),
        )
        outcome: Final = pooler.start()
        assert isinstance(outcome, PgBouncerError)
        assert "status 3" in outcome.reason

    def test_a_missing_binary_is_reported(self, tmp_path: Path):
        outcome: Final = PgBouncerProcess(
            argv=("/nonexistent/pgbouncer",), port=_free_port(), socket_path=tmp_path / "sock"
        ).start()
        assert isinstance(outcome, PgBouncerError)
        assert "/nonexistent/pgbouncer" in outcome.reason

    def test_a_port_owned_by_someone_else_is_refused_before_spawning(self, tmp_path: Path):
        with socket.socket() as squatter:
            squatter.bind(("127.0.0.1", 0))
            squatter.listen()
            port: Final = _bound_port(squatter)
            pooler: Final = PgBouncerProcess(
                argv=(str(_fake_pooler(tmp_path, port)),), port=port, socket_path=unix_socket_path(tmp_path, port)
            )
            outcome: Final = pooler.start()
        assert isinstance(outcome, PgBouncerError)
        assert f"127.0.0.1:{port} is already in use" in outcome.reason
        assert pooler.pid is None

    def test_a_replacement_waits_until_a_squatter_leaves_the_port(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        port: Final = _free_port()
        pooler: Final = PgBouncerProcess(
            argv=(str(_fake_pooler(tmp_path, port)),),
            port=port,
            socket_path=unix_socket_path(tmp_path, port),
            restart_delay_seconds=0.5,
        )
        assert pooler.start() is None
        first_pid: Final = pooler.pid
        assert first_pid is not None
        os.kill(first_pid, signal.SIGKILL)
        assert _wait_until(lambda: not _listening(port))
        with socket.socket() as squatter, caplog.at_level(logging.ERROR, logger=verbose_proxy_logger.name):
            squatter.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            squatter.bind(("127.0.0.1", port))
            squatter.listen()
            assert _wait_until(lambda: any("already in use" in record.message for record in caplog.records))
            assert pooler.pid == first_pid
        assert _wait_until(lambda: pooler.pid not in (None, first_pid) and _listening(port))
        pooler.stop()
        assert _wait_until(lambda: not _listening(port))

    def test_a_listener_that_grabs_the_port_after_the_spawn_is_not_taken_for_the_pooler(self, tmp_path: Path):
        port: Final = _free_port()
        pooler: Final = PgBouncerProcess(
            argv=(str(_fake_pooler(tmp_path, port, bind_delay_seconds=0.5)),),
            port=port,
            socket_path=unix_socket_path(tmp_path, port),
            ready_timeout_seconds=3.0,
        )
        with socket.socket() as squatter, ThreadPoolExecutor(max_workers=1) as starter:
            squatter.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            starting: Final = starter.submit(pooler.start)
            assert _wait_until(lambda: pooler.pid is not None)
            squatter.bind(("127.0.0.1", port))
            squatter.listen()
            outcome: Final = starting.result()
        assert isinstance(outcome, PgBouncerError)
        assert "exited with status 1" in outcome.reason

    def test_a_port_served_by_a_stranger_while_the_pooler_is_still_starting_is_reported(self, tmp_path: Path):
        port: Final = _free_port()
        pooler: Final = PgBouncerProcess(
            argv=(str(_fake_pooler(tmp_path, port, bind_delay_seconds=30.0)),),
            port=port,
            socket_path=unix_socket_path(tmp_path, port),
            ready_timeout_seconds=0.5,
        )
        with socket.socket() as squatter, ThreadPoolExecutor(max_workers=1) as starter:
            squatter.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            starting: Final = starter.submit(pooler.start)
            assert _wait_until(lambda: pooler.pid is not None)
            squatter.bind(("127.0.0.1", port))
            squatter.listen()
            outcome: Final = starting.result()
        assert isinstance(outcome, PgBouncerError)
        assert f"127.0.0.1:{port} is served by another process" in outcome.reason
        pid: Final = pooler.pid
        assert pid is not None
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)

    def test_a_pooler_that_never_listens_times_out(self, tmp_path: Path):
        port: Final = _free_port()
        pooler: Final = PgBouncerProcess(
            argv=(str(_fake_pooler(tmp_path, _free_port())),),
            port=port,
            socket_path=unix_socket_path(tmp_path, port),
            ready_timeout_seconds=0.5,
        )
        outcome: Final = pooler.start()
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

    def test_token_auth_is_refused_without_starting_anything(self, tmp_path: Path):
        port: Final = _free_port()
        settings: Final = PgBouncerSettings(enabled=True, port=port, binary=str(_fake_pooler(tmp_path, port)))
        outcome: Final = start_in_container_pgbouncer(
            settings, "postgresql://app:pw@db/litellm", token_auth_enabled=True
        )
        assert isinstance(outcome, PgBouncerError)
        assert "IAM_TOKEN_DB_AUTH" in outcome.reason
        assert "AZURE_POSTGRESQL_AUTH" in outcome.reason
        assert not _listening(port)

    def test_a_pgbouncer_that_survives_a_failed_tcp_bind_is_refused_without_starting(self, tmp_path: Path):
        port: Final = _free_port()
        binary: Final = _fake_pooler(tmp_path, port, version_banner="PgBouncer 1.18.1\nlibevent 2.1.12-stable")
        settings: Final = PgBouncerSettings(enabled=True, port=port, binary=str(binary))
        outcome: Final = start_in_container_pgbouncer(settings, "postgresql://app:pw@db/litellm")
        assert isinstance(outcome, PgBouncerError)
        assert "PgBouncer 1.18" in outcome.reason
        assert "1.19" in outcome.reason
        assert not _listening(port)

    def test_the_first_version_that_dies_on_a_failed_tcp_bind_is_accepted(self, tmp_path: Path):
        port: Final = _free_port()
        binary: Final = _fake_pooler(tmp_path, port, version_banner="PgBouncer 1.19.0")
        settings: Final = PgBouncerSettings(enabled=True, port=port, binary=str(binary))
        assert start_in_container_pgbouncer(settings, "postgresql://app:pw@db/litellm") == (
            f"postgresql://app:pw@127.0.0.1:{port}/litellm?pgbouncer=true"
        )
        assert _listening(port)


class TestPgBouncerVersion:
    def test_reads_major_and_minor_from_the_banner(self, tmp_path: Path):
        assert pgbouncer_version(str(_fake_pooler(tmp_path, _free_port()))) == (1, 25)

    def test_a_binary_that_cannot_run_is_reported(self, tmp_path: Path):
        outcome: Final = pgbouncer_version(str(tmp_path / "missing-pgbouncer"))
        assert isinstance(outcome, PgBouncerError)
        assert "missing-pgbouncer" in outcome.reason

    def test_a_banner_without_a_version_is_reported(self, tmp_path: Path):
        outcome: Final = pgbouncer_version(str(_fake_pooler(tmp_path, _free_port(), version_banner="something else")))
        assert isinstance(outcome, PgBouncerError)
        assert "something else" in outcome.reason


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
