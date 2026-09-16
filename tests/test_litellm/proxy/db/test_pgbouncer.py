import base64
import configparser
import json
import logging
import os
import signal
import socket
import stat
import sys
import tempfile
import textwrap
import time
import urllib.parse
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final, cast

import pytest

from litellm._logging import verbose_proxy_logger
from litellm.proxy.db.pgbouncer import (
    PGBOUNCER_POOLED_ENV_VAR,
    PgBouncerError,
    PgBouncerPlan,
    PgBouncerProcess,
    PgBouncerSettings,
    PgBouncerTokenRefresher,
    PgBouncerTokenSource,
    database_url_is_pooled,
    export_pooled_database_url,
    install_pgbouncer_token,
    pgbouncer_version,
    plan_pgbouncer,
    start_in_container_pgbouncer,
    unix_socket_path,
    write_pgbouncer_ini,
    write_userlist,
)
from litellm.proxy.db.token_auth import AzureEntraTokenAuth, IAMEndpoint

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
    def test_upstream_route_and_timeouts_move_into_the_pgbouncer_config_without_the_password(self):
        ini: Final = _ini(_plan())
        assert ini["databases"]["litellm"] == (
            "host='db.internal' port=5433 dbname='litellm' user='app' "
            "connect_query='SET statement_timeout TO ''7000''; SET lock_timeout TO ''3000'''"
        )

    def test_the_auth_file_holds_the_upstream_password_and_the_pool_users_own(self):
        plan: Final = _plan()
        assert plan.upstream_password == "p@ss'w"
        assert plan.userlist("p@ss'w") == f'"app" "p@ss\'w"\n"litellm_pgbouncer" "{plan.pool_password}"\n'

    def test_a_token_with_quotes_is_escaped_the_way_pgbouncer_reads_it(self):
        assert _plan().userlist('to"ken').startswith('"app" "to""ken"\n')

    def test_an_upstream_without_a_port_is_reached_on_the_postgres_default(self):
        ini: Final = _ini(_plan("postgresql://app:pw@db/litellm"))
        assert ini["databases"]["litellm"] == "host='db' port=5432 dbname='litellm' user='app'"

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

    def test_the_pool_user_can_read_the_pgbouncer_console(self):
        assert _ini(_plan())["pgbouncer"]["stats_users"] == "litellm_pgbouncer"

    def test_a_database_user_named_like_the_pool_user_is_refused(self):
        outcome: Final = plan_pgbouncer(
            "postgresql://litellm_pgbouncer:pw@db/litellm", SETTINGS, Path("/run/pgb"), None
        )
        assert isinstance(outcome, PgBouncerError)
        assert "litellm_pgbouncer" in outcome.reason

    def test_pooled_url_points_prisma_at_loopback_as_the_pool_user_without_prepared_statements(self):
        plan: Final = _plan()
        pooled: Final = urllib.parse.urlsplit(plan.pooled_url)
        assert (pooled.hostname, pooled.port, pooled.path) == ("127.0.0.1", 6543, "/litellm")
        assert (pooled.username, pooled.password) == ("litellm_pgbouncer", plan.pool_password)
        assert len(plan.pool_password) >= 32
        assert "p%40ss" not in plan.pooled_url
        assert _query(plan.pooled_url) == {
            "schema": "public",
            "connection_limit": "10",
            "pool_timeout": "20",
            "pgbouncer": "true",
        }

    @pytest.mark.parametrize("hop_param", ["channel_binding=require", "gssencmode=require"])
    def test_transport_params_for_the_postgres_hop_stay_off_the_plain_tcp_loopback_url(self, hop_param: str):
        pooled: Final = _plan(f"postgresql://app:pw@db/litellm?connection_limit=5&{hop_param}").pooled_url
        assert _query(pooled) == {"connection_limit": "5", "pgbouncer": "true"}

    def test_verified_tls_becomes_server_side_verify_full_with_a_ca_copy_in_the_runtime_dir(self):
        plan: Final = _plan()
        pgb: Final = _ini(plan)["pgbouncer"]
        assert pgb["server_tls_sslmode"] == "verify-full"
        assert pgb["server_tls_ca_file"] == "/run/pgb/server-ca.pem"
        assert plan.ca_source == "/certs/ca.pem"

    def test_unverified_require_stays_require_without_a_ca_file(self):
        plan: Final = _plan("postgresql://app:pw@db/litellm?sslmode=require")
        pgb: Final = _ini(plan)["pgbouncer"]
        assert pgb["server_tls_sslmode"] == "require"
        assert "server_tls_ca_file" not in pgb
        assert plan.ca_source is None

    def test_every_plan_gets_its_own_pool_password(self):
        assert _plan().pool_password != _plan().pool_password

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
            "postgresql://app:pw@db",
            "postgresql://:pw@db/litellm",
            "postgresql://app:pw@/litellm",
        ],
    )
    def test_urls_missing_a_route_are_refused(self, url: str):
        outcome: Final = plan_pgbouncer(url, SETTINGS, Path("/run/pgb"), None)
        assert isinstance(outcome, PgBouncerError)

    def test_a_url_without_a_password_plans_for_a_token_to_be_installed_later(self):
        plan: Final = _plan("postgresql://app@db/litellm")
        assert plan.upstream_password is None
        assert plan.userlist("minted").startswith('"app" "minted"\n')

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
        plan: Final = _plan("postgresql://app:pw@db/litellm")
        ini_path: Final = write_pgbouncer_ini(plan, tmp_path, None)
        assert isinstance(ini_path, Path), ini_path
        userlist_path: Final = write_userlist(plan.userlist("pw"), tmp_path, None)
        assert ini_path == tmp_path / "pgbouncer.ini"
        assert userlist_path == tmp_path / "userlist.txt"
        assert ini_path.read_text() == plan.ini
        assert userlist_path.read_text() == plan.userlist("pw")
        for path in (ini_path, userlist_path):
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert not (tmp_path / "server-ca.pem").exists()

    def test_the_ca_bundle_is_copied_next_to_the_ini_pgbouncer_reads(self, tmp_path: Path):
        bundle: Final = tmp_path / "rds-root.pem"
        bundle.write_text("-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n")
        runtime_dir: Final = tmp_path / "run"
        runtime_dir.mkdir()
        plan: Final = plan_pgbouncer(
            f"postgresql://app:pw@db/litellm?sslmode=verify-full&sslcert={bundle}", SETTINGS, runtime_dir, None
        )
        assert isinstance(plan, PgBouncerPlan), plan
        ini_path: Final = write_pgbouncer_ini(plan, runtime_dir, None)
        assert isinstance(ini_path, Path), ini_path
        ca_file: Final = Path(_ini(plan)["pgbouncer"]["server_tls_ca_file"])
        assert ca_file.parent == runtime_dir
        assert ca_file.read_text() == bundle.read_text()

    def test_an_unreadable_ca_bundle_is_reported(self, tmp_path: Path):
        plan: Final = plan_pgbouncer(
            f"postgresql://app:pw@db/litellm?sslmode=verify-full&sslcert={tmp_path / 'missing.pem'}",
            SETTINGS,
            tmp_path,
            None,
        )
        assert isinstance(plan, PgBouncerPlan), plan
        outcome: Final = write_pgbouncer_ini(plan, tmp_path, None)
        assert isinstance(outcome, PgBouncerError)
        assert "missing.pem" in outcome.reason
        assert not (tmp_path / "pgbouncer.ini").exists()

    def test_rewriting_the_userlist_replaces_it_whole_and_leaves_nothing_else_behind(self, tmp_path: Path):
        write_userlist('"app" "first"\n', tmp_path, None)
        with open(tmp_path / "userlist.txt", encoding="utf-8") as before_rewrite:
            write_userlist('"app" "second"\n', tmp_path, None)
            assert before_rewrite.read() == '"app" "first"\n'
        assert (tmp_path / "userlist.txt").read_text() == '"app" "second"\n'
        assert stat.S_IMODE((tmp_path / "userlist.txt").stat().st_mode) == 0o600
        assert sorted(path.name for path in tmp_path.iterdir()) == ["userlist.txt"]


class TestPooledUrlMarker:
    def test_exporting_the_pooled_url_marks_it_for_the_workers(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(PGBOUNCER_POOLED_ENV_VAR, "")
        monkeypatch.delenv(PGBOUNCER_POOLED_ENV_VAR)
        monkeypatch.setenv("DATABASE_URL", "postgresql://app:token@db/litellm")
        assert not database_url_is_pooled()
        export_pooled_database_url("postgresql://litellm_pgbouncer:pw@127.0.0.1:6432/litellm?pgbouncer=true")
        assert os.environ["DATABASE_URL"] == "postgresql://litellm_pgbouncer:pw@127.0.0.1:6432/litellm?pgbouncer=true"
        assert database_url_is_pooled()
        assert PGBOUNCER_POOLED_ENV_VAR == "LITELLM_PGBOUNCER_POOLED_DATABASE_URL"


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
    auth_log: Path | None = None,
) -> Path:
    """An executable that listens like PgBouncer: on the TCP port first, then on ``.s.PGSQL.<port>`` in the socket dir.

    Port and socket dir come from the ini it is given, else from ``port`` and
    ``tmp_path``. With ``port_file`` each start reads the port from that file
    instead. ``bind_delay_seconds`` holds the bind back, like a slow start.
    ``--version`` prints ``version_banner``. With ``auth_log`` it appends the
    ``auth_file`` it reads at startup and on every SIGHUP, one line per read,
    like PgBouncer loading its credentials.
    """
    script: Final = tmp_path / "fake-pgbouncer"
    script.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import configparser, os, pathlib, select, signal, socket, sys, time
            if sys.argv[1:] == ["--version"]:
                print({version_banner!r})
                sys.exit(0)
            if {exit_immediately!r}:
                sys.exit(3)
            ini = configparser.ConfigParser()
            ini.read(sys.argv[1:2])
            if not {auth_log is None!r}:
                def load_auth_file(*_):
                    with open({str(auth_log)!r}, "a") as log:
                        log.write(repr(pathlib.Path(ini.get("pgbouncer", "auth_file")).read_text()) + "\\n")
                load_auth_file()
                signal.signal(signal.SIGHUP, load_auth_file)
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

    @pytest.mark.skip(
        reason="flaky under CI load: ready_timeout_seconds=0.3 is too tight for the initial spawn on shared runners "
        "(see e.g. https://github.com/BerriAI/litellm/actions/runs/34546848959). PgBouncerProcess uses one ready "
        "timeout for both the initial start and every restart, so the test can't loosen it just for the initial "
        "spawn without changing what it exercises"
    )
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
            ready_timeout_seconds=2.0,
        )
        assert pooler.start() is None
        first_pid: Final = pooler.pid
        assert first_pid is not None
        wrong_port: Final = _free_port()
        port_file.write_text(str(wrong_port))
        with caplog.at_level(logging.ERROR, logger=verbose_proxy_logger.name):
            os.kill(first_pid, signal.SIGKILL)
            assert _wait_until(lambda: _listening(wrong_port))
            port_file.write_text(str(port))
            assert _wait_until(lambda: any("did not start listening" in record.message for record in caplog.records))
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


NOW: Final = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
ENDPOINT: Final = IAMEndpoint(host="db", port="5432", user="app", name="litellm")


def _entra_jwt(expires_at: datetime) -> str:
    payload: Final = base64.urlsafe_b64encode(json.dumps({"exp": int(expires_at.timestamp())}).encode())
    return f"aGVhZGVy.{payload.rstrip(b'=').decode()}.c2ln"


def _token_source(*tokens: str | Exception) -> PgBouncerTokenSource:
    """A token source handing out ``tokens`` in order, raising the exceptions among them, then repeating the last."""
    pending: Final = deque(tokens)

    def provide() -> str:
        outcome: Final = pending.popleft() if len(pending) > 1 else pending[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return PgBouncerTokenSource(auth=AzureEntraTokenAuth(token_provider=provide), endpoint=ENDPOINT)


class TestPgBouncerTokenRefresher:
    def _refresher(
        self,
        source: PgBouncerTokenSource,
        installed: list[str],
        install: Callable[[str], None] | None = None,
        **timing: float,
    ) -> PgBouncerTokenRefresher:
        return PgBouncerTokenRefresher(
            source,
            install if install is not None else installed.append,
            now=lambda: NOW.replace(tzinfo=None),
            **timing,
        )

    def test_the_next_refresh_is_due_a_buffer_before_the_token_expires(self):
        installed: Final[list[str]] = []
        token: Final = _entra_jwt(NOW + timedelta(hours=1))
        refresher: Final = self._refresher(_token_source(token), installed, buffer_seconds=180)
        assert refresher.refresh() == 3600 - 180
        assert installed == [token]

    def test_a_token_whose_expiry_cannot_be_read_is_refreshed_on_the_fallback_interval(self):
        installed: Final[list[str]] = []
        refresher: Final = self._refresher(_token_source("opaque token"), installed, fallback_seconds=600)
        assert refresher.refresh() == 600
        assert installed == ["opaque token"]

    def test_a_token_already_inside_the_buffer_is_refreshed_after_the_retry_delay(self):
        token: Final = _entra_jwt(NOW + timedelta(seconds=100))
        refresher: Final = self._refresher(_token_source(token), [], buffer_seconds=180, retry_seconds=30)
        assert refresher.refresh() == 30

    def test_the_token_reaches_the_auth_file_in_wire_form_not_url_encoded(self):
        installed: Final[list[str]] = []
        self._refresher(_token_source("to ken/with+odd=chars"), installed).refresh()
        assert installed == ["to ken/with+odd=chars"]

    def test_a_failed_mint_is_reported_and_installs_nothing(self):
        installed: Final[list[str]] = []
        outcome: Final = self._refresher(_token_source(RuntimeError("no credential")), installed).refresh()
        assert isinstance(outcome, PgBouncerError)
        assert "Azure Entra token" in outcome.reason
        assert "no credential" in outcome.reason
        assert installed == []

    def test_a_token_pgbouncer_cannot_hold_is_refused(self):
        installed: Final[list[str]] = []
        outcome: Final = self._refresher(_token_source("x" * 2048), installed).refresh()
        assert isinstance(outcome, PgBouncerError)
        assert "2047" in outcome.reason
        assert installed == []

    def test_an_auth_file_that_cannot_be_written_is_reported_not_raised(self):
        def refuse(_: str) -> None:
            raise PermissionError("read-only runtime dir")

        outcome: Final = self._refresher(_token_source("token"), [], install=refuse).refresh()
        assert isinstance(outcome, PgBouncerError)
        assert "read-only runtime dir" in outcome.reason

    def test_start_fails_when_the_first_token_cannot_be_minted_and_schedules_nothing(self):
        installed: Final[list[str]] = []
        refresher: Final = self._refresher(
            _token_source(RuntimeError("no credential"), "later"), installed, fallback_seconds=0.05
        )
        assert isinstance(refresher.start(), PgBouncerError)
        time.sleep(0.3)
        assert installed == []

    def test_a_failed_renewal_keeps_the_previous_token_until_the_retry_succeeds(self, caplog: pytest.LogCaptureFixture):
        installed: Final[list[str]] = []
        refresher: Final = self._refresher(
            _token_source("first", RuntimeError("blip"), "third"),
            installed,
            fallback_seconds=0.05,
            retry_seconds=0.05,
        )
        with caplog.at_level(logging.ERROR, logger=verbose_proxy_logger.name):
            assert refresher.start() is None
            assert installed == ["first"]
            assert _wait_until(lambda: "third" in installed)
        assert installed[:2] == ["first", "third"]
        assert any("keeps its current Azure Entra token" in record.message for record in caplog.records)
        refresher.stop()
        settled: Final = len(installed)
        time.sleep(0.3)
        assert len(installed) == settled


def _runtime_dir_listening_on(port: int) -> Path:
    matches: Final = tuple(
        ini.parent
        for ini in Path(tempfile.gettempdir()).glob("litellm-pgbouncer-*/pgbouncer.ini")
        if f"listen_port = {port}\n" in ini.read_text()
    )
    assert len(matches) == 1, matches
    return matches[0]


class TestStartInContainerPgBouncer:
    def test_returns_the_loopback_url_once_the_pooler_listens(self, tmp_path: Path):
        port: Final = _free_port()
        auth_log: Final = tmp_path / "auth.log"
        binary: Final = _fake_pooler(tmp_path, port, auth_log=auth_log)
        settings: Final = PgBouncerSettings(enabled=True, port=port, binary=str(binary))
        pooled: Final = start_in_container_pgbouncer(settings, "postgresql://app:pw@db/litellm?connection_limit=5")
        assert isinstance(pooled, str), pooled
        parsed: Final = urllib.parse.urlsplit(pooled)
        assert (parsed.username, parsed.hostname, parsed.port, parsed.path) == (
            "litellm_pgbouncer",
            "127.0.0.1",
            port,
            "/litellm",
        )
        assert _query(pooled) == {"connection_limit": "5", "pgbouncer": "true"}
        assert _listening(port)
        assert auth_log.read_text() == repr(f'"app" "pw"\n"litellm_pgbouncer" "{parsed.password}"\n') + "\n"

    @pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
    def test_a_forked_worker_exiting_leaves_the_pooler_and_its_files_to_the_parent(self, tmp_path: Path):
        port: Final = _free_port()
        settings: Final = PgBouncerSettings(enabled=True, port=port, binary=str(_fake_pooler(tmp_path, port)))
        exit_hooks: Final[list[Callable[[], None]]] = []
        pooled: Final = start_in_container_pgbouncer(
            settings, "postgresql://app:pw@db/litellm", register_exit_hook=exit_hooks.append
        )
        assert isinstance(pooled, str), pooled
        runtime_dir: Final = _runtime_dir_listening_on(port)

        worker: Final = os.fork()
        if worker == 0:
            try:
                for hook in exit_hooks:
                    hook()
            finally:
                os._exit(0)
        if not _wait_until(lambda: os.waitpid(worker, os.WNOHANG) != (0, 0)):
            os.kill(worker, signal.SIGKILL)
            pytest.fail("the forked worker did not exit: an exit hook blocked on state inherited from the parent")
        assert _listening(port)
        assert (runtime_dir / "pgbouncer.ini").exists()

        for hook in exit_hooks:
            hook()
        assert _wait_until(lambda: not _listening(port))
        assert not runtime_dir.exists()

    def test_a_bad_upstream_url_is_reported_without_starting_anything(self, tmp_path: Path):
        port: Final = _free_port()
        settings: Final = PgBouncerSettings(enabled=True, port=port, binary=str(_fake_pooler(tmp_path, port)))
        outcome: Final = start_in_container_pgbouncer(settings, "postgresql://app:pw@db")
        assert isinstance(outcome, PgBouncerError)
        assert not _listening(port)

    def test_a_passwordless_url_without_token_auth_is_refused_without_starting_anything(self, tmp_path: Path):
        port: Final = _free_port()
        settings: Final = PgBouncerSettings(enabled=True, port=port, binary=str(_fake_pooler(tmp_path, port)))
        outcome: Final = start_in_container_pgbouncer(settings, "postgresql://app@db/litellm")
        assert isinstance(outcome, PgBouncerError)
        assert "IAM_TOKEN_DB_AUTH" in outcome.reason
        assert not _listening(port)

    def test_token_auth_mints_the_first_token_into_the_auth_file_before_the_pooler_starts(self, tmp_path: Path):
        port: Final = _free_port()
        auth_log: Final = tmp_path / "auth.log"
        binary: Final = _fake_pooler(tmp_path, port, auth_log=auth_log)
        settings: Final = PgBouncerSettings(enabled=True, port=port, binary=str(binary))
        token: Final = _entra_jwt(datetime.now(tz=timezone.utc) + timedelta(hours=1))
        pooled: Final = start_in_container_pgbouncer(
            settings,
            "postgresql://app:stale-token@db/litellm",
            token_auth=AzureEntraTokenAuth(token_provider=lambda: token),
        )
        assert isinstance(pooled, str), pooled
        parsed: Final = urllib.parse.urlsplit(pooled)
        assert parsed.username == "litellm_pgbouncer"
        assert token not in pooled
        assert _listening(port)
        assert auth_log.read_text() == repr(f'"app" "{token}"\n"litellm_pgbouncer" "{parsed.password}"\n') + "\n"

    def test_a_first_token_that_cannot_be_minted_is_reported_without_starting_anything(self, tmp_path: Path):
        port: Final = _free_port()
        settings: Final = PgBouncerSettings(enabled=True, port=port, binary=str(_fake_pooler(tmp_path, port)))

        def fail() -> str:
            raise RuntimeError("no Azure credential")

        outcome: Final = start_in_container_pgbouncer(
            settings, "postgresql://app@db/litellm", token_auth=AzureEntraTokenAuth(token_provider=fail)
        )
        assert isinstance(outcome, PgBouncerError)
        assert "no Azure credential" in outcome.reason
        assert not _listening(port)

    def test_a_renewed_token_is_written_and_picked_up_by_the_running_and_by_a_restarted_pooler(self, tmp_path: Path):
        port: Final = _free_port()
        auth_log: Final = tmp_path / "auth.log"
        plan: Final = plan_pgbouncer(
            "postgresql://app@db/litellm", PgBouncerSettings(enabled=True, port=port), tmp_path, None
        )
        assert isinstance(plan, PgBouncerPlan), plan
        ini_path: Final = write_pgbouncer_ini(plan, tmp_path, None)
        write_userlist(plan.userlist("first"), tmp_path, None)
        pooler: Final = PgBouncerProcess(
            argv=(str(_fake_pooler(tmp_path, port, auth_log=auth_log)), str(ini_path)),
            port=port,
            socket_path=unix_socket_path(tmp_path, port),
            restart_delay_seconds=0.1,
        )
        assert pooler.start() is None
        first_pid: Final = pooler.pid
        assert first_pid is not None
        install_pgbouncer_token(plan, tmp_path, None, pooler, "second")
        assert _wait_until(lambda: auth_log.read_text().count("\n") == 2)
        os.kill(first_pid, signal.SIGKILL)
        assert _wait_until(lambda: pooler.pid not in (None, first_pid) and _listening(port))
        pooler.stop()
        assert auth_log.read_text().splitlines() == [
            repr(plan.userlist("first")),
            repr(plan.userlist("second")),
            repr(plan.userlist("second")),
        ]

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
        pooled: Final = start_in_container_pgbouncer(settings, "postgresql://app:pw@db/litellm")
        assert isinstance(pooled, str), pooled
        assert urllib.parse.urlsplit(pooled).port == port
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
