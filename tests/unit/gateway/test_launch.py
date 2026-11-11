import gc
import importlib
import os
import socket
import subprocess
import sys
import textwrap
import urllib.parse
from collections.abc import Iterator
from pathlib import Path
from typing import Final, cast
from unittest.mock import MagicMock, patch

import pytest
from uvicorn.importer import import_from_string
from uvicorn.main import main as uvicorn_main

from gateway.launch import GATEWAY_APP, main, pool_database_url, python_server, uvicorn_argv
from litellm.proxy.db.db_url_settings import DatabaseURLSettings
from litellm.proxy.db.pgbouncer import PGBOUNCER_POOLED_ENV_VAR, PgBouncerError, PgBouncerSettings
from litellm.proxy.proxy_server import app as proxy_app
from litellm.proxy.rust_forward.mode import (
    Mode,
    RustForwardMode,
    Supervised,
    uvicorn_flags,
)
from litellm.proxy.rust_forward.server import NO_TRUSTED_PROXIES, Listener
from litellm.proxy.server_listener import AssignedPort, InheritedFd
from litellm.proxy.shutdown.graceful_shutdown_manager import GracefulShutdownManager

DB_ENV: Final = {
    "DATABASE_HOST": "db.internal",
    "DATABASE_PORT": "5432",
    "DATABASE_USER": "litellm_pool",
    "DATABASE_NAME": "litellm",
    "DATABASE_PASSWORD": "p@ss",
}


def test_a_uvicorn_worker_reimports_the_launcher_without_loading_litellm():
    reimported: Final = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            'import runpy, sys; runpy.run_module("gateway.launch", run_name="__mp_main__", alter_sys=True); '
            'print("litellm" in sys.modules)',
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert reimported.returncode == 0, reimported.stderr
    assert reimported.stdout.strip() == "False"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return cast(tuple[str, int], probe.getsockname())[1]


def _fake_pooler(tmp_path: Path) -> Path:
    script: Final = tmp_path / "fake-pgbouncer"
    script.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import configparser, select, socket, sys
            if sys.argv[1:] == ["--version"]:
                print("PgBouncer 1.25.2")
                sys.exit(0)
            ini = configparser.ConfigParser()
            ini.read(sys.argv[1])
            port = ini.getint("pgbouncer", "listen_port")
            tcp = socket.socket()
            tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            tcp.bind(("127.0.0.1", port))
            tcp.listen()
            unix = socket.socket(socket.AF_UNIX)
            unix.bind(ini.get("pgbouncer", "unix_socket_dir") + f"/.s.PGSQL.{{port}}")
            unix.listen()
            while True:
                for ready in select.select([tcp, unix], [], [])[0]:
                    ready.accept()[0].close()
            """
        )
    )
    script.chmod(0o700)
    return script


def _query(url: str) -> dict[str, str]:
    return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))


@pytest.fixture
def password_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, str]]:
    for var in (
        "DATABASE_URL",
        "IAM_TOKEN_DB_AUTH",
        "AZURE_POSTGRESQL_AUTH",
        "DATABASE_HOST_READ_REPLICA",
        PGBOUNCER_POOLED_ENV_VAR,
    ):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    for var, value in DB_ENV.items():
        monkeypatch.setenv(var, value)
    yield dict(DB_ENV)
    os.environ.pop("DATABASE_URL", None)


def _minted_iam_token(token: str):
    rds: Final = MagicMock()
    rds.generate_db_auth_token.return_value = token
    return patch("boto3.client", return_value=rds)


def _uvicorn_params(argv: tuple[str, ...]) -> dict[str, object]:
    return uvicorn_main.make_context("uvicorn", list(argv)).params


class TestUvicornArgv:
    @pytest.mark.parametrize(
        ("arguments", "listener"),
        [
            ((), Listener.TCP),
            (("--uds", "/tmp/gateway.sock"), Listener.UNIX_SOCKET),
            (("--fd", "0"), Listener.INHERITED_FD),
            (("--fd=17",), Listener.INHERITED_FD),
            (("--ssl-keyfile", "unused"), Listener.TLS),
            (("--ssl-certfile", "unused"), Listener.TLS),
            (("--ssl-ciphers", "TLSv1.2"), Listener.TCP),
        ],
    )
    def test_forwarding_sees_the_public_listener_uvicorn_would_open(
        self, arguments: tuple[str, ...], listener: Listener
    ) -> None:
        assert python_server(arguments).listener is listener

    @pytest.mark.parametrize(
        ("argv", "environ", "trust"),
        [
            ((), {}, "127.0.0.1"),
            ((), {"FORWARDED_ALLOW_IPS": "10.9.9.9"}, "10.9.9.9"),
            (("--forwarded-allow-ips", "10.0.0.5"), {"FORWARDED_ALLOW_IPS": "10.9.9.9"}, "10.0.0.5"),
            (("--forwarded-allow-ips=*",), {}, "*"),
            (("--no-proxy-headers", "--forwarded-allow-ips=*"), {}, NO_TRUSTED_PROXIES),
        ],
    )
    def test_forwarding_carries_the_trust_uvicorn_resolves_to_the_child(
        self,
        argv: tuple[str, ...],
        environ: dict[str, str],
        trust: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("FORWARDED_ALLOW_IPS", raising=False)
        for name, value in environ.items():
            monkeypatch.setenv(name, value)
        server: Final = python_server(("--host", "0.0.0.0", "--port", "4000", *argv))
        assert (server.host, server.port) == ("0.0.0.0", 4000)
        assert server.trusted_proxies == trust

    @pytest.mark.parametrize(
        ("argv", "environ"),
        [(("--timeout-keep-alive", "2"), {}), ((), {"KEEPALIVE_TIMEOUT": "7"}), ((), {})],
    )
    def test_rust_retires_connections_by_the_keepalive_uvicorn_runs_with(
        self, argv: tuple[str, ...], environ: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KEEPALIVE_TIMEOUT", raising=False)
        for name, value in environ.items():
            monkeypatch.setenv(name, value)
        served: Final = _uvicorn_params(uvicorn_argv(argv, os.environ))["timeout_keep_alive"]
        assert python_server(argv).keepalive_seconds == served

    @pytest.mark.parametrize("argv", [(), ("--timeout-graceful-shutdown", "12")])
    def test_rust_drains_for_pythons_graceful_timeout_plus_teardown(
        self, argv: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GRACEFUL_SHUTDOWN_TIMEOUT", "7.5")
        flag: Final = _uvicorn_params(uvicorn_argv(argv, {}))["timeout_graceful_shutdown"]
        graceful: Final = GracefulShutdownManager.get_timeout() if flag is None else float(str(flag))
        assert python_server(argv).graceful_timeout_seconds == graceful

    @pytest.mark.parametrize(
        ("mode", "served"),
        [
            (Mode.DISABLED, {"host": "0.0.0.0", "port": 4000, "fd": None, "proxy_headers": True}),
            (
                Supervised(InheritedFd(17), "*"),
                {"host": "0.0.0.0", "port": 4000, "fd": 17, "proxy_headers": False},
            ),
            (
                Supervised(AssignedPort(45556), "*"),
                {"host": "127.0.0.1", "port": 45556, "fd": None, "proxy_headers": False},
            ),
        ],
    )
    def test_supervised_uvicorn_serves_the_private_hop_and_never_trusts_it(
        self, mode: RustForwardMode, served: dict[str, object]
    ) -> None:
        argv: Final = ("--host", "0.0.0.0", "--port", "4000", "--workers", "2", "--proxy-headers")
        params: Final = _uvicorn_params((*uvicorn_argv((*argv, "--forwarded-allow-ips=*"), {}), *uvicorn_flags(mode)))
        assert {name: params[name] for name in served} == served
        assert (params["workers"], params["forwarded_allow_ips"]) == (2, "*")

    def test_keepalive_env_reaches_uvicorn(self):
        params: Final = _uvicorn_params(uvicorn_argv(("--workers", "4"), {"KEEPALIVE_TIMEOUT": "75"}))
        assert params["app"] == GATEWAY_APP
        assert params["workers"] == 4
        assert params["timeout_keep_alive"] == 75

    def test_unset_env_keeps_the_uvicorn_default(self):
        assert _uvicorn_params(uvicorn_argv(("--workers", "4"), {}))["timeout_keep_alive"] == 5

    def test_an_explicit_flag_wins_over_the_env(self):
        argv: Final = uvicorn_argv(("--timeout-keep-alive", "30"), {"KEEPALIVE_TIMEOUT": "75"})
        assert _uvicorn_params(argv)["timeout_keep_alive"] == 30

    def test_the_app_uvicorn_is_told_to_serve_is_the_trimmed_gateway(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(proxy_app.router, "lifespan_context", proxy_app.router.lifespan_context)
        for key in (
            "DATABASE_URL",
            "DIRECT_URL",
            "DATABASE_URL_READ_REPLICA",
            "DATABASE_HOST",
            "DATABASE_HOST_READ_REPLICA",
            "DATABASE_PASSWORD",
            "IAM_TOKEN_DB_AUTH",
            "AZURE_POSTGRESQL_AUTH",
        ):
            monkeypatch.delenv(key, raising=False)

        params: Final = _uvicorn_params(uvicorn_argv((), {}))
        assert params["factory"] is True
        try:
            served: Final = import_from_string(cast(str, params["app"]))()
        finally:
            gc.unfreeze()
        assert served is importlib.import_module("gateway.main").app


class TestPoolDatabaseUrl:
    def test_a_disabled_pooler_yields_no_url_to_install(self, password_env: dict[str, str]):
        settings: Final = DatabaseURLSettings.from_env()
        settings.apply_to_env()
        environ: Final = {"DATABASE_URL": "postgresql://litellm_pool:p%40ss@db.internal:5432/litellm"}
        assert pool_database_url(settings, PgBouncerSettings(enabled=False), environ) is None

    def test_a_missing_upstream_url_is_reported(self, password_env: dict[str, str]):
        environ: Final[dict[str, str]] = {}
        outcome: Final = pool_database_url(DatabaseURLSettings.from_env(), PgBouncerSettings(enabled=True), environ)
        assert isinstance(outcome, PgBouncerError)
        assert "DATABASE_URL" in outcome.reason

    def test_token_auth_hands_the_workers_the_pool_user_not_the_token(
        self, password_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, socket_tempdir: Path
    ):
        monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
        monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
        port: Final = _free_port()
        environ: Final = {"DATABASE_URL": "postgresql://litellm:MINTED_TOKEN@db.internal:5432/litellm"}
        with _minted_iam_token("MINTED_TOKEN"):
            outcome: Final = pool_database_url(
                DatabaseURLSettings.from_env(),
                PgBouncerSettings(enabled=True, port=port, binary=str(_fake_pooler(socket_tempdir))),
                environ,
            )
        assert isinstance(outcome, str), outcome
        pooled: Final = urllib.parse.urlsplit(outcome)
        assert (pooled.username, pooled.hostname, pooled.port) == ("litellm_pgbouncer", "127.0.0.1", port)
        assert "MINTED_TOKEN" not in outcome


class TestMain:
    def test_workers_inherit_the_loopback_url_the_supervisor_installed(
        self, password_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, socket_tempdir: Path
    ):
        port: Final = _free_port()
        monkeypatch.setenv("LITELLM_PGBOUNCER_ENABLED", "true")
        monkeypatch.setenv("LITELLM_PGBOUNCER_PORT", str(port))
        monkeypatch.setenv("LITELLM_PGBOUNCER_BINARY", str(_fake_pooler(socket_tempdir)))
        monkeypatch.setenv("KEEPALIVE_TIMEOUT", "75")
        served: Final[list[tuple[str, ...]]] = []
        main(("--workers", "4"), serve=lambda argv: served.append(tuple(argv)))

        pooled: Final = os.environ["DATABASE_URL"]
        assert urllib.parse.urlsplit(pooled).hostname == "127.0.0.1"
        assert urllib.parse.urlsplit(pooled).port == port
        assert urllib.parse.urlsplit(pooled).username == "litellm_pgbouncer"
        assert "p%40ss" not in pooled
        assert _query(pooled)["pgbouncer"] == "true"
        assert _uvicorn_params(served[0])["timeout_keep_alive"] == 75

        DatabaseURLSettings.from_env().apply_to_env()
        assert urllib.parse.urlsplit(os.environ["DATABASE_URL"]).netloc == urllib.parse.urlsplit(pooled).netloc
        assert _query(os.environ["DATABASE_URL"])["pgbouncer"] == "true"

    def test_iam_workers_keep_the_loopback_url_instead_of_minting_their_own(
        self, password_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, socket_tempdir: Path
    ):
        port: Final = _free_port()
        monkeypatch.delenv("DATABASE_PASSWORD")
        monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
        monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
        monkeypatch.setenv("LITELLM_PGBOUNCER_ENABLED", "true")
        monkeypatch.setenv("LITELLM_PGBOUNCER_PORT", str(port))
        monkeypatch.setenv("LITELLM_PGBOUNCER_BINARY", str(_fake_pooler(socket_tempdir)))
        served: Final[list[tuple[str, ...]]] = []
        with _minted_iam_token("SUPERVISOR_TOKEN"):
            main(("--workers", "4"), serve=lambda argv: served.append(tuple(argv)))
        pooled: Final = os.environ["DATABASE_URL"]
        assert urllib.parse.urlsplit(pooled).netloc.endswith(f"@127.0.0.1:{port}")
        assert "SUPERVISOR_TOKEN" not in pooled
        assert os.environ[PGBOUNCER_POOLED_ENV_VAR] == "true"
        assert len(served) == 1

        with _minted_iam_token("WORKER_TOKEN"):
            DatabaseURLSettings.from_env().apply_to_env()
        assert os.environ["DATABASE_URL"] == pooled

    @pytest.mark.parametrize(
        ("rust", "warning"),
        [
            ("false", ""),
            (
                "true",
                "LITELLM_RUST_FIRST does not support a Unix socket on the public port, "
                "so Python serves the public port without Rust forwarding\n",
            ),
        ],
    )
    def test_python_alone_keeps_uvicorn_socket_options(
        self,
        rust: str,
        warning: str,
        password_env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("LITELLM_RUST_FIRST", rust)
        monkeypatch.delenv("_LITELLM_GATEWAY_SUPERVISED", raising=False)
        monkeypatch.delenv("LITELLM_PGBOUNCER_ENABLED", raising=False)
        served: Final[list[tuple[str, ...]]] = []
        main(("--uds", "/tmp/gateway.sock"), serve=lambda argv: served.append(tuple(argv)))
        assert _uvicorn_params(served[0])["uds"] == "/tmp/gateway.sock"
        assert capsys.readouterr().err == warning

    def test_an_invalid_rust_switch_stops_the_gateway_naming_the_variable(
        self, password_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LITELLM_RUST_FIRST", "maybe")
        monkeypatch.delenv("_LITELLM_GATEWAY_SUPERVISED", raising=False)
        served: Final[list[tuple[str, ...]]] = []
        with pytest.raises(SystemExit) as stopped:
            main(("--workers", "4"), serve=lambda argv: served.append(tuple(argv)))
        assert str(stopped.value).startswith("LITELLM_RUST_FIRST is invalid: ")
        assert served == []

    @pytest.mark.parametrize(
        ("argv", "status", "said"),
        [
            pytest.param(("--nope",), 2, "--nope", id="a bad flag is a usage error"),
            pytest.param(("--help",), 0, "--timeout-keep-alive", id="help is printed once and exits cleanly"),
        ],
    )
    def test_the_supervisor_reports_uvicorn_arguments_the_way_uvicorn_does(
        self,
        argv: tuple[str, ...],
        status: int,
        said: str,
        password_env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("LITELLM_RUST_FIRST", "true")
        monkeypatch.delenv("_LITELLM_GATEWAY_SUPERVISED", raising=False)
        served: Final[list[tuple[str, ...]]] = []
        with pytest.raises(SystemExit) as stopped:
            main(argv, serve=lambda served_argv: served.append(tuple(served_argv)))
        printed: Final = capsys.readouterr()
        assert stopped.value.code == status
        assert (printed.out + printed.err).count(said) == 1
        assert served == []

    def test_a_pooler_that_cannot_start_stops_the_gateway_before_uvicorn(
        self, password_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("LITELLM_PGBOUNCER_ENABLED", "true")
        monkeypatch.setenv("LITELLM_PGBOUNCER_BINARY", str(tmp_path / "missing-pgbouncer"))
        served: Final[list[tuple[str, ...]]] = []
        with pytest.raises(SystemExit) as stopped:
            main(("--workers", "4"), serve=lambda argv: served.append(tuple(argv)))
        assert "missing-pgbouncer" in str(stopped.value)
        assert served == []
