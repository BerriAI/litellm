import importlib
import os
import socket
import sys
import textwrap
import time
import urllib.parse
from collections.abc import Iterator
from pathlib import Path
from typing import Final, cast
from unittest.mock import MagicMock, patch

import pytest
from uvicorn.importer import import_from_string
from uvicorn.main import main as uvicorn_main

from gateway.launch import (
    GATEWAY_APP,
    _pooled_child_environments,
    main,
    managed_pool_database_url,
    pool_database_url,
    uvicorn_argv,
)
from gateway.supervisor import (
    RUST_GATEWAY_BINARY,
    ConfigurationFailure,
    build_configuration,
    gateway_enabled,
)
from litellm.proxy.db.db_url_settings import DatabaseURLSettings
from litellm.proxy.db.pgbouncer import (
    PGBOUNCER_POOLED_ENV_VAR,
    ManagedPgBouncerRuntime,
    PgBouncerError,
    PgBouncerSettings,
)
from litellm.proxy.proxy_server import app as proxy_app

DB_ENV: Final = {
    "DATABASE_HOST": "db.internal",
    "DATABASE_PORT": "5432",
    "DATABASE_USER": "litellm_pool",
    "DATABASE_NAME": "litellm",
    "DATABASE_PASSWORD": "p@ss",
}


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

        served: Final = import_from_string(cast(str, _uvicorn_params(uvicorn_argv((), {}))["app"]))
        assert served is importlib.import_module("gateway.main").app


class TestPoolDatabaseUrl:
    def test_rust_mode_without_a_writer_stays_database_free(self, password_env: dict[str, str]) -> None:
        assert (
            managed_pool_database_url(
                DatabaseURLSettings.from_env(),
                PgBouncerSettings(enabled=True),
                {"DATABASE_URL": ""},
            )
            is None
        )

    def test_rust_managed_pooler_is_forced_for_a_writer_when_the_flag_is_off(
        self, password_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        port: Final = _free_port()
        binary: Final = _fake_pooler(tmp_path)
        monkeypatch.setenv("DATABASE_URL", "postgresql://litellm_pool:p%40ss@db.internal:5432/litellm")
        outcome: Final = managed_pool_database_url(
            DatabaseURLSettings.from_env(),
            PgBouncerSettings(enabled=False, port=port, binary=str(binary)),
            os.environ,
        )
        assert isinstance(outcome, ManagedPgBouncerRuntime), outcome
        try:
            assert urllib.parse.urlsplit(outcome.pooled_url).port == port
            configuration: Final = build_configuration(
                ("--host", "0.0.0.0", "--port", "4000"),
                {RUST_GATEWAY_BINARY: "/bin/true"},
                sys.executable,
            )
            assert not isinstance(configuration, ConfigurationFailure)
            python_environment, rust_environment = _pooled_child_environments(
                configuration.python_environment,
                configuration.rust_environment,
                outcome,
            )
            assert python_environment["DATABASE_URL"] == rust_environment["DATABASE_URL"] == outcome.pooled_url
            assert python_environment[PGBOUNCER_POOLED_ENV_VAR] == rust_environment[PGBOUNCER_POOLED_ENV_VAR] == "true"
            assert python_environment["LITELLM_PGBOUNCER_ENABLED"] == "false"
            assert rust_environment.get("LITELLM_PGBOUNCER_ENABLED") == configuration.rust_environment.get(
                "LITELLM_PGBOUNCER_ENABLED"
            )
        finally:
            outcome.stop(time.monotonic() + 2)

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
        self, password_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
        monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
        port: Final = _free_port()
        environ: Final = {"DATABASE_URL": "postgresql://litellm:MINTED_TOKEN@db.internal:5432/litellm"}
        with _minted_iam_token("MINTED_TOKEN"):
            outcome: Final = pool_database_url(
                DatabaseURLSettings.from_env(),
                PgBouncerSettings(enabled=True, port=port, binary=str(_fake_pooler(tmp_path))),
                environ,
            )
        assert isinstance(outcome, str), outcome
        pooled: Final = urllib.parse.urlsplit(outcome)
        assert (pooled.username, pooled.hostname, pooled.port) == ("litellm_pgbouncer", "127.0.0.1", port)
        assert "MINTED_TOKEN" not in outcome


class TestRustGatewayConfiguration:
    @pytest.fixture
    def rust_binary(self, tmp_path: Path) -> Path:
        binary: Final = tmp_path / "litellm-gateway"
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o700)
        return binary

    @pytest.fixture
    def clear_uvicorn_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "UVICORN_HOST",
            "UVICORN_PORT",
            "UVICORN_UDS",
            "UVICORN_FD",
            "UVICORN_RELOAD",
            "UVICORN_PROXY_HEADERS",
            "UVICORN_SSL_KEYFILE",
        ):
            monkeypatch.delenv(name, raising=False)

    def test_flag_requires_exact_lowercase_values(self) -> None:
        assert gateway_enabled({}) is False
        assert gateway_enabled({"LITELLM_RUST_GATEWAY": "true"}) is True
        assert isinstance(gateway_enabled({"LITELLM_RUST_GATEWAY": "TRUE"}), ConfigurationFailure)

    def test_private_worker_keeps_config_rootpath_keepalive_and_unknown_options(
        self,
        rust_binary: Path,
        clear_uvicorn_environment: None,
    ) -> None:
        environ: Final = {
            RUST_GATEWAY_BINARY: str(rust_binary),
            "CONFIG_FILE_PATH": "/app/config/config.yaml",
            "KEEPALIVE_TIMEOUT": "41",
        }
        configuration: Final = build_configuration(
            (
                "--workers",
                "3",
                "--proxy-headers",
                "--host",
                "0.0.0.0",
                "--port",
                "4000",
                "--root-path",
                "/gateway",
                "--unknown-option",
                "preserved",
                "--proxy-headers",
            ),
            environ,
            sys.executable,
        )

        assert not isinstance(configuration, ConfigurationFailure)
        assert configuration.python_command == (
            sys.executable,
            "-m",
            "gateway.launch",
            "--host",
            "127.0.0.1",
            "--port",
            "4002",
            "--proxy-headers",
            "--forwarded-allow-ips",
            "127.0.0.1",
            "--workers",
            "3",
            "--root-path",
            "/gateway",
            "--unknown-option",
            "preserved",
            "--timeout-keep-alive",
            "41",
        )
        assert configuration.python_environment["CONFIG_FILE_PATH"] == "/app/config/config.yaml"
        assert configuration.python_environment["LITELLM_RUST_GATEWAY"] == "false"
        assert configuration.rust_environment["LITELLM_RUST_GATEWAY"] == "true"
        assert configuration.rust_environment["LITELLM_PYTHON_UPSTREAM"] == "http://127.0.0.1:4002"
        assert configuration.rust_environment["HOST"] == "0.0.0.0"
        assert configuration.rust_environment["PORT"] == "4000"

    def test_database_writer_reserves_pgbouncer_port_when_the_python_only_flag_is_false(
        self, rust_binary: Path, clear_uvicorn_environment: None
    ) -> None:
        configuration: Final = build_configuration(
            ("--host", "0.0.0.0", "--port", "4000"),
            {
                RUST_GATEWAY_BINARY: str(rust_binary),
                "DATABASE_URL": "postgresql://app:pw@db/litellm",
                "LITELLM_PGBOUNCER_ENABLED": "false",
                "LITELLM_PGBOUNCER_PORT": "4000",
            },
            sys.executable,
        )
        assert isinstance(configuration, ConfigurationFailure)
        assert "PgBouncer" in configuration.reason
        assert "same TCP port" in configuration.reason

    @pytest.mark.parametrize(
        ("arguments", "extra_environment", "reason"),
        (
            (("--no-proxy-headers",), {}, "--no-proxy-headers"),
            (("--ssl-certfile", "/tmp/cert.pem"), {}, "TLS"),
            (("--env-file", "/tmp/uvicorn.env"), {}, "--env-file"),
            (("--uds", "/tmp/gateway.sock"), {}, "Unix socket"),
            (("--fd", "5"), {}, "file descriptor"),
            (("--http", "h2"), {}, "HTTP/1.1"),
            (("--host", "0.0.0.0", "--port", "4000"), {"PORT": "4001"}, "PORT conflicts"),
            (("--host", "0.0.0.0", "--port", "4000"), {"LITELLM_GATEWAY_METRICS_PORT": "4002"}, "same TCP port"),
            (
                ("--host", "0.0.0.0", "--port", "4000"),
                {"LITELLM_RUST_GATEWAY_TERMINATION_GRACE_SECONDS": "20"},
                "must allow Rust drain",
            ),
            (
                ("--host", "0.0.0.0", "--port", "4000"),
                {"LITELLM_RUST_GATEWAY_TERMINATION_GRACE_SECONDS": "0"},
                "must be a positive integer",
            ),
            (
                ("--host", "0.0.0.0", "--port", "4000"),
                {"LITELLM_COLLECTOR_ENABLED": "true", "LITELLM_COLLECTOR_ADDRESS": "tcp://127.0.0.1:4002"},
                "same TCP port",
            ),
        ),
    )
    def test_rejects_unsupported_or_conflicting_configuration(
        self,
        rust_binary: Path,
        clear_uvicorn_environment: None,
        arguments: tuple[str, ...],
        extra_environment: dict[str, str],
        reason: str,
    ) -> None:
        environ: Final = {RUST_GATEWAY_BINARY: str(rust_binary), **extra_environment}
        outcome: Final = build_configuration(arguments, environ, sys.executable)

        assert isinstance(outcome, ConfigurationFailure)
        assert reason in outcome.reason


class TestMain:
    def test_workers_inherit_the_loopback_url_the_supervisor_installed(
        self, password_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        port: Final = _free_port()
        monkeypatch.setenv("LITELLM_PGBOUNCER_ENABLED", "true")
        monkeypatch.setenv("LITELLM_PGBOUNCER_PORT", str(port))
        monkeypatch.setenv("LITELLM_PGBOUNCER_BINARY", str(_fake_pooler(tmp_path)))
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
        self, password_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        port: Final = _free_port()
        monkeypatch.delenv("DATABASE_PASSWORD")
        monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
        monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
        monkeypatch.setenv("LITELLM_PGBOUNCER_ENABLED", "true")
        monkeypatch.setenv("LITELLM_PGBOUNCER_PORT", str(port))
        monkeypatch.setenv("LITELLM_PGBOUNCER_BINARY", str(_fake_pooler(tmp_path)))
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
