import os
import socket
import sys
import textwrap
import urllib.parse
from pathlib import Path
from typing import Final, cast

import pytest
from uvicorn.importer import import_from_string
from uvicorn.main import main as uvicorn_main

import gateway.main
from gateway.launch import GATEWAY_APP, main, pool_database_url, uvicorn_argv
from litellm.proxy.db.db_url_settings import DatabaseURLSettings
from litellm.proxy.db.pgbouncer import PgBouncerError, PgBouncerSettings

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
def password_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    for var in ("DATABASE_URL", "IAM_TOKEN_DB_AUTH", "AZURE_POSTGRESQL_AUTH", "DATABASE_HOST_READ_REPLICA"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    for var, value in DB_ENV.items():
        monkeypatch.setenv(var, value)
    return dict(DB_ENV)


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

    def test_the_app_uvicorn_is_told_to_serve_is_the_trimmed_gateway(self):
        assert import_from_string(cast(str, _uvicorn_params(uvicorn_argv((), {}))["app"])) is gateway.main.app


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

    def test_token_auth_is_refused(self, password_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
        environ: Final = {"DATABASE_URL": "postgresql://litellm:token@db.internal:5432/litellm"}
        outcome: Final = pool_database_url(
            DatabaseURLSettings.from_env(),
            PgBouncerSettings(enabled=True, port=_free_port(), binary=str(_fake_pooler(tmp_path))),
            environ,
        )
        assert isinstance(outcome, PgBouncerError)
        assert "IAM_TOKEN_DB_AUTH" in outcome.reason


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
        assert urllib.parse.urlsplit(pooled).netloc == f"litellm_pool:p%40ss@127.0.0.1:{port}"
        assert _query(pooled)["pgbouncer"] == "true"
        assert _uvicorn_params(served[0])["timeout_keep_alive"] == 75

        DatabaseURLSettings.from_env().apply_to_env()
        worker_url: Final = os.environ["DATABASE_URL"]
        assert urllib.parse.urlsplit(worker_url).netloc == f"litellm_pool:p%40ss@127.0.0.1:{port}"
        assert _query(worker_url)["pgbouncer"] == "true"

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
