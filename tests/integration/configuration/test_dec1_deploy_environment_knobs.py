"""Environment variables and launch flags a deployment manifest sets, and how the proxy treats them at boot.

Contract (litellm/proxy/proxy_cli.py click options and `run_server`, litellm/proxy/db/db_url_settings.py,
litellm/proxy/rust_forward/mode.py and supervisor/preflight.py):

* Launch flags win over their environment variables (`--port` over `PORT`, `--host` over `HOST`,
  `--num_workers` over `NUM_WORKERS`), because click reads `envvar` only when the flag is absent. Helm
  charts set both, so a change of precedence silently moves a pod to another port.
* A database URL whose scheme is not `postgresql` or `postgres` stops the CLI at once with exit 1 and a
  message that names the variable and the scheme, never the credentials.
* `LITELLM_RUST_FIRST` is read strictly: an invalid value stops the CLI with one line naming it, a blank value
  is the same as unset, and a server Rust cannot front (Hypercorn) falls back to Python alone with a line
  saying so.
* `SERVER_ROOT_PATH` mounts the API under a prefix for a reverse proxy subpath while the bare paths keep
  working.

`LITELLM_LOG`, `NO_DOCS`, `DOCS_URL`, `PROXY_BASE_URL`, the UI discovery document and the `DATABASE_*` family
are in `test_dec1_config_process_environment.py`; these tests cover what it does not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway
from tests.integration._support.database import scratch_database
from tests.integration._support.process import owned_proxy_process, refused_boot_log
from tests.integration.configuration.dec1_deploy_support import (
    ALIVE,
    CHAT,
    FALLBACK_SUFFIX,
    NOT_CONNECTED,
    READY,
    budget_line,
    chat,
    deploy_config,
    log_text,
    plain,
    provider,
    readiness,
    write_config,
)

pytestmark: Final = pytest.mark.timeout(600)

SECRET: Final = "dec1-scheme-secret"
UNSUPPORTED_TAIL: Final = (
    ". LiteLLM's database features (virtual keys, store_model_in_db, spend tracking) require PostgreSQL; use a "
    "'postgresql://' connection string. SQLite and other engines are not supported. "
    "See https://docs.litellm.ai/docs/proxy/virtual_keys"
)
ROOT: Final = "/dec1-deploy-root"


def _config(tmp_path: Path, api_base: str, **general: JsonValue) -> Path:
    return write_config(tmp_path, deploy_config(api_base, general_settings=general))


def test_the_port_flag_wins_over_the_port_environment_variable(gateway: Gateway, tmp_path: Path) -> None:
    """The harness passes `--port <free port>` and the environment also sets `PORT=1`. The proxy must listen on
    the flag's port and serve a chat completion there: click reads `envvar="PORT"` only when the flag is absent
    (proxy_cli.py:1017)."""
    with (
        provider() as upstream,
        owned_proxy_process(
            gateway, tmp_path / "proxy", {"PORT": "1"}, config=_config(tmp_path, upstream.api_base)
        ) as owned,
    ):
        assert chat(owned.gateway).status_code == 200


def test_the_host_flag_wins_over_the_host_environment_variable(gateway: Gateway, tmp_path: Path) -> None:
    """The harness passes `--host 127.0.0.1` and the environment sets `HOST=203.0.113.1`, an address this
    machine does not own. The proxy must still bind loopback and answer `GET /health/liveliness` with
    `"I'm alive!"` (proxy_cli.py:1016)."""
    with (
        provider() as upstream,
        owned_proxy_process(
            gateway, tmp_path / "proxy", {"HOST": "203.0.113.1"}, config=_config(tmp_path, upstream.api_base)
        ) as owned,
    ):
        assert owned.gateway.client.get("/health/liveliness").json() == ALIVE


def test_the_num_workers_flag_wins_over_the_num_workers_environment_variable(gateway: Gateway, tmp_path: Path) -> None:
    """The harness passes `--num_workers 1` while `NUM_WORKERS=3` is set, with
    `database_connection_pool_limit: 4` and no read replica. The connection budget line must read
    `... = 1 worker(s) x writer connection_limit 4 = up to 4 connections; ...`: the flag decided the worker
    count (proxy_cli.py:1026, db_url_settings.py:306)."""
    with (
        provider() as upstream,
        owned_proxy_process(
            gateway,
            tmp_path / "proxy",
            {"NUM_WORKERS": "3"},
            config=_config(tmp_path, upstream.api_base, database_connection_pool_limit=4),
            remove_environment=("DATABASE_URL_READ_REPLICA",),
        ) as owned,
    ):
        line: Final = budget_line(log_text(owned))
    assert "= 1 worker(s) x writer connection_limit 4 = up to 4 connections;" in line, line


@pytest.mark.parametrize(
    ("variable", "url", "scheme"),
    [
        ("DATABASE_URL", f"mysql://dec1:{SECRET}@127.0.0.1:3306/litellm", "mysql"),
        ("DATABASE_URL", f"sqlite:///tmp/{SECRET}.db", "sqlite"),
        ("DIRECT_URL", f"sqlite:///tmp/{SECRET}.db", "sqlite"),
        ("DATABASE_URL", f"//dec1:{SECRET}@127.0.0.1:5432/litellm", "<missing scheme>"),
    ],
    ids=["mysql", "sqlite", "direct-url-sqlite", "schemeless"],
)
def test_a_database_url_that_is_not_postgres_stops_the_boot_with_a_named_reason(
    gateway: Gateway, tmp_path: Path, variable: str, url: str, scheme: str
) -> None:
    """`<variable>` is set to a non Postgres URL. The proxy must exit non-zero before serving and print
    `LiteLLM Proxy: <variable> uses unsupported scheme '<scheme>'` followed by the PostgreSQL guidance
    (proxy_cli.py:1747, db_url_settings.py:366). The URL's password `dec1-scheme-secret` must not appear in
    the log: a schemeless value is reported as `<missing scheme>` so credentials are never echoed."""
    with provider() as upstream:
        log: Final = plain(
            refused_boot_log(gateway, tmp_path / "proxy", {variable: url}, config=_config(tmp_path, upstream.api_base))
        )
    assert f"LiteLLM Proxy: {variable} uses unsupported scheme '{scheme}'{UNSUPPORTED_TAIL}" in log, log[-3000:]
    assert SECRET not in log


def test_the_short_postgres_scheme_is_accepted(gateway: Gateway, tmp_path: Path) -> None:
    """`DATABASE_URL` is a fresh scratch database spelled with the `postgres://` scheme instead of
    `postgresql://`. The proxy must boot and `GET /health/readiness` must answer 200
    `{"status": "healthy", "db": "connected"}`: both spellings are in `SUPPORTED_DB_SCHEMES`
    (db_url_settings.py:82) and many managed Postgres offerings hand out the short one."""
    with scratch_database("dec1_deploy") as database, provider() as upstream:
        short: Final = urlunsplit(urlsplit(database)._replace(scheme="postgres"))
        with owned_proxy_process(
            gateway,
            tmp_path / "proxy",
            {"DATABASE_URL": short},
            config=_config(tmp_path, upstream.api_base),
            remove_environment=("DATABASE_URL_READ_REPLICA",),
        ) as owned:
            assert readiness(owned.gateway) == (200, dict(READY))


def test_an_invalid_rust_first_value_stops_the_cli_with_one_line_naming_it(gateway: Gateway, tmp_path: Path) -> None:
    """`LITELLM_RUST_FIRST=sometimes`. The proxy must exit non-zero before serving, and its log must contain
    `LITELLM_RUST_FIRST is invalid:` (`environment_error`, rust_forward/mode.py:82, raised as a click error at
    proxy_cli.py:1481). A typo in the switch must not silently pick a process model."""
    with provider() as upstream:
        log: Final = plain(
            refused_boot_log(
                gateway,
                tmp_path / "proxy",
                {"LITELLM_RUST_FIRST": "sometimes", "STORE_MODEL_IN_DB": "False"},
                config=_config(tmp_path, upstream.api_base),
            )
        )
    assert "LITELLM_RUST_FIRST is invalid:" in log, log[-3000:]


@pytest.mark.parametrize("value", ["", "   ", "false", "0"], ids=["empty", "blank", "false", "zero"])
def test_a_blank_or_false_rust_first_value_serves_from_python_alone_without_a_fallback_notice(
    gateway: Gateway, tmp_path: Path, value: str
) -> None:
    """`LITELLM_RUST_FIRST` is empty, whitespace, `false` or `0`. The proxy must boot and serve a chat
    completion, and its log must not contain `so Python serves the public port without Rust forwarding`: a
    blank value means unset (`_blank_as_unset`, litellm/rust_bridge/configuration.py:14), not a refused
    Rust start."""
    with (
        provider() as upstream,
        owned_proxy_process(
            gateway, tmp_path / "proxy", {"LITELLM_RUST_FIRST": value}, config=_config(tmp_path, upstream.api_base)
        ) as owned,
    ):
        assert chat(owned.gateway).status_code == 200
        assert FALLBACK_SUFFIX not in log_text(owned)


def test_rust_first_with_hypercorn_falls_back_to_python_alone_and_says_why(gateway: Gateway, tmp_path: Path) -> None:
    """`LITELLM_RUST_FIRST=true` and the proxy is launched with `--run_hypercorn`, which Rust cannot front.
    The proxy must still serve a chat completion, and its log must contain
    `LITELLM_RUST_FIRST does not support Hypercorn, so Python serves the public port without Rust forwarding`
    (supervisor/preflight.py:21, supervisor/__init__.py:38).
    UNVERIFIED: that the harness worker startup hooks (integration readiness route) also run under Hypercorn."""
    with (
        provider() as upstream,
        owned_proxy_process(
            gateway,
            tmp_path / "proxy",
            {"LITELLM_RUST_FIRST": "true"},
            config=_config(tmp_path, upstream.api_base),
            extra_arguments=("--run_hypercorn",),
        ) as owned,
    ):
        assert chat(owned.gateway).status_code == 200
        assert f"LITELLM_RUST_FIRST does not support Hypercorn{FALLBACK_SUFFIX}" in log_text(owned)


def test_hypercorn_counts_one_worker_in_the_connection_budget_whatever_num_workers_says(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The proxy is launched with `--run_hypercorn`, `database_connection_pool_limit: 6` and no read replica.
    The budget line must read `= 1 worker(s) x writer connection_limit 6 = up to 6 connections;`: Hypercorn
    runs one process, so the CLI passes `num_workers=1` to the message (proxy_cli.py:1850).
    UNVERIFIED: that the harness readiness probe works under Hypercorn."""
    with (
        provider() as upstream,
        owned_proxy_process(
            gateway,
            tmp_path / "proxy",
            {},
            config=_config(tmp_path, upstream.api_base, database_connection_pool_limit=6),
            remove_environment=("DATABASE_URL_READ_REPLICA",),
            extra_arguments=("--run_hypercorn",),
        ) as owned,
    ):
        line: Final = budget_line(log_text(owned))
    assert "= 1 worker(s) x writer connection_limit 6 = up to 6 connections;" in line, line


def test_server_root_path_serves_readiness_under_the_prefix(gateway: Gateway, tmp_path: Path) -> None:
    """`SERVER_ROOT_PATH=/dec1-deploy-root`. `GET /dec1-deploy-root/health/readiness` must answer 200 with
    `{"status": "healthy", "db": "connected"}`, so a reverse proxy that keeps the subpath can point the
    readiness probe at it.
    UNVERIFIED: that the prefixed path is routed by the app itself (the liveliness case is asserted the same way
    in test_dec1_config_process_environment.py)."""
    with (
        provider() as upstream,
        owned_proxy_process(
            gateway,
            tmp_path / "proxy",
            {"SERVER_ROOT_PATH": ROOT},
            config=_config(tmp_path, upstream.api_base),
            remove_environment=("SERVER_ROOT_PATHS",),
        ) as owned,
    ):
        assert readiness(owned.gateway, f"{ROOT}/health/readiness") == (200, dict(READY))


def test_server_root_path_serves_chat_completions_under_the_prefix_and_bare(gateway: Gateway, tmp_path: Path) -> None:
    """`SERVER_ROOT_PATH=/dec1-deploy-root`. A master key chat completion posted to
    `/dec1-deploy-root/v1/chat/completions` and one posted to the bare `/v1/chat/completions` must both answer
    200 and both reach the provider: a reverse proxy may or may not strip the prefix.
    UNVERIFIED: prefixed routing by the app for inference routes."""
    with (
        provider() as upstream,
        owned_proxy_process(
            gateway,
            tmp_path / "proxy",
            {"SERVER_ROOT_PATH": ROOT},
            config=_config(tmp_path, upstream.api_base),
            remove_environment=("SERVER_ROOT_PATHS",),
        ) as owned,
    ):
        prefixed: Final = chat(owned.gateway, path=f"{ROOT}/v1/chat/completions")
        bare: Final = chat(owned.gateway)
        assert (prefixed.status_code, bare.status_code) == (200, 200), (prefixed.text, bare.text)
        assert len([request for request in upstream.posted() if request.method == "POST"]) == 2


def test_server_root_path_serves_key_management_under_the_prefix(gateway: Gateway, tmp_path: Path) -> None:
    """`SERVER_ROOT_PATH=/dec1-deploy-root`. `POST /dec1-deploy-root/key/generate` from the master key must
    answer 200 with a `key` that then authorises a chat completion: the admin UI behind a subpath calls the
    management API through the prefix.
    UNVERIFIED: prefixed routing by the app for management routes."""
    with (
        provider() as upstream,
        owned_proxy_process(
            gateway,
            tmp_path / "proxy",
            {"SERVER_ROOT_PATH": ROOT},
            config=_config(tmp_path, upstream.api_base),
            remove_environment=("SERVER_ROOT_PATHS",),
        ) as owned,
    ):
        created: Final = owned.gateway.request("POST", f"{ROOT}/key/generate", {"models": [CHAT]})
        assert created.status_code == 200, created.text
        assert chat(owned.gateway, key=str(created.json()["key"])).status_code == 200


def test_litellm_mode_production_boots_and_serves_like_the_default(gateway: Gateway, tmp_path: Path) -> None:
    """`LITELLM_MODE=PRODUCTION`, which only skips loading a `.env` file from the working directory
    (proxy_cli.py:66). The proxy must boot, answer readiness 200 `connected` and serve a chat completion:
    production mode is not a different feature set."""
    with (
        provider() as upstream,
        owned_proxy_process(
            gateway, tmp_path / "proxy", {"LITELLM_MODE": "PRODUCTION"}, config=_config(tmp_path, upstream.api_base)
        ) as owned,
    ):
        assert readiness(owned.gateway) == (200, dict(READY))
        assert chat(owned.gateway).status_code == 200


@pytest.mark.parametrize(("level", "reported"), [("WARNING", "WARNING"), ("ERROR", "WARNING")])
def test_litellm_log_warning_and_error_are_reported_by_readiness_details(
    gateway: Gateway, tmp_path: Path, level: str, reported: str
) -> None:
    """`LITELLM_LOG` is `WARNING` or `ERROR`. `GET /health/readiness/details` must report that `log_level`
    and `is_detailed_debug: false`. DEBUG and INFO are covered elsewhere; these are the levels production
    manifests use.
    The value is `logging.getLevelName(verbose_logger.getEffectiveLevel())` (_health_endpoints.py:2204 on main).
    Pinned from CI run 44176682961: `LITELLM_LOG=ERROR` is reported as `WARNING`, because only INFO and DEBUG set
    the `verbose_logger` level (proxy_server.py:10304-10331 on main); ERROR only raises the handler's level
    (litellm/_logging.py:716-719), so records below ERROR are still dropped while the logger stays at its
    WARNING default."""
    with (
        provider() as upstream,
        owned_proxy_process(
            gateway, tmp_path / "proxy", {"LITELLM_LOG": level}, config=_config(tmp_path, upstream.api_base)
        ) as owned,
    ):
        details: Final = owned.gateway.get("/health/readiness/details")
    assert (details["log_level"], details["is_detailed_debug"]) == (reported, False)


def test_a_proxy_without_any_database_prints_no_connection_budget_line(gateway: Gateway, tmp_path: Path) -> None:
    """No `DATABASE_URL`, `DIRECT_URL` or read replica is set. The proxy must boot, readiness must say
    `{"status": "healthy", "db": "Not connected"}`, and the log must not contain
    `Postgres connection budget`: the line is printed only inside the database setup branch
    (proxy_cli.py:1725)."""
    with (
        provider() as upstream,
        owned_proxy_process(
            gateway,
            tmp_path / "proxy",
            {"STORE_MODEL_IN_DB": "False"},
            config=_config(tmp_path, upstream.api_base),
            remove_environment=("DATABASE_URL", "DATABASE_URL_READ_REPLICA", "DIRECT_URL"),
        ) as owned,
    ):
        assert readiness(owned.gateway) == (200, dict(NOT_CONNECTED))
        assert "Postgres connection budget" not in log_text(owned)
