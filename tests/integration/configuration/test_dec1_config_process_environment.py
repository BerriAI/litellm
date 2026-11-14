"""Environment variables and command-line flags that shape the proxy outside the config file.

Covered: log level (`LITELLM_LOG`, `--detailed_debug`), the docs routes (`NO_DOCS`, `NO_REDOC`,
`DOCS_URL`), the admin UI switch (`DISABLE_ADMIN_UI`), the public URL hints (`SERVER_ROOT_PATH`,
`PROXY_BASE_URL`), where the database URL comes from (`DATABASE_URL`, `general_settings.database_url`
as a literal or an `os.environ/` reference, or the `DATABASE_HOST` family), and the unauthenticated
health probes, which must reflect what was configured.

These are read at import or boot time by the Python process today. Whatever replaces the front door and
the config loader has to honour the same names with the same precedence, because they live in deployment
manifests rather than in the config file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import pytest

from tests.integration._support.client import Gateway, string_value
from tests.integration._support.database import read_rows, scratch_database
from tests.integration.configuration.dec1_config_support import booted, readiness_details

_UI_CONFIG: Final = "/.well-known/litellm-ui-config"
_EMPTY: Final[dict[str, object]] = {"model_list": []}
_NO_DATABASE: Final = ("DATABASE_URL", "DATABASE_URL_READ_REPLICA", "DIRECT_URL")
_DATABASE_PARTS: Final = ("DATABASE_HOST", "DATABASE_USERNAME", "DATABASE_PASSWORD", "DATABASE_NAME", "DATABASE_SCHEMA")


def _key_lands_in(gateway: Gateway, database_url: str) -> bool:
    """Create a virtual key through the proxy and say whether its row is in the database at `database_url`."""
    alias: Final = "dec1-database-probe"
    gateway.post("/key/generate", {"key_alias": alias})
    return (
        len(
            read_rows(
                'SELECT token FROM "LiteLLM_VerificationToken" WHERE key_alias = %s',
                (alias,),
                database_url=database_url,
            )
        )
        == 1
    )


@pytest.mark.parametrize(
    ("level", "reported", "detailed"),
    [("DEBUG", "DEBUG", True), ("debug", "DEBUG", True), ("INFO", "INFO", False), ("info", "INFO", False)],
)
def test_litellm_log_sets_the_log_level_reported_by_readiness_details(
    gateway: Gateway, tmp_path: Path, level: str, reported: str, detailed: bool
) -> None:
    """`LITELLM_LOG` is set to DEBUG or INFO in either case. The authenticated `GET
    /health/readiness/details` must report `log_level` as the upper-case level name and
    `is_detailed_debug` as true only for DEBUG."""
    with booted(gateway, tmp_path, _EMPTY, environment={"LITELLM_LOG": level}) as owned:
        details: Final = readiness_details(owned.gateway)
    assert (details["log_level"], details["is_detailed_debug"]) == (reported, detailed)


def test_the_detailed_debug_flag_sets_debug_level_whatever_litellm_log_says(gateway: Gateway, tmp_path: Path) -> None:
    """The proxy is started with `--detailed_debug` while `LITELLM_LOG=INFO`. `GET
    /health/readiness/details` must report `log_level: DEBUG` and `is_detailed_debug: true`: the flag wins
    over the variable."""
    with booted(
        gateway, tmp_path, _EMPTY, environment={"LITELLM_LOG": "INFO"}, extra_arguments=("--detailed_debug",)
    ) as owned:
        details: Final = readiness_details(owned.gateway)
    assert (details["log_level"], details["is_detailed_debug"]) == ("DEBUG", True)


def test_no_docs_removes_the_swagger_page_from_the_root_path(gateway: Gateway, tmp_path: Path) -> None:
    """`NO_DOCS=True`. `GET /` must no longer serve the Swagger UI page. With the docs off the root path
    falls through to the proxy's own `home` route (litellm/proxy/proxy_server.py `@router.get("/")`), which
    answers 200 with the JSON string "LiteLLM: RUNNING". The API itself keeps working (`/health/liveliness`
    is 200)."""
    with booted(
        gateway, tmp_path, _EMPTY, environment={"NO_DOCS": "True"}, remove_environment=("DOCS_URL", "ROOT_REDIRECT_URL")
    ) as owned:
        root: Final = owned.gateway.client.get("/")
        alive: Final = owned.gateway.client.get("/health/liveliness")
    assert root.status_code == 200, root.text
    assert root.json() == "LiteLLM: RUNNING"
    assert "swagger-ui" not in root.text
    assert alive.status_code == 200


def test_docs_url_moves_the_swagger_page_and_wins_over_no_docs(gateway: Gateway, tmp_path: Path) -> None:
    """`DOCS_URL=/dec1-docs` together with `NO_DOCS=True`. `GET /dec1-docs` must return the Swagger UI
    page (200, HTML containing `swagger-ui`): an explicit docs URL takes precedence over the off switch.
    UNVERIFIED: the marker text `swagger-ui` in the served HTML."""
    with booted(gateway, tmp_path, _EMPTY, environment={"DOCS_URL": "/dec1-docs", "NO_DOCS": "True"}) as owned:
        docs: Final = owned.gateway.client.get("/dec1-docs")
    assert docs.status_code == 200, docs.text
    assert "swagger-ui" in docs.text


def test_no_redoc_removes_the_redoc_page(gateway: Gateway, tmp_path: Path) -> None:
    """`NO_REDOC=True`. `GET /redoc` must return 404, while without the variable it is the ReDoc page.
    UNVERIFIED: that nothing else is mounted at `/redoc` when ReDoc is off."""
    with booted(
        gateway, tmp_path, _EMPTY, environment={"NO_REDOC": "True"}, remove_environment=("REDOC_URL",)
    ) as owned:
        redoc: Final = owned.gateway.client.get("/redoc")
    assert redoc.status_code == 404, redoc.text


def test_disable_admin_ui_is_reported_by_the_ui_discovery_document(gateway: Gateway, tmp_path: Path) -> None:
    """`DISABLE_ADMIN_UI=true`. The unauthenticated `GET /.well-known/litellm-ui-config` must report
    `admin_ui_disabled: true`."""
    with booted(gateway, tmp_path, _EMPTY, environment={"DISABLE_ADMIN_UI": "true"}) as owned:
        body: Final = owned.gateway.client.get(_UI_CONFIG).json()
    assert body["admin_ui_disabled"] is True


def test_disable_admin_ui_replaces_the_sso_login_entry_with_a_disabled_notice(gateway: Gateway, tmp_path: Path) -> None:
    """`DISABLE_ADMIN_UI=true`. `GET /sso/key/generate`, the UI login entry point, must return a 200 HTML
    page whose heading is `Admin UI is Disabled` instead of starting a login."""
    with booted(gateway, tmp_path, _EMPTY, environment={"DISABLE_ADMIN_UI": "true"}) as owned:
        page: Final = owned.gateway.client.get("/sso/key/generate")
    assert page.status_code == 200, page.text
    assert "<h1>Admin UI is Disabled</h1>" in page.text


def test_the_admin_ui_is_reported_enabled_by_default(gateway: Gateway, tmp_path: Path) -> None:
    """Without `DISABLE_ADMIN_UI`, `GET /.well-known/litellm-ui-config` must report
    `admin_ui_disabled: false`."""
    with booted(gateway, tmp_path, _EMPTY, remove_environment=("DISABLE_ADMIN_UI",)) as owned:
        body: Final = owned.gateway.client.get(_UI_CONFIG).json()
    assert body["admin_ui_disabled"] is False


def test_proxy_base_url_is_reported_by_the_ui_discovery_document(gateway: Gateway, tmp_path: Path) -> None:
    """`PROXY_BASE_URL=https://dec1-proxy.example`. `GET /.well-known/litellm-ui-config` must report
    `proxy_base_url` as exactly that value; without the variable it is null."""
    with booted(
        gateway, tmp_path / "set", _EMPTY, environment={"PROXY_BASE_URL": "https://dec1-proxy.example"}
    ) as owned:
        with_url: Final = owned.gateway.client.get(_UI_CONFIG).json()
    with booted(gateway, tmp_path / "unset", _EMPTY, remove_environment=("PROXY_BASE_URL",)) as owned:
        without_url: Final = owned.gateway.client.get(_UI_CONFIG).json()
    assert with_url["proxy_base_url"] == "https://dec1-proxy.example"
    assert without_url["proxy_base_url"] is None


def test_server_root_path_is_reported_by_the_ui_discovery_document(gateway: Gateway, tmp_path: Path) -> None:
    """`SERVER_ROOT_PATH=/dec1-root`. `GET /.well-known/litellm-ui-config` must report `server_root_path`
    as `/dec1-root`; the UI builds every link from it."""
    with booted(
        gateway,
        tmp_path,
        _EMPTY,
        environment={"SERVER_ROOT_PATH": "/dec1-root"},
        remove_environment=("SERVER_ROOT_PATHS",),
    ) as owned:
        body: Final = owned.gateway.client.get(_UI_CONFIG).json()
    assert body["server_root_path"] == "/dec1-root"


def test_server_root_path_serves_the_api_under_the_prefix_and_at_the_bare_path(
    gateway: Gateway, tmp_path: Path
) -> None:
    """`SERVER_ROOT_PATH=/dec1-root`. `GET /dec1-root/health/liveliness` must return 200 `"I'm alive!"`,
    and so must the unprefixed `GET /health/liveliness` (a reverse proxy may or may not strip the prefix).
    UNVERIFIED: that the prefixed path is routed by the app itself rather than only by an upstream proxy."""
    with booted(
        gateway,
        tmp_path,
        _EMPTY,
        environment={"SERVER_ROOT_PATH": "/dec1-root"},
        remove_environment=("SERVER_ROOT_PATHS",),
    ) as owned:
        prefixed: Final = owned.gateway.client.get("/dec1-root/health/liveliness")
        bare: Final = owned.gateway.client.get("/health/liveliness")
    assert (prefixed.status_code, prefixed.json()) == (200, "I'm alive!")
    assert (bare.status_code, bare.json()) == (200, "I'm alive!")


def test_liveliness_and_liveness_answer_without_a_key(gateway: Gateway, tmp_path: Path) -> None:
    """`GET /health/liveliness` and its Kubernetes-spelled alias `GET /health/liveness` must both return
    200 with the JSON string `"I'm alive!"` to a caller with no key."""
    with booted(gateway, tmp_path, _EMPTY) as owned:
        answers: Final = [owned.gateway.client.get(path) for path in ("/health/liveliness", "/health/liveness")]
    assert [(answer.status_code, answer.json()) for answer in answers] == [(200, "I'm alive!")] * 2


def test_readiness_options_advertises_get_and_options(gateway: Gateway, tmp_path: Path) -> None:
    """`OPTIONS /health/readiness` must return 200 with `Allow: GET, OPTIONS` and
    `Access-Control-Allow-Methods: GET, OPTIONS`, for load balancers that preflight the probe."""
    with booted(gateway, tmp_path, _EMPTY) as owned:
        response: Final = owned.gateway.client.options("/health/readiness")
    assert response.status_code == 200, response.text
    assert response.headers["allow"] == "GET, OPTIONS"
    assert response.headers["access-control-allow-methods"] == "GET, OPTIONS"


def test_readiness_details_reports_connected_database_no_cache_and_the_running_version(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A proxy with a database and no `cache` setting. The authenticated `GET /health/readiness/details`
    must report `status: healthy`, `db: connected`, `cache: null` and a non-empty `litellm_version`."""
    with booted(gateway, tmp_path, _EMPTY) as owned:
        details: Final = readiness_details(owned.gateway)
    assert (details["status"], details["db"], details["cache"]) == ("healthy", "connected", None)
    assert string_value(details["litellm_version"])


def test_without_any_database_setting_readiness_says_not_connected_and_stays_200(
    gateway: Gateway, tmp_path: Path
) -> None:
    """No `DATABASE_URL`, no `general_settings.database_url` and no `DATABASE_HOST` family. The proxy must
    still boot and the public `GET /health/readiness` must return 200 `{"status": "healthy", "db": "Not
    connected"}`: running without a database is a supported mode, not an unhealthy one."""
    with booted(gateway, tmp_path, _EMPTY, remove_environment=(*_NO_DATABASE, *_DATABASE_PARTS)) as owned:
        response: Final = owned.gateway.client.get("/health/readiness")
    assert response.status_code == 200, response.text
    assert response.json() == {"status": "healthy", "db": "Not connected"}


def test_without_a_database_a_key_management_call_explains_that_a_database_is_needed(
    gateway: Gateway, tmp_path: Path
) -> None:
    """With no database configured, `POST /key/generate` must fail with a message starting `DB not
    connected. This endpoint needs a database; set DATABASE_URL to a PostgreSQL connection string`.
    UNVERIFIED: the status code (the message constant is shared by endpoints that use 400 and 500)."""
    with booted(gateway, tmp_path, _EMPTY, remove_environment=(*_NO_DATABASE, *_DATABASE_PARTS)) as owned:
        response: Final = owned.gateway.request("POST", "/key/generate", {})
    assert response.status_code != 200, response.text
    assert (
        "DB not connected. This endpoint needs a database; set DATABASE_URL to a PostgreSQL connection string"
        in response.text
    )


def test_database_url_in_the_file_as_a_literal_connects_the_proxy_to_that_database(
    gateway: Gateway, tmp_path: Path
) -> None:
    """`DATABASE_URL` is absent from the environment and `general_settings.database_url` holds a literal
    connection string for a scratch database. Readiness must report `db: connected` and a key created
    through the proxy must be stored in that scratch database."""
    with scratch_database("dec1cfg") as url:
        config: Final = {"model_list": [], "general_settings": {"database_url": url}}
        with booted(gateway, tmp_path, config, remove_environment=(*_NO_DATABASE, *_DATABASE_PARTS)) as owned:
            db_status: Final = owned.gateway.client.get("/health/readiness").json()["db"]
            stored: Final = _key_lands_in(owned.gateway, url)
    assert db_status == "connected"
    assert stored


def test_database_url_in_the_file_as_an_environment_reference_is_resolved(gateway: Gateway, tmp_path: Path) -> None:
    """`general_settings.database_url: os.environ/DEC1_DATABASE_URL`, with that variable holding a scratch
    database's connection string and no `DATABASE_URL`. A key created through the proxy must be stored in
    the scratch database."""
    with scratch_database("dec1cfg") as url:
        config: Final = {"model_list": [], "general_settings": {"database_url": "os.environ/DEC1_DATABASE_URL"}}
        with booted(
            gateway,
            tmp_path,
            config,
            environment={"DEC1_DATABASE_URL": url},
            remove_environment=(*_NO_DATABASE, *_DATABASE_PARTS),
        ) as owned:
            stored: Final = _key_lands_in(owned.gateway, url)
    assert stored


def test_database_url_in_the_file_wins_over_the_environment_variable(gateway: Gateway, tmp_path: Path) -> None:
    """`DATABASE_URL` in the environment points at one scratch database and `general_settings.database_url`
    at another. A key created through the proxy must be stored in the file's database and not in the
    environment's: the file value replaces the variable before the database client starts.
    Assumes passing `DATABASE_URL` explicitly makes the harness use it as is (no private database)."""
    with scratch_database("dec1env") as from_environment, scratch_database("dec1file") as from_file:
        config: Final = {"model_list": [], "general_settings": {"database_url": from_file}}
        with booted(
            gateway,
            tmp_path,
            config,
            environment={"DATABASE_URL": from_environment},
            remove_environment=("DATABASE_URL_READ_REPLICA",),
        ) as owned:
            in_file_database: Final = _key_lands_in(owned.gateway, from_file)
        tables: Final = read_rows(
            "SELECT 1 AS present FROM information_schema.tables WHERE table_name = %s",
            ("LiteLLM_VerificationToken",),
            database_url=from_environment,
        )
        in_environment_database: Final = bool(tables) and bool(
            read_rows(
                'SELECT token FROM "LiteLLM_VerificationToken" WHERE key_alias = %s',
                ("dec1-database-probe",),
                database_url=from_environment,
            )
        )
    assert in_file_database
    assert not in_environment_database


def test_database_host_username_password_and_name_are_assembled_into_a_connection(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Neither `DATABASE_URL` nor `general_settings.database_url` is set; `DATABASE_HOST` (host and port),
    `DATABASE_USERNAME`, `DATABASE_PASSWORD` and `DATABASE_NAME` describe a scratch database. The proxy
    must assemble `postgresql://user:password@host/name` from them: readiness reports `db: connected` and
    a key created through the proxy is stored in that database.
    Assumes the harness database URL has a user, a password and an explicit host."""
    with scratch_database("dec1cfg") as url:
        parts: Final = urlsplit(url)
        environment: Final = {
            "DATABASE_HOST": parts.netloc.rsplit("@", 1)[-1],
            "DATABASE_USERNAME": parts.username or "",
            "DATABASE_PASSWORD": parts.password or "",
            "DATABASE_NAME": parts.path.lstrip("/"),
        }
        with booted(
            gateway,
            tmp_path,
            _EMPTY,
            environment=environment,
            remove_environment=(*_NO_DATABASE, "DATABASE_SCHEMA"),
        ) as owned:
            db_status: Final = owned.gateway.client.get("/health/readiness").json()["db"]
            stored: Final = _key_lands_in(owned.gateway, url)
    assert db_status == "connected"
    assert stored


def test_database_url_wins_over_the_database_host_family(gateway: Gateway, tmp_path: Path) -> None:
    """`DATABASE_URL` points at a scratch database while `DATABASE_HOST`, `DATABASE_USERNAME` and
    `DATABASE_NAME` describe a host that does not exist. The proxy must boot and store a new key in the
    `DATABASE_URL` database: the parts are only used when no URL is given."""
    with scratch_database("dec1cfg") as url:
        environment: Final = {
            "DATABASE_URL": url,
            "DATABASE_HOST": "dec1-no-such-host.invalid:5432",
            "DATABASE_USERNAME": "dec1-nobody",
            "DATABASE_NAME": "dec1-nothing",
        }
        with booted(
            gateway, tmp_path, _EMPTY, environment=environment, remove_environment=("DATABASE_URL_READ_REPLICA",)
        ) as owned:
            stored: Final = _key_lands_in(owned.gateway, url)
    assert stored
