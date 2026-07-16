"""Prisma connection-pool query params, applied consistently across startups.

The pool knobs (``connection_limit``, ``pool_timeout``, ``connect_timeout``,
``socket_timeout``, ``pgbouncer``, and any custom params) live in the database
URL's query string. They must be appended before Prisma initializes, on every
startup path: the CLI (``proxy_cli.py``), the componentized entrypoints
(``uvicorn gateway.main:app`` / ``backend.main:app``), and the proxy startup
event that all three funnel through.

Kept deliberately free of ``pydantic_settings`` (unlike ``db_url_settings``) so
``proxy_cli`` can import it at module scope without dragging a proxy-only
dependency into the base ``import litellm`` path.
"""

import os
import urllib.parse
from collections.abc import Mapping
from typing import Final, cast

from litellm._logging import verbose_proxy_logger
from litellm.secret_managers.main import str_to_bool

DEFAULT_DB_CONNECTION_POOL_LIMIT: Final[int] = 10
DEFAULT_DB_CONNECTION_POOL_TIMEOUT: Final[int] = 60

POOL_PARAM_DB_URL_ENV_VARS: Final[tuple[str, ...]] = (
    "DATABASE_URL",
    "DIRECT_URL",
    "DATABASE_URL_READ_REPLICA",
)


def build_db_connection_url_params(
    connection_limit: int,
    pool_timeout: float | None,
    connect_timeout: float | None = None,
    socket_timeout: float | None = None,
    disable_prepared_statements: bool = False,
    extra_params: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the Prisma DATABASE_URL query params controlling connection pool behavior.

    ``connect_timeout`` / ``socket_timeout`` map to the Prisma URL params of the
    same name (https://www.prisma.io/docs/orm/overview/databases/postgresql) and
    are omitted when None so Prisma's defaults apply. ``disable_prepared_statements``
    sets ``pgbouncer=true``, which makes Prisma stop using server-side prepared
    statements (pgbouncer transaction-pool compatible; also sidesteps the
    "cached plan must not change result type" error during rolling migrations).
    ``extra_params`` is an untyped passthrough: keys it provides win over the
    named arguments above, so it can override any default set here.
    """
    named: dict[str, object] = {
        "connection_limit": connection_limit,
        **({"pool_timeout": pool_timeout} if pool_timeout is not None else {}),
        **({"connect_timeout": connect_timeout} if connect_timeout is not None else {}),
        **({"socket_timeout": socket_timeout} if socket_timeout is not None else {}),
        **({"pgbouncer": "true"} if disable_prepared_statements else {}),
    }
    return {**named, **dict(extra_params)} if extra_params else named


def append_query_params(url: str | None, params: Mapping[str, object]) -> str:
    """Merge ``params`` into ``url``'s query string, params winning on conflict.

    Never logs the URL itself: it embeds the database credential (and, under IAM
    auth, a presigned token), so echoing it even at debug level leaks secrets.
    """
    if not isinstance(url, str) or url == "":
        verbose_proxy_logger.warning("append_query_params received empty or non-string URL, returning empty string")
        return ""
    parsed_url = urllib.parse.urlparse(url)
    merged: dict[str, object] = {**urllib.parse.parse_qs(parsed_url.query), **params}
    encoded_query = urllib.parse.urlencode(merged, doseq=True)
    return urllib.parse.urlunparse(parsed_url._replace(query=encoded_query))


def _optional_number(value: object) -> float | None:
    """Coerce a general_settings value to a number, or None when absent/invalid.

    ``bool`` is rejected explicitly because it is an ``int`` subclass and a stray
    ``true`` in YAML must not become a timeout of ``1``.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def pool_params_from_general_settings(general_settings: Mapping[str, object]) -> dict[str, object]:
    """Resolve the Prisma pool query params from a config ``general_settings`` block.

    Mirrors the keys the CLI reads so every startup path applies the same pool
    ceiling. An empty mapping yields the engine defaults.
    """
    raw_limit = _optional_number(general_settings.get("database_connection_pool_limit"))
    connection_limit = int(raw_limit) if raw_limit is not None else DEFAULT_DB_CONNECTION_POOL_LIMIT

    pool_timeout = _optional_number(general_settings.get("database_connection_timeout"))
    if pool_timeout is None:
        pool_timeout = _optional_number(general_settings.get("database_connection_pool_timeout"))
    if pool_timeout is None:
        pool_timeout = DEFAULT_DB_CONNECTION_POOL_TIMEOUT

    raw_disable = general_settings.get("database_disable_prepared_statements", False)
    disable_prepared_statements = (
        str_to_bool(raw_disable) is True if isinstance(raw_disable, str) else bool(raw_disable)
    )

    extra_raw = general_settings.get("database_extra_connection_params")
    extra_params = (
        cast("Mapping[str, object]", extra_raw)  # cast-ok: isinstance-guarded Mapping, YAML config keys are strings
        if isinstance(extra_raw, Mapping)
        else None
    )

    return build_db_connection_url_params(
        connection_limit=connection_limit,
        pool_timeout=pool_timeout,
        connect_timeout=_optional_number(general_settings.get("database_connect_timeout")),
        socket_timeout=_optional_number(general_settings.get("database_socket_timeout")),
        disable_prepared_statements=disable_prepared_statements,
        extra_params=extra_params,
    )


def apply_pool_params_to_db_urls(general_settings: Mapping[str, object]) -> None:
    """Append the resolved pool params to every DB URL env var that is set.

    Idempotent: params overwrite same-named keys, so calling it again (e.g. once
    in the CLI pre-migration and once in proxy startup) leaves the URL stable.
    Runs for the writer, DIRECT_URL, and the read replica so a componentized
    ``uvicorn gateway.main:app`` / ``backend.main:app`` startup gets the same
    ceiling the CLI applies, and so the reader URL stops silently keeping
    Prisma's default pool size.
    """
    params = pool_params_from_general_settings(general_settings)
    for env_var in POOL_PARAM_DB_URL_ENV_VARS:
        current = os.getenv(env_var)
        if current:
            os.environ[env_var] = append_query_params(current, params)
