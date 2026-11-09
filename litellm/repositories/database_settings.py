from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Final
from urllib.parse import parse_qsl, unquote, urlsplit

from litellm.proxy.db.token_auth import DatabaseTokenAuth, mint_database_token, parse_iam_endpoint_from_url


class NativeDatabaseSettingsError(ValueError):
    def __init__(self, detail: str) -> None:
        super().__init__(f"Invalid native database settings. {detail}")


def _password(auth: DatabaseTokenAuth, url: str) -> str:
    return unquote(mint_database_token(auth, parse_iam_endpoint_from_url(url)))


@dataclass(frozen=True, slots=True)
class NativeDatabaseSettings:
    host: str
    port: int
    username: str
    password: str | None = field(repr=False)
    database: str
    schema: str
    ssl_mode: str
    ssl_root_cert: str | None
    max_connections: int
    acquire_timeout: float
    idle_timeout: float | None
    statement_cache_capacity: int
    connect_timeout: float
    password_provider: Callable[[], str] | None = field(default=None, repr=False)
    token_auth: DatabaseTokenAuth | None = field(default=None, repr=False)

    @classmethod
    def from_url(cls, url: str, token_auth: DatabaseTokenAuth | None = None) -> NativeDatabaseSettings:
        parsed: Final = urlsplit(url)
        if parsed.scheme not in ("postgres", "postgresql"):
            raise NativeDatabaseSettingsError("PostgreSQL is required")
        params: Final = dict(parse_qsl(parsed.query, keep_blank_values=True))
        host: Final = params.get("host", parsed.hostname or "localhost")
        ssl_mode: Final = params.get("sslmode", "prefer")
        return cls(
            host=host,
            port=parsed.port or 5432,
            username=unquote(parsed.username or "postgres"),
            password=None if parsed.password is None else unquote(parsed.password),
            database=unquote(parsed.path.lstrip("/") or "postgres"),
            schema=params.get("schema", "public"),
            ssl_mode="verify-full" if params.get("sslaccept") == "strict" and ssl_mode != "disable" else ssl_mode,
            ssl_root_cert=params.get("sslrootcert", params.get("sslcert")),
            max_connections=int(params.get("connection_limit", str((os.cpu_count() or 1) * 2 + 1))),
            acquire_timeout=float(params.get("pool_timeout", "10")),
            idle_timeout=float(params.get("max_idle_connection_lifetime", "300")) or None,
            statement_cache_capacity=0 if params.get("pgbouncer", "false").lower() == "true" else 100,
            connect_timeout=float(params.get("connect_timeout", "5")),
            password_provider=None if token_auth is None else partial(_password, token_auth, url),
            token_auth=token_auth,
        )
