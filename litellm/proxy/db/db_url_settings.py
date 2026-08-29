"""Assemble DATABASE_URL (+ optional DATABASE_URL_READ_REPLICA) from env.

The CLI (`proxy_cli.py`) assembles ``DATABASE_URL`` from discrete
``DATABASE_*`` env vars before Prisma initializes. The componentized
entrypoints (gateway / backend / migrations) bypass the CLI by uvicorn'ing
the app directly, so they call ``DatabaseURLSettings.from_env().apply_to_env()``
to do the same thing before importing ``proxy_server``.

The env var names this module reads are exactly the ones emitted by the
``helm/litellm`` chart's ``litellm.serverEnv`` block
(``helm/litellm/templates/_helpers.tpl``). Both auth styles and both
endpoints are covered:

  * Token auth (``IAM_TOKEN_DB_AUTH`` truthy for AWS RDS IAM, or
    ``AZURE_POSTGRESQL_AUTH`` truthy for Azure Database for PostgreSQL with
    Microsoft Entra ID): mint a short-lived token and embed it as the
    password. The writer URL is always (re)written because the token is
    freshly minted on every startup. The chart omits ``DATABASE_PASSWORD``
    in this mode. Enabling both toggles is a startup error.
  * Password auth: build a percent-encoded URL from ``DATABASE_PASSWORD``.
    The chart emits the discrete ``DATABASE_*`` fields (never a
    pre-assembled URL), so URL-reserved characters in the password survive
    instead of corrupting the URL. A pre-existing ``DATABASE_URL`` — e.g.
    one an operator pinned via ``extraEnv`` — is left untouched and wins.

The read replica is opt-in via ``DATABASE_HOST_READ_REPLICA`` and never
clobbers a pre-existing ``DATABASE_URL_READ_REPLICA``, so a token-auth writer
can run alongside a password-auth reader (or a precomputed reader URL). Reader
token auth is gated on the same global toggle as the writer: the chart only
emits the reader token env vars when the writer also uses token auth.
Reader-side fields fall back to the writer's user / name / schema / port /
password when their ``*_READ_REPLICA`` counterpart is unset, and to the
writer's connection params (pool size, timeouts, pgbouncer mode) for the
ones the reader URL does not pin itself.
"""

import os
import urllib.parse
from collections.abc import Mapping
from functools import partial
from types import MappingProxyType
from typing import Annotated, Final, cast

from pydantic import AliasChoices, BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm.proxy.db.token_auth import (
    AZURE_POSTGRESQL_AUTH_ENV_VAR,
    DEFAULT_POSTGRES_PORT,
    IAM_TOKEN_DB_AUTH_ENV_VAR,
    DatabaseTokenAuth,
    IAMEndpoint,
    build_database_token_auth,
    mint_database_token,
    token_auth_flag_enabled,
)

IamTokenAuthFlag = Annotated[bool, BeforeValidator(partial(token_auth_flag_enabled, env_var=IAM_TOKEN_DB_AUTH_ENV_VAR))]
AzureTokenAuthFlag = Annotated[
    bool, BeforeValidator(partial(token_auth_flag_enabled, env_var=AZURE_POSTGRESQL_AUTH_ENV_VAR))
]

DISABLE_PREPARED_STATEMENTS_ENV_VAR: Final = "DATABASE_DISABLE_PREPARED_STATEMENTS"
DisablePreparedStatementsFlag = Annotated[
    bool, BeforeValidator(partial(token_auth_flag_enabled, env_var=DISABLE_PREPARED_STATEMENTS_ENV_VAR))
]

# schema.prisma pins `provider = "postgresql"`, so these are the only schemes
# Prisma can actually connect with.
SUPPORTED_DB_SCHEMES: Final[frozenset[str]] = frozenset({"postgresql", "postgres"})
_MISSING_SCHEME: Final = "<missing scheme>"


# An allowlist, deliberately not a denylist: only these pool and timeout params
# follow the writer to the read replica, so nothing that decides which tables a
# query resolves against (``schema``, or a ``search_path`` inside ``options``)
# can ever repoint the reader. Without them the reader pool silently falls back
# to Prisma's default size.
CONNECTION_PARAM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "connection_limit",
        "pool_timeout",
        "connect_timeout",
        "socket_timeout",
        "pgbouncer",
    }
)


def add_missing_query_params(url: str, params: Mapping[str, str | int | float]) -> str:
    """Return ``url`` with the ``params`` it does not already carry appended.

    Params the operator pinned on the URL win, so a hand-tuned replica URL keeps
    its values. Returns the URL untouched when there is nothing to add, leaving
    its existing encoding alone.
    """
    parsed: Final = urllib.parse.urlsplit(url)
    existing: Final = tuple(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    pinned: Final = frozenset(key for key, _ in existing)
    additions: Final = tuple((key, str(value)) for key, value in params.items() if key not in pinned)
    if not additions:
        return url
    query: Final = urllib.parse.urlencode(existing + additions)
    return urllib.parse.urlunsplit(parsed._replace(query=query))


def reader_shareable_params(params: Mapping[str, str | int | float]) -> Mapping[str, str | int | float]:
    """Return the subset of ``params`` the read replica is allowed to inherit."""
    return MappingProxyType({key: value for key, value in params.items() if key in CONNECTION_PARAM_KEYS})


def connection_params_from_url(url: str) -> Mapping[str, str | int | float]:
    """Return the connection params on ``url`` that the read replica shares."""
    return reader_shareable_params(
        MappingProxyType({key: value for key, value in urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query)})
    )


def unsupported_db_scheme(database_url: str) -> str | None:
    """Return the connection URL scheme when it is not PostgreSQL, else None.

    A `sqlite://` / `mysql://` URL can never connect against the
    postgresql-only datasource, but the resulting Prisma failure is opaque and
    version-dependent (a confusing migration error, or a startup that never
    binds). Callers use this to reject the URL up front with an actionable
    error instead.

    A schemeless value (e.g. a malformed DSN like ``user:pass@host/db``) yields
    the ``_MISSING_SCHEME`` placeholder rather than the raw URL, so callers that
    log the return value never echo embedded credentials.
    """
    scheme: Final = urllib.parse.urlsplit(database_url).scheme.lower()
    if scheme in SUPPORTED_DB_SCHEMES:
        return None
    return scheme or _MISSING_SCHEME


def unsupported_db_scheme_message(env_var: str, scheme: str) -> str:
    """Operator-facing message naming the offending env var and scheme."""
    return (
        f"{env_var} uses unsupported scheme '{scheme}'. LiteLLM's database "
        "features (virtual keys, store_model_in_db, spend tracking) require "
        "PostgreSQL; use a 'postgresql://' connection string. SQLite and other "
        "engines are not supported. "
        "See https://docs.litellm.ai/docs/proxy/virtual_keys"
    )


class DatabaseURLSettings(BaseSettings):
    """Discrete ``DATABASE_*`` env vars, loaded once at process start.

    Field names are internal; ``validation_alias`` pins each one to the exact
    env var the helm chart emits. ``DATABASE_USER`` doubles as
    ``DATABASE_USERNAME`` for parity with ``construct_database_url_from_env_vars``.
    """

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    iam_token_db_auth: IamTokenAuthFlag = Field(default=False, validation_alias=IAM_TOKEN_DB_AUTH_ENV_VAR)
    azure_postgresql_auth: AzureTokenAuthFlag = Field(default=False, validation_alias=AZURE_POSTGRESQL_AUTH_ENV_VAR)
    disable_prepared_statements: DisablePreparedStatementsFlag = Field(
        default=False, validation_alias=DISABLE_PREPARED_STATEMENTS_ENV_VAR
    )

    # Writer
    database_url: str | None = Field(default=None, validation_alias="DATABASE_URL")
    direct_url: str | None = Field(default=None, validation_alias="DIRECT_URL")
    database_host: str | None = Field(default=None, validation_alias="DATABASE_HOST")
    database_port: str = Field(default=DEFAULT_POSTGRES_PORT, validation_alias="DATABASE_PORT")
    database_user: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_USER", "DATABASE_USERNAME"),
    )
    database_name: str | None = Field(default=None, validation_alias="DATABASE_NAME")
    database_schema: str | None = Field(default=None, validation_alias="DATABASE_SCHEMA")
    database_password: str | None = Field(default=None, validation_alias="DATABASE_PASSWORD")

    # Read replica
    database_url_read_replica: str | None = Field(default=None, validation_alias="DATABASE_URL_READ_REPLICA")
    database_host_read_replica: str | None = Field(default=None, validation_alias="DATABASE_HOST_READ_REPLICA")
    database_port_read_replica: str | None = Field(default=None, validation_alias="DATABASE_PORT_READ_REPLICA")
    database_user_read_replica: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_USER_READ_REPLICA", "DATABASE_USERNAME_READ_REPLICA"),
    )
    database_name_read_replica: str | None = Field(default=None, validation_alias="DATABASE_NAME_READ_REPLICA")
    database_schema_read_replica: str | None = Field(default=None, validation_alias="DATABASE_SCHEMA_READ_REPLICA")
    database_password_read_replica: str | None = Field(default=None, validation_alias="DATABASE_PASSWORD_READ_REPLICA")

    @classmethod
    def from_env(cls) -> "DatabaseURLSettings":
        """Load the settings from ``os.environ`` (read at call time)."""
        return cls()

    def token_auth(self) -> DatabaseTokenAuth | None:
        """The token strategy the toggles ask for, or ``None`` for password auth.

        Raises ``RuntimeError`` when both toggles are on, since the password can only
        come from one source.
        """
        return build_database_token_auth(
            iam_token_db_auth=self.iam_token_db_auth,
            azure_postgresql_auth=self.azure_postgresql_auth,
        )

    def build_writer_url(self) -> str | None:
        """Return the writer URL to set, or ``None`` to leave it as-is.

        Raises ``RuntimeError`` (naming the offending vars) when token auth is
        enabled but a required field is missing — the proxy cannot recover
        from this and a clear startup error beats a Prisma connect failure.
        """
        auth: Final = self.token_auth()
        if auth is not None:
            missing: Final = tuple(
                env
                for env, val in (
                    ("DATABASE_HOST", self.database_host),
                    ("DATABASE_USER", self.database_user),
                    ("DATABASE_NAME", self.database_name),
                )
                if not val
            )
            if missing:
                raise RuntimeError(
                    f"{auth.env_var} is enabled but required DB env var(s) "
                    f"are unset: {', '.join(missing)}. Set them so the writer "
                    f"DATABASE_URL can be assembled with a minted {auth.label}."
                )
            endpoint: Final = IAMEndpoint(
                host=cast(str, self.database_host),
                port=self.database_port,
                user=cast(str, self.database_user),
                name=cast(str, self.database_name),
                schema=self.database_schema,
            )
            return endpoint.build_url(mint_database_token(auth, endpoint))

        # Password auth: an operator-pinned DATABASE_URL always wins.
        if self.database_url:
            return None
        if self.database_host and self.database_user and self.database_name:
            return self._password_url(
                user=self.database_user,
                password=self.database_password,
                host=self.database_host,
                port=self.database_port,
                name=self.database_name,
                schema=self.database_schema,
            )
        return None

    def build_reader_url(self) -> str | None:
        """Return the read-replica URL to set, or ``None`` to leave it as-is.

        Opt-in via ``DATABASE_HOST_READ_REPLICA``; never clobbers a
        pre-existing ``DATABASE_URL_READ_REPLICA``. Reader fields fall back
        to the writer's values.
        """
        if not self.database_host_read_replica:
            return None  # reader is opt-in
        if self.database_url_read_replica:
            return None  # never clobber an operator-supplied reader URL

        host: Final = self.database_host_read_replica
        port: Final = self.database_port_read_replica or self.database_port
        user: Final = self.database_user_read_replica or self.database_user
        name: Final = self.database_name_read_replica or self.database_name
        schema: Final = self.database_schema_read_replica or self.database_schema
        password: Final = self.database_password_read_replica or self.database_password

        auth: Final = self.token_auth()
        if auth is not None:
            missing: Final = tuple(
                env
                for env, val in (
                    ("DATABASE_USER[_READ_REPLICA]", user),
                    ("DATABASE_NAME[_READ_REPLICA]", name),
                )
                if not val
            )
            if missing:
                raise RuntimeError(
                    f"{auth.env_var} is enabled and DATABASE_HOST_READ_REPLICA "
                    "is set, but the reader could not resolve: "
                    f"{', '.join(missing)} (no *_READ_REPLICA value and no "
                    "writer fallback). Set the reader fields or the writer "
                    "defaults."
                )
            endpoint: Final = IAMEndpoint(
                host=host,
                port=port,
                user=cast(str, user),
                name=cast(str, name),
                schema=schema,
            )
            return endpoint.build_url(mint_database_token(auth, endpoint))

        if user and name:
            return self._password_url(
                user=user,
                password=password,
                host=host,
                port=port,
                name=name,
                schema=schema,
            )
        return None

    @staticmethod
    def _password_url(
        *,
        user: str,
        password: str | None,
        host: str,
        port: str,
        name: str,
        schema: str | None,
    ) -> str:
        """Percent-encode credentials into a ``postgresql://`` URL.

        Parity with ``construct_database_url_from_env_vars`` in
        ``proxy/utils.py``; ``password`` may be empty for a passwordless URL.
        """
        quote: Final = urllib.parse.quote_plus
        user_p: Final = quote(user)
        name_p: Final = quote(name)
        if password:
            url = f"postgresql://{user_p}:{quote(password)}@{host}:{port}/{name_p}"
        else:
            url = f"postgresql://{user_p}@{host}:{port}/{name_p}"
        if schema:
            url += f"?schema={schema}"
        return url

    def _raise_for_unsupported_scheme(self) -> None:
        """Reject an operator-pinned non-PostgreSQL writer / direct / reader URL.

        The componentized entrypoints (gateway / backend / migrations) call
        ``apply_to_env`` and then hand the URL straight to Prisma, bypassing
        the CLI's own guard. A pinned URL flows through untouched, so validate
        the same three vars the CLI guard checks (DATABASE_URL, DIRECT_URL, and
        the read replica) rather than letting Prisma stall on an unusable scheme.
        """
        for env_var, url in (
            ("DATABASE_URL", self.database_url),
            ("DIRECT_URL", self.direct_url),
            ("DATABASE_URL_READ_REPLICA", self.database_url_read_replica),
        ):
            if not url:
                continue
            bad_scheme = unsupported_db_scheme(url)
            if bad_scheme is not None:
                raise RuntimeError(unsupported_db_scheme_message(env_var, bad_scheme))

    def apply_writer_url_to_env(self) -> bool:
        """Write just the assembled writer URL into ``os.environ``.

        Split out because the CLI shares this minting path but resolves the read
        replica separately, so it must not pick up reader behavior on the way. The
        CLI runs its own scheme guard over the pinned URLs, so unlike
        ``apply_to_env`` this does not repeat it.
        """
        writer_url: Final = self.build_writer_url()
        if writer_url is None:
            return False
        os.environ["DATABASE_URL"] = writer_url
        # Normalize the toggles so downstream readers (PrismaWrapper's token
        # refresh) reliably see token auth on, regardless of spelling.
        if self.iam_token_db_auth:
            os.environ[IAM_TOKEN_DB_AUTH_ENV_VAR] = "True"
        if self.azure_postgresql_auth:
            os.environ[AZURE_POSTGRESQL_AUTH_ENV_VAR] = "True"
        return True

    def apply_to_env(self) -> bool:
        """Write the assembled URL(s) into ``os.environ``.

        Returns True iff this call set ``DATABASE_URL`` (token mint, or
        password auth that assembled a fresh URL). False means there was
        nothing to do — an operator-pinned URL, or no discrete fields.
        """
        self._raise_for_unsupported_scheme()
        wrote_writer: Final = self.apply_writer_url_to_env()

        # DATABASE_DISABLE_PREPARED_STATEMENTS maps to Prisma's `pgbouncer=true`
        # URL param, same as the CLI's `database_disable_prepared_statements`
        # config key. An explicit `pgbouncer` value already on the URL wins.
        if self.disable_prepared_statements:
            for env_var in ("DATABASE_URL", "DIRECT_URL"):
                url = os.environ.get(env_var)
                if url:
                    os.environ[env_var] = add_missing_query_params(url, MappingProxyType({"pgbouncer": "true"}))

        # The reader inherits the writer's connection params (pool size, timeouts,
        # pgbouncer mode). Without this the reader pool ignores the configured cap
        # and falls back to Prisma's `num_physical_cpus * 2 + 1` default.
        reader_url: Final = self.build_reader_url() or self.database_url_read_replica
        if reader_url is not None:
            os.environ["DATABASE_URL_READ_REPLICA"] = add_missing_query_params(
                reader_url,
                connection_params_from_url(os.environ.get("DATABASE_URL", "")),
            )

        return wrote_writer
