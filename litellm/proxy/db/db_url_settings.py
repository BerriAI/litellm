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

import _ssl
import hashlib
import os
import socket
import ssl
import struct
import sys
import tempfile
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Final, Protocol, TypeAlias, cast

from pydantic import AliasChoices, BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm.proxy.db.pgbouncer import database_url_is_pooled
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
MAX_IDLE_CONNECTION_LIFETIME_ENV_VAR: Final = "DATABASE_MAX_IDLE_CONNECTION_LIFETIME"
DATABASE_SSLMODE_ENV_VAR: Final = "DATABASE_SSLMODE"
DATABASE_SSLROOTCERT_ENV_VAR: Final = "DATABASE_SSLROOTCERT"

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
        "max_idle_connection_lifetime",
        "pgbouncer",
    }
)

# Quaint never tests pooled connections on checkout and keeps them idle for
# 300s by default, past many infra idle timeouts, so dead sockets surface as
# `Error { kind: Closed }`. 60s recycles them first; explicit values win.
DEFAULT_MAX_IDLE_CONNECTION_LIFETIME: Final = 60
IDLE_LIFETIME_DEFAULT_PARAMS: Final[Mapping[str, int]] = MappingProxyType(
    {"max_idle_connection_lifetime": DEFAULT_MAX_IDLE_CONNECTION_LIFETIME}
)


def idle_lifetime_params(configured: float | None) -> Mapping[str, str | int | float]:
    """The `max_idle_connection_lifetime` to add to URLs that do not pin one.

    Applied via ``add_missing_query_params`` so a URL-pinned value always wins,
    whether the operator configured `database_max_idle_connection_lifetime` or not.
    """
    if configured is None:
        return IDLE_LIFETIME_DEFAULT_PARAMS
    return MappingProxyType({"max_idle_connection_lifetime": configured})


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


LIBPQ_VERIFY_SSLMODES: Final[frozenset[str]] = frozenset({"verify-ca", "verify-full"})
PRISMA_TLS_PARAM_KEYS: Final[frozenset[str]] = frozenset({"sslmode", "sslcert", "sslaccept"})
PEM_CERT_HEADER: Final = b"-----BEGIN CERTIFICATE-----"
PG_SSL_REQUEST: Final = struct.pack("!ii", 8, 80877103)
TLS_PROBE_TIMEOUT_SECONDS: Final = 10.0

RootCertResolver: TypeAlias = Callable[[str, str, int], str]


class _VerifiedChainSource(Protocol):
    def get_verified_chain(self) -> Sequence[_ssl.Certificate] | None: ...


def _verified_chain_der(tls: ssl.SSLSocket) -> tuple[bytes, ...]:
    if sys.version_info >= (3, 13):
        return tuple(tls.get_verified_chain())
    legacy: Final = cast(  # cast-ok: the stub omits _sslobj, the C object has get_verified_chain since 3.10
        "_VerifiedChainSource | None",
        tls._sslobj,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # public API only from 3.13
    )
    chain: Final = () if legacy is None else legacy.get_verified_chain() or ()
    return tuple(cert.public_bytes(_ssl.ENCODING_DER) for cert in chain)


def _server_trust_anchor(cafile: str, host: str, port: int) -> bytes | None:
    try:
        context: Final = ssl.create_default_context(cafile=cafile)
        with socket.create_connection((host, port), timeout=TLS_PROBE_TIMEOUT_SECONDS) as raw:
            raw.sendall(PG_SSL_REQUEST)
            if raw.recv(1) != b"S":
                return None
            with context.wrap_socket(raw, server_hostname=host) as tls:
                chain: Final = _verified_chain_der(tls)
    except (OSError, ValueError):
        return None
    return chain[-1] if chain else None


def pin_bundle_root(cert_path: str, host: str, port: int) -> str:
    """Reduce a multi-root CA bundle to the one root that verifies ``host``.

    Prisma's ``sslcert`` loads a single PEM certificate (native-tls
    ``Certificate::from_pem``), so pointing it at a bundle such as the AWS RDS
    global bundle trusts only the first of its 108 regional roots and the
    handshake fails with "unable to get local issuer certificate" for every
    other region. A single-certificate file is returned as is. For a bundle,
    one verifying handshake (chain and hostname, whole bundle as trust store)
    identifies the trust anchor the server actually chains to, which is
    written to a single-certificate file for Prisma. If the probe fails the
    bundle path is returned unchanged, so Prisma fails closed exactly as
    before rather than trusting anything the bundle would not.
    """
    try:
        if Path(cert_path).read_bytes().count(PEM_CERT_HEADER) < 2:
            return cert_path
    except OSError:
        return cert_path
    root: Final = _server_trust_anchor(cert_path, host, port)
    if root is None:
        return cert_path
    pinned: Final = Path(tempfile.gettempdir()) / f"litellm-sslcert-{hashlib.sha256(root).hexdigest()[:16]}.pem"
    return str(pinned) if _replace_file(pinned, ssl.DER_cert_to_PEM_cert(root)) else cert_path


def _replace_file(target: Path, content: str) -> bool:
    """Write ``content`` to a private temp file and rename it over ``target``, so
    readers never see a partial file and a symlink planted at ``target`` is
    replaced rather than followed."""
    try:
        fd, staged = tempfile.mkstemp(dir=target.parent, prefix=f"{target.name}.")
    except OSError:
        return False
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
        os.replace(staged, target)
    except OSError:
        Path(staged).unlink(missing_ok=True)
        return False
    return True


def translate_libpq_ssl_params(url: str, resolve_root_cert: RootCertResolver = pin_bundle_root) -> str:
    """Rewrite libpq's certificate-verification params into Prisma's dialect.

    Prisma's engine only knows ``sslmode=disable|prefer|require``, ``sslcert``
    (a single CA certificate) and ``sslaccept=strict``. It silently discards
    ``sslrootcert`` and downgrades ``sslmode=verify-ca`` / ``verify-full`` to
    ``prefer``, so a URL copied from libpq / RDS docs connects over TLS with no
    certificate check at all. ``verify-ca`` and ``verify-full`` both become
    ``require`` (Prisma has no CA-only mode), ``sslrootcert`` becomes
    ``sslcert`` (run through ``resolve_root_cert``, which pins a multi-root
    bundle down to the server's root), and either one turns on
    ``sslaccept=strict`` (chain and hostname), matching libpq where a root
    cert makes ``require`` verify. Prisma params the operator pinned
    themselves win; anything else is left untouched.
    """
    parsed: Final = urllib.parse.urlsplit(url)
    pairs: Final = tuple(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    keys: Final = frozenset(key for key, _ in pairs)
    wants_verify: Final = any(key == "sslmode" and value in LIBPQ_VERIFY_SSLMODES for key, value in pairs)
    if not wants_verify and "sslrootcert" not in keys:
        return url
    translated: Final = tuple(
        ("sslmode", "require") if key == "sslmode" and value in LIBPQ_VERIFY_SSLMODES else (key, value)
        for key, value in pairs
        if key != "sslrootcert"
    )
    root_cert: Final = tuple(
        ("sslcert", resolve_root_cert(value, parsed.hostname or "", parsed.port or int(DEFAULT_POSTGRES_PORT)))
        for key, value in pairs
        if key == "sslrootcert" and "sslcert" not in keys
    )
    strict: Final = () if "sslaccept" in keys else (("sslaccept", "strict"),)
    query: Final = urllib.parse.urlencode(translated + root_cert + strict)
    return urllib.parse.urlunsplit(parsed._replace(query=query))


def postgres_connection_budget_message(writer_limit: str, reader_limit: str | None, num_workers: str) -> str:
    """The startup line that states this pod's worst-case Postgres connection demand.

    Prisma's ``connection_limit`` is per query engine, and every uvicorn worker
    owns one engine per configured database (writer, plus the reader when
    ``DATABASE_URL_READ_REPLICA`` is set). The limits are read off the final URLs,
    so a reader that pins its own ``connection_limit`` or an operator override in
    ``database_extra_connection_params`` is counted at its real value. The
    server-side cap is shared by every pod, so the number an operator has to keep
    under ``max_connections`` minus ``superuser_reserved_connections`` is
    pods x workers x the per-worker sum, not one engine's limit.
    """
    try:
        workers: Final = max(1, int(num_workers))
        writer: Final = int(writer_limit)
        reader: Final = int(reader_limit) if reader_limit is not None else 0
    except ValueError:
        return (
            "LiteLLM Proxy: Postgres connection budget per pod = workers x (writer connection_limit + reader "
            f"connection_limit) (workers={num_workers!r}, writer={writer_limit!r}, reader={reader_limit!r}); "
            "keep pods x that figure under max_connections minus superuser_reserved_connections"
        )
    engines: Final = (
        f"(writer connection_limit {writer} + reader connection_limit {reader})"
        if reader_limit is not None
        else f"writer connection_limit {writer}"
    )
    per_pod: Final = workers * (writer + reader)
    return (
        f"LiteLLM Proxy: Postgres connection budget per pod = {workers} worker(s) x {engines} = up to {per_pod} "
        "connections; keep pods x that figure under max_connections minus superuser_reserved_connections"
    )


def reader_shareable_params(params: Mapping[str, str | int | float]) -> Mapping[str, str | int | float]:
    """Return the subset of ``params`` the read replica is allowed to inherit."""
    return MappingProxyType({key: value for key, value in params.items() if key in CONNECTION_PARAM_KEYS})


def connection_params_from_url(url: str) -> Mapping[str, str | int | float]:
    """Return the connection params on ``url`` that the read replica shares."""
    return reader_shareable_params(
        MappingProxyType({key: value for key, value in urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query)})
    )


# A re-minted token URL replaces a URL to the same database, so unlike the
# reader allowlist it may also carry ``options``: that is where the server-side
# timeouts (statement, lock, idle-in-transaction) live, and a refresh that
# dropped them would leave the replacement engine's sessions unbounded.
TOKEN_REFRESH_PARAM_KEYS: Final[frozenset[str]] = CONNECTION_PARAM_KEYS | PRISMA_TLS_PARAM_KEYS | frozenset({"options"})


def token_refresh_params_from_url(url: str) -> Mapping[str, str | int | float]:
    """Return the params a re-minted token URL carries over from the URL it replaces.

    The pool and timeout params, Prisma's TLS params (already translated from
    libpq spelling) and the ``options`` string, so a refreshed URL keeps verifying
    the server and bounding its sessions the way the first one did.
    """
    return MappingProxyType(
        {
            key: value
            for key, value in urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query)
            if key in TOKEN_REFRESH_PARAM_KEYS
        }
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
    max_idle_connection_lifetime: int | None = Field(
        default=None, validation_alias=MAX_IDLE_CONNECTION_LIFETIME_ENV_VAR
    )

    database_sslmode: str | None = Field(default=None, validation_alias=DATABASE_SSLMODE_ENV_VAR)
    database_sslrootcert: str | None = Field(default=None, validation_alias=DATABASE_SSLROOTCERT_ENV_VAR)

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

    def token_auth(self, *, read_replica: bool = False) -> DatabaseTokenAuth | None:
        """The token strategy the toggles ask for, or ``None`` for password auth.

        Raises ``RuntimeError`` when both toggles are on, since the password can only
        come from one source.
        """
        return build_database_token_auth(
            read_replica=read_replica,
            iam_token_db_auth=self.iam_token_db_auth,
            azure_postgresql_auth=self.azure_postgresql_auth,
        )

    def tls_params(self) -> Mapping[str, str]:
        """``sslmode`` / ``sslrootcert`` query params for every URL assembled from the discrete vars.

        A root cert on its own means ``verify-full``: under libpq's default
        ``prefer`` the CA would never be consulted, and PgBouncer would dial
        Postgres unverified with the bundle loaded.
        """
        sslmode: Final = self.database_sslmode or ("verify-full" if self.database_sslrootcert else None)
        return MappingProxyType(
            {
                key: value
                for key, value in (
                    ("sslmode", sslmode),
                    ("sslrootcert", self.database_sslrootcert),
                )
                if value
            }
        )

    def build_writer_url(self) -> str | None:
        """Return the writer URL to set, or ``None`` to leave it as-is.

        Raises ``RuntimeError`` (naming the offending vars) when token auth is
        enabled but a required field is missing — the proxy cannot recover
        from this and a clear startup error beats a Prisma connect failure.
        A ``DATABASE_URL`` the supervisor pointed at the in-container PgBouncer
        is kept even under token auth: the pooler renews the token upstream.
        """
        assembled: Final = self._assemble_writer_url()
        if assembled is None:
            return None
        return add_missing_query_params(assembled, self.tls_params())

    def _assemble_writer_url(self) -> str | None:
        auth: Final = self.token_auth()
        if auth is not None and database_url_is_pooled():
            return None
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
        assembled: Final = self._assemble_reader_url()
        if assembled is None:
            return None
        return add_missing_query_params(assembled, self.tls_params())

    def _assemble_reader_url(self) -> str | None:
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

        auth: Final = self.token_auth(read_replica=True)
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

        for env_var in ("DATABASE_URL", "DIRECT_URL"):
            url = os.environ.get(env_var)
            if url:
                os.environ[env_var] = translate_libpq_ssl_params(url)

        # DATABASE_DISABLE_PREPARED_STATEMENTS maps to Prisma's `pgbouncer=true`
        # URL param, same as the CLI's `database_disable_prepared_statements`
        # config key. An explicit `pgbouncer` value already on the URL wins.
        if self.disable_prepared_statements:
            for env_var in ("DATABASE_URL", "DIRECT_URL"):
                url = os.environ.get(env_var)
                if url:
                    os.environ[env_var] = add_missing_query_params(url, MappingProxyType({"pgbouncer": "true"}))

        lifetime_params: Final = idle_lifetime_params(self.max_idle_connection_lifetime)
        for env_var in ("DATABASE_URL", "DIRECT_URL"):
            url = os.environ.get(env_var)
            if url:
                os.environ[env_var] = add_missing_query_params(url, lifetime_params)

        # The reader inherits the writer's connection params (pool size, timeouts,
        # pgbouncer mode). Without this the reader pool ignores the configured cap
        # and falls back to Prisma's `num_physical_cpus * 2 + 1` default.
        reader_url: Final = self.build_reader_url() or self.database_url_read_replica
        if reader_url is not None:
            os.environ["DATABASE_URL_READ_REPLICA"] = add_missing_query_params(
                translate_libpq_ssl_params(reader_url),
                connection_params_from_url(os.environ.get("DATABASE_URL", "")),
            )

        return wrote_writer
