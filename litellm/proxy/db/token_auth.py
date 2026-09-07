"""Token-based authentication for the proxy's Postgres connection.

Two managed Postgres offerings hand the client a short-lived credential that is used as
the Postgres password: AWS RDS with IAM auth, and Azure Database for PostgreSQL Flexible
Server with Microsoft Entra ID. Both need the same machinery (mint at startup, read the
expiry back off the token, mint again before it lapses) and differ only in how the token
is produced and how its expiry is encoded, so the difference lives in a tagged union that
is resolved once from the environment and injected into whatever needs a token.
"""

import base64
import functools
import os
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final, TypeAlias

from pydantic import BaseModel
from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger

IAM_TOKEN_DB_AUTH_ENV_VAR: Final = "IAM_TOKEN_DB_AUTH"
AZURE_POSTGRESQL_AUTH_ENV_VAR: Final = "AZURE_POSTGRESQL_AUTH"
AZURE_POSTGRESQL_SCOPE: Final = "https://ossrdbms-aad.database.windows.net/.default"

CONFLICTING_TOKEN_AUTH_MESSAGE: Final = (
    f"{IAM_TOKEN_DB_AUTH_ENV_VAR} and {AZURE_POSTGRESQL_AUTH_ENV_VAR} are both enabled, but the "
    "database password can only come from one token source. Keep "
    f"{IAM_TOKEN_DB_AUTH_ENV_VAR} for AWS RDS IAM auth, or {AZURE_POSTGRESQL_AUTH_ENV_VAR} for "
    "Azure Database for PostgreSQL with Microsoft Entra ID, and unset the other one."
)

DEFAULT_POSTGRES_PORT: Final = "5432"

TRUTHY_TOKEN_AUTH_VALUES: Final[frozenset[str]] = frozenset({"1", "on", "t", "true", "y", "yes"})
FALSY_TOKEN_AUTH_VALUES: Final[frozenset[str]] = frozenset({"", "0", "f", "false", "n", "no", "off"})


def token_auth_flag_enabled(value: str | bool | None, *, env_var: str) -> bool:
    """Whether a token-auth toggle is on, rejecting anything it cannot read.

    The single parser for both toggles. Every entry point (the settings model, the
    CLI, and the refresh loop's own env lookup) routes through this, so a value like
    ``"1"`` cannot enable minting in one place and leave the refresh loop convinced
    token auth is off, which would strand a pod on a token it never renews.

    A value that is neither recognizably on nor recognizably off raises: silently
    reading a typo as off would downgrade an operator from token auth to password
    auth, and the first sign of it would be a connection refused by the server.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    normalized: Final = value.strip().lower()
    if normalized in TRUTHY_TOKEN_AUTH_VALUES:
        return True
    if normalized in FALSY_TOKEN_AUTH_VALUES:
        return False
    raise ValueError(
        f"{env_var}={value!r} is not a recognized boolean. Set it to one of "
        f"{', '.join(sorted(TRUTHY_TOKEN_AUTH_VALUES))} to turn it on, or to one of "
        f"{', '.join(sorted(v for v in FALSY_TOKEN_AUTH_VALUES if v))} to turn it off."
    )


def _quote(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def _normalize_quote(value: str) -> str:
    """Percent-encode a URL component that may already be percent-encoded.

    ``DATABASE_USER`` used to be interpolated raw, so pre-encoding was the only way to
    put an ``@`` in it. Encoding such a value again would double-escape it, so decode
    first: the round trip is idempotent and leaves an already-encoded value byte for
    byte as it was, while a raw UPN like ``svc@corp`` still comes out encoded.
    """
    return urllib.parse.quote(urllib.parse.unquote(value), safe="")


@dataclass(frozen=True, slots=True)
class IAMEndpoint:
    """Static parts of a token-authenticated Postgres connection.

    The token rotates every few minutes to an hour depending on the provider;
    everything else (host, port, user, database name, schema) stays fixed. Capturing
    the static fields once means a refresh only regenerates the token and reassembles
    the URL.
    """

    host: str
    port: str
    user: str
    name: str
    schema: str | None = None

    def build_url(self, token: str) -> str:
        """Assemble the connection URL, inserting ``token`` verbatim as the password.

        User, database name, and schema are normalized rather than encoded outright,
        because an Entra principal is a UPN containing ``@`` while an operator on the
        older RDS path may already have encoded that ``@`` themselves. The token is
        left alone: both providers hand it back already in wire form, and re-encoding
        it would double-escape the password.
        """
        base: Final = (
            f"postgresql://{_normalize_quote(self.user)}:{token}@{self.host}:{self.port}/{_normalize_quote(self.name)}"
        )
        if not self.schema:
            return base
        return f"{base}?schema={_normalize_quote(self.schema)}"


def parse_iam_endpoint_from_url(url: str) -> IAMEndpoint:
    """Parse an :class:`IAMEndpoint` back out of a Postgres URL.

    Used so a reader URL can drive its own token refresh without requiring callers to
    set parallel ``DATABASE_HOST_READ_REPLICA`` / etc. env vars.
    """
    parsed: Final = urllib.parse.urlparse(url)
    if not parsed.hostname or not parsed.username:
        raise ValueError("Cannot parse IAM endpoint from URL: missing host or username")
    name: Final = urllib.parse.unquote((parsed.path or "/").lstrip("/"))
    if not name:
        raise ValueError("Cannot parse IAM endpoint from URL: missing database name")
    port: Final = str(parsed.port) if parsed.port else DEFAULT_POSTGRES_PORT
    schema_values: Final = urllib.parse.parse_qs(parsed.query).get("schema") if parsed.query else None
    return IAMEndpoint(
        host=parsed.hostname,
        port=port,
        user=urllib.parse.unquote(parsed.username),
        name=name,
        schema=schema_values[0] if schema_values else None,
    )


@dataclass(frozen=True, slots=True)
class RdsIamTokenAuth:
    """AWS RDS IAM auth: a SigV4-presigned token minted from the ambient AWS credentials."""

    @property
    def label(self) -> str:
        return "RDS IAM token"

    @property
    def env_var(self) -> str:
        return IAM_TOKEN_DB_AUTH_ENV_VAR


@dataclass(frozen=True, slots=True)
class AzureEntraTokenAuth:
    """Azure Database for PostgreSQL auth: a Microsoft Entra ID access token as the password.

    The provider is injected rather than resolved here so callers (and tests) decide which
    Azure credential mints the token.
    """

    token_provider: Callable[[], str]

    @property
    def label(self) -> str:
        return "Azure Entra token"

    @property
    def env_var(self) -> str:
        return AZURE_POSTGRESQL_AUTH_ENV_VAR


DatabaseTokenAuth: TypeAlias = RdsIamTokenAuth | AzureEntraTokenAuth


def mint_database_token(auth: DatabaseTokenAuth, endpoint: IAMEndpoint) -> str:
    """Mint a fresh database password for ``endpoint``, already percent-encoded."""
    match auth:
        case RdsIamTokenAuth():
            from litellm.proxy.auth.rds_iam_token import generate_iam_auth_token

            return generate_iam_auth_token(db_host=endpoint.host, db_port=endpoint.port, db_user=endpoint.user)
        case AzureEntraTokenAuth():
            return _quote(auth.token_provider())
        case _:
            assert_never(auth)


def parse_database_token_expiration(auth: DatabaseTokenAuth, token: str) -> datetime | None:
    """Return when ``token`` expires as a naive UTC datetime, or None when unreadable.

    Callers fall back to a fixed refresh interval on None, so an unparseable token
    degrades to periodic refresh instead of failing.
    """
    match auth:
        case RdsIamTokenAuth():
            return _parse_rds_token_expiration(token)
        case AzureEntraTokenAuth():
            return _parse_entra_token_expiration(token)
        case _:
            assert_never(auth)


def _parse_rds_token_expiration(token: str) -> datetime | None:
    if "?" not in token:
        return None
    try:
        params: Final = urllib.parse.parse_qs(token.split("?", 1)[1])
        expires_values: Final = params.get("X-Amz-Expires")
        date_values: Final = params.get("X-Amz-Date")
        if not expires_values or not date_values:
            return None
        created: Final = datetime.strptime(date_values[0], "%Y%m%dT%H%M%SZ")
        return created + timedelta(seconds=int(expires_values[0]))
    except (ValueError, OverflowError, OSError) as exc:
        verbose_proxy_logger.debug("Failed to parse RDS IAM token expiration: %s", exc)
        return None


class _EntraAccessTokenClaims(BaseModel):
    exp: int


def _parse_entra_token_expiration(token: str) -> datetime | None:
    segments: Final = token.split(".")
    if len(segments) != 3:
        return None
    payload: Final = segments[1]
    try:
        claims: Final = _EntraAccessTokenClaims.model_validate_json(
            base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        )
    except ValueError as exc:
        verbose_proxy_logger.debug("Failed to parse Azure Entra token expiration: %s", exc)
        return None
    return datetime.fromtimestamp(claims.exp, tz=timezone.utc).replace(tzinfo=None)


@functools.cache
def build_azure_entra_token_provider() -> Callable[[], str]:
    """The process-wide Entra token provider for the Azure Postgres OSS RDBMS scope.

    Cached because the writer URL, the reader URL, and the refresh loop each ask for a
    strategy, and every uncached call would build another Azure credential with its own
    HTTP transport and its own token cache that nothing ever closes.
    """
    from litellm.secret_managers.get_azure_ad_token_provider import (
        get_azure_ad_token_provider,
    )

    return get_azure_ad_token_provider(azure_scope=AZURE_POSTGRESQL_SCOPE)


def build_database_token_auth(*, iam_token_db_auth: bool, azure_postgresql_auth: bool) -> DatabaseTokenAuth | None:
    """Pick the token strategy the two toggles ask for, or None when neither is on."""
    if iam_token_db_auth and azure_postgresql_auth:
        raise RuntimeError(CONFLICTING_TOKEN_AUTH_MESSAGE)
    if azure_postgresql_auth:
        return AzureEntraTokenAuth(token_provider=build_azure_entra_token_provider())
    if iam_token_db_auth:
        return RdsIamTokenAuth()
    return None


def resolve_database_token_auth() -> DatabaseTokenAuth | None:
    """Resolve the token strategy from the environment, raising when both toggles are set."""
    return build_database_token_auth(
        iam_token_db_auth=token_auth_flag_enabled(
            os.getenv(IAM_TOKEN_DB_AUTH_ENV_VAR), env_var=IAM_TOKEN_DB_AUTH_ENV_VAR
        ),
        azure_postgresql_auth=token_auth_flag_enabled(
            os.getenv(AZURE_POSTGRESQL_AUTH_ENV_VAR), env_var=AZURE_POSTGRESQL_AUTH_ENV_VAR
        ),
    )
