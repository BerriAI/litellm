"""Tests for the database token auth strategies.

``litellm/proxy/db/token_auth.py`` decides where the proxy's Postgres password
comes from: an AWS RDS IAM token, a Microsoft Entra ID access token for Azure
Database for PostgreSQL, or neither. Minting and expiry parsing dispatch over
that union, so both variants are exercised here, together with the URL encoding
that lets an Entra principal (a UPN containing ``@``) survive being embedded in
a connection URL.
"""

import base64
import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from litellm.proxy.db.token_auth import (
    AZURE_POSTGRESQL_AUTH_ENV_VAR,
    AZURE_POSTGRESQL_SCOPE,
    IAM_TOKEN_DB_AUTH_ENV_VAR,
    AzureEntraTokenAuth,
    IAMEndpoint,
    RdsIamTokenAuth,
    build_azure_entra_token_provider,
    mint_database_token,
    parse_database_token_expiration,
    parse_iam_endpoint_from_url,
    resolve_database_token_auth,
)


def _entra_token(exp: int, *, header: str = "eyJhbGciOiJSUzI1NiJ9") -> str:
    """A JWT shaped like a real Entra access token, carrying ``exp``."""
    payload = base64.urlsafe_b64encode(
        json.dumps({"aud": "https://ossrdbms-aad.database.windows.net", "exp": exp}).encode()
    ).rstrip(b"=")
    return f"{header}.{payload.decode()}.c2lnbmF0dXJl"


def _endpoint(**overrides) -> IAMEndpoint:
    fields = {
        "host": "pg.postgres.database.azure.com",
        "port": "5432",
        "user": "litellm",
        "name": "litellm_db",
    }
    fields.update(overrides)
    return IAMEndpoint(**fields)


# ---------------------------------------------------------------------------
# Minting
# ---------------------------------------------------------------------------


def test_rds_mint_delegates_to_the_sigv4_token_generator():
    endpoint = _endpoint(host="writer.aurora.local", user="litellm_rds")

    with patch(
        "litellm.proxy.auth.rds_iam_token.generate_iam_auth_token",
        return_value="SIGV4_TOKEN",
    ) as generate:
        token = mint_database_token(RdsIamTokenAuth(), endpoint)

    assert token == "SIGV4_TOKEN"
    generate.assert_called_once_with(
        db_host="writer.aurora.local",
        db_port="5432",
        db_user="litellm_rds",
    )


def test_entra_mint_calls_the_injected_provider_and_encodes_the_token():
    """A real compact JWT is already URL-safe, but the provider is an Azure SDK call
    whose output we do not control, and an unencoded ``/`` or ``=`` in a password
    silently truncates the connection URL."""
    auth = AzureEntraTokenAuth(token_provider=lambda: "head.pay/load+x=.sig")

    assert mint_database_token(auth, _endpoint()) == "head.pay%2Fload%2Bx%3D.sig"


def test_entra_mint_asks_the_provider_every_time():
    """A refresh must get a new token, not a cached one from construction time."""
    tokens = iter(["first", "second"])
    auth = AzureEntraTokenAuth(token_provider=lambda: next(tokens))

    assert mint_database_token(auth, _endpoint()) == "first"
    assert mint_database_token(auth, _endpoint()) == "second"


# ---------------------------------------------------------------------------
# Expiry parsing
# ---------------------------------------------------------------------------


def test_rds_expiry_reads_the_sigv4_query_params():
    token = "writer.aurora.local:5432/?Action=connect&X-Amz-Date=20260820T101500Z&X-Amz-Expires=900"

    assert parse_database_token_expiration(RdsIamTokenAuth(), token) == datetime(2026, 8, 20, 10, 30, 0)


@pytest.mark.parametrize(
    "token",
    [
        "no-query-params",
        "host/?X-Amz-Date=20260820T101500Z",
        "host/?X-Amz-Expires=900",
        "host/?X-Amz-Date=not-a-date&X-Amz-Expires=900",
    ],
)
def test_rds_expiry_returns_none_when_unreadable(token):
    assert parse_database_token_expiration(RdsIamTokenAuth(), token) is None


@pytest.mark.parametrize("exp", [1787000000, 1787000001, 1787000012, 1787000123])
def test_entra_expiry_decodes_the_jwt_exp_claim(exp):
    """Parametrized over several ``exp`` values so the payload length lands on every
    base64 padding remainder: the JWT payload is stripped of its ``=`` padding and has
    to be re-padded before it can be decoded."""
    auth = AzureEntraTokenAuth(token_provider=lambda: "unused")

    parsed = parse_database_token_expiration(auth, _entra_token(exp))

    assert parsed is not None
    assert parsed.tzinfo is None
    assert parsed == datetime.fromtimestamp(exp, tz=timezone.utc).replace(tzinfo=None)


@pytest.mark.parametrize(
    "token",
    [
        "not-a-jwt",
        "only.two",
        "head.{}.sig",
        "head.bm90LWpzb24.sig",
        f"head.{base64.urlsafe_b64encode(b'{}').decode()}.sig",
        f"head.{base64.urlsafe_b64encode(json.dumps({'exp': 'soon'}).encode()).decode()}.sig",
    ],
)
def test_entra_expiry_returns_none_when_unreadable(token):
    """An unreadable expiry must degrade to the caller's fallback refresh interval
    rather than blowing up the refresh loop."""
    auth = AzureEntraTokenAuth(token_provider=lambda: "unused")

    assert parse_database_token_expiration(auth, token) is None


# ---------------------------------------------------------------------------
# URL building and parsing
# ---------------------------------------------------------------------------


def test_build_url_encodes_a_upn_user_and_the_schema():
    endpoint = _endpoint(user="litellm@contoso.onmicrosoft.com", name="litellm db", schema="app/schema")

    assert endpoint.build_url("TOKEN") == (
        "postgresql://litellm%40contoso.onmicrosoft.com:TOKEN"
        "@pg.postgres.database.azure.com:5432/litellm%20db?schema=app%2Fschema"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("user", "svc%40corp"),
        ("name", "litellm%20db"),
        ("schema", "app%2Fschema"),
    ],
)
def test_build_url_leaves_an_already_encoded_component_alone(field, value):
    """RDS IAM auth interpolated these raw, so pre-encoding was the only way to get an
    ``@`` into ``DATABASE_USER``. Encoding again turns ``svc%40corp`` into
    ``svc%2540corp``, which Postgres rejects with ``User `svc%40corp` was denied
    access``, so an operator who did that on RDS breaks on upgrade."""
    url = _endpoint(**{field: value}).build_url("TOKEN")

    assert value in url
    assert "%25" not in url


def test_build_url_inserts_the_token_verbatim():
    """Both providers hand the token back already in wire form, so re-encoding it here
    would double-escape the password."""
    rds_token = "writer.aurora.local%3A5432%2F%3FAction%3Dconnect%26X-Amz-Date%3D20260820T101500Z"

    assert _endpoint().build_url(rds_token) == (
        f"postgresql://litellm:{rds_token}@pg.postgres.database.azure.com:5432/litellm_db"
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        IAMEndpoint(host="h.example.com", port="5432", user="litellm", name="litellm_db"),
        IAMEndpoint(host="h.example.com", port="6543", user="litellm@contoso.com", name="db", schema="public"),
        IAMEndpoint(host="h.example.com", port="5432", user="u", name="litellm db", schema="app schema"),
    ],
)
def test_build_url_and_parse_round_trip(endpoint):
    assert parse_iam_endpoint_from_url(endpoint.build_url("TOKEN")) == endpoint


def test_parse_leaves_an_already_escaped_schema_alone():
    """``parse_qs`` unquotes query values itself, so unquoting again here would turn a
    schema that legitimately contains ``%40`` into one containing ``@``."""
    url = "postgresql://u:TOKEN@h.example.com:5432/db?schema=raw%2540schema"

    assert parse_iam_endpoint_from_url(url).schema == "raw%40schema"


# ---------------------------------------------------------------------------
# Strategy resolution from the environment
# ---------------------------------------------------------------------------


def test_resolve_returns_none_when_neither_toggle_is_set(monkeypatch):
    monkeypatch.delenv(IAM_TOKEN_DB_AUTH_ENV_VAR, raising=False)
    monkeypatch.delenv(AZURE_POSTGRESQL_AUTH_ENV_VAR, raising=False)

    assert resolve_database_token_auth() is None


def test_resolve_returns_the_rds_strategy(monkeypatch):
    monkeypatch.setenv(IAM_TOKEN_DB_AUTH_ENV_VAR, "true")
    monkeypatch.delenv(AZURE_POSTGRESQL_AUTH_ENV_VAR, raising=False)

    assert resolve_database_token_auth() == RdsIamTokenAuth()


def test_resolve_returns_the_entra_strategy(monkeypatch):
    monkeypatch.delenv(IAM_TOKEN_DB_AUTH_ENV_VAR, raising=False)
    monkeypatch.setenv(AZURE_POSTGRESQL_AUTH_ENV_VAR, "true")

    with patch(
        "litellm.secret_managers.get_azure_ad_token_provider.get_azure_ad_token_provider",
        return_value=lambda: "ENTRA_TOKEN",
    ):
        auth = resolve_database_token_auth()

    assert isinstance(auth, AzureEntraTokenAuth)
    assert auth.token_provider() == "ENTRA_TOKEN"


def test_resolve_raises_when_both_toggles_are_set(monkeypatch):
    monkeypatch.setenv(IAM_TOKEN_DB_AUTH_ENV_VAR, "true")
    monkeypatch.setenv(AZURE_POSTGRESQL_AUTH_ENV_VAR, "true")

    with pytest.raises(RuntimeError, match="can only come from one token source"):
        resolve_database_token_auth()


def test_entra_provider_uses_the_ossrdbms_scope():
    """The wrong scope mints a token Azure Postgres rejects, so the scope is pinned."""
    with patch(
        "litellm.secret_managers.get_azure_ad_token_provider.get_azure_ad_token_provider",
        return_value=lambda: "ENTRA_TOKEN",
    ) as get_provider:
        build_azure_entra_token_provider()

    get_provider.assert_called_once_with(azure_scope="https://ossrdbms-aad.database.windows.net/.default")
    assert AZURE_POSTGRESQL_SCOPE == "https://ossrdbms-aad.database.windows.net/.default"


# ---------------------------------------------------------------------------
# Toggle parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["true", "TRUE", " True ", "1", "yes", "y", "on", "t"])
def test_every_truthy_spelling_enables_token_auth(monkeypatch, value):
    """The settings model reads these toggles with pydantic (which accepts all of these)
    while the refresh loop reads them here. When the two disagreed, `AZURE_POSTGRESQL_AUTH=1`
    minted a token at startup and then never refreshed it, so the proxy died an hour in."""
    monkeypatch.setenv(AZURE_POSTGRESQL_AUTH_ENV_VAR, value)
    monkeypatch.delenv(IAM_TOKEN_DB_AUTH_ENV_VAR, raising=False)

    with patch(
        "litellm.secret_managers.get_azure_ad_token_provider.get_azure_ad_token_provider",
        return_value=lambda: "ENTRA_TOKEN",
    ):
        assert isinstance(resolve_database_token_auth(), AzureEntraTokenAuth)


@pytest.mark.parametrize("value", ["", "  ", "false", "False", "0", "no", "off", "F", "N"])
def test_falsy_spellings_leave_token_auth_off(monkeypatch, value):
    """An empty string is how a Kubernetes manifest spells 'off'."""
    monkeypatch.setenv(AZURE_POSTGRESQL_AUTH_ENV_VAR, value)
    monkeypatch.setenv(IAM_TOKEN_DB_AUTH_ENV_VAR, value)

    assert resolve_database_token_auth() is None


@pytest.mark.parametrize("env_var", [IAM_TOKEN_DB_AUTH_ENV_VAR, AZURE_POSTGRESQL_AUTH_ENV_VAR])
@pytest.mark.parametrize("value", ["enabled", "maybe", "TRUEE", "2"])
def test_an_unreadable_toggle_is_a_startup_error(monkeypatch, env_var, value):
    """Reading a typo as 'off' would quietly downgrade an operator who asked for token
    auth to password auth, and the first sign of it is the server refusing the
    connection. Pydantic rejected these before token auth had its own parser."""
    monkeypatch.setenv(env_var, value)
    monkeypatch.delenv(
        AZURE_POSTGRESQL_AUTH_ENV_VAR if env_var == IAM_TOKEN_DB_AUTH_ENV_VAR else IAM_TOKEN_DB_AUTH_ENV_VAR,
        raising=False,
    )

    with pytest.raises(ValueError, match=env_var) as raised:
        resolve_database_token_auth()

    assert value in str(raised.value)


def test_the_entra_provider_is_built_once_per_process():
    """Each build is another Azure credential with its own transport and token cache
    that nothing closes, and the writer, the reader, and the refresh loop each ask."""
    with patch(
        "litellm.secret_managers.get_azure_ad_token_provider.get_azure_ad_token_provider",
        return_value=lambda: "ENTRA_TOKEN",
    ) as get_provider:
        assert build_azure_entra_token_provider() is build_azure_entra_token_provider()

    get_provider.assert_called_once()


def test_an_unparseable_rds_expiry_degrades_instead_of_raising():
    """This runs inside `PrismaWrapper.__getattr__`, so anything it raises turns every
    database call into that error."""
    absurd = "https://host/?X-Amz-Date=20260820T101500Z&X-Amz-Expires=99999999999999999999"

    assert parse_database_token_expiration(RdsIamTokenAuth(), absurd) is None
