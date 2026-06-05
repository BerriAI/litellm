"""Provider dispatch for database identity-token authentication.

LiteLLM's proxy can authenticate to its Postgres database with a short-lived,
auto-refreshed identity token instead of a static password (gated on the
``IAM_TOKEN_DB_AUTH`` flag). Two token providers are supported:

  * ``aws``   — AWS RDS / Aurora IAM auth
    (:mod:`litellm.proxy.auth.rds_iam_token`)
  * ``azure`` — Microsoft Entra ID for Azure Database for PostgreSQL
    (:mod:`litellm.proxy.auth.azure_entra_db_token`)

The provider is auto-detected from the database host — Azure Database for
PostgreSQL endpoints always end in ``.postgres.database.azure.com``; anything
else (e.g. AWS RDS/Aurora) uses the default AWS RDS IAM provider. Every
DB-token mint site (CLI startup, the componentized ``DatabaseURLSettings``
entrypoints, and the runtime ``PrismaWrapper`` refresh loop) routes through
:func:`generate_db_iam_token`, so the same provider is used consistently on
first connect and on every refresh.
"""

import os
import urllib.parse
from typing import Optional

DB_IAM_PROVIDER_AWS = "aws"
DB_IAM_PROVIDER_AZURE = "azure"

# Azure Database for PostgreSQL Flexible Server endpoints always end in this
# suffix (private-endpoint FQDNs included), so the host unambiguously
# identifies the cloud — no extra env var needed to select the provider.
_AZURE_DB_HOST_SUFFIX = ".postgres.database.azure.com"


def get_db_iam_auth_provider(db_host: Optional[str] = None) -> str:
    """Detect the DB IAM token provider from the database host.

    Returns ``azure`` for an Azure Database for PostgreSQL endpoint
    (``*.postgres.database.azure.com``), otherwise ``aws`` (the default — AWS
    RDS / Aurora IAM). ``db_host`` falls back to the ``DATABASE_HOST`` env var
    when not provided.
    """
    host = (db_host or os.getenv("DATABASE_HOST") or "").lower()
    if _AZURE_DB_HOST_SUFFIX in host:
        return DB_IAM_PROVIDER_AZURE
    return DB_IAM_PROVIDER_AWS


def generate_db_iam_token(
    db_host: Optional[str] = None,
    db_port: Optional[str] = None,
    db_user: Optional[str] = None,
) -> str:
    """Mint a DB auth token for the detected provider.

    Returns a URL-quoted token suitable for use as the Postgres password in a
    ``postgresql://user:<token>@host:port/name`` URL.
    """
    if get_db_iam_auth_provider(db_host) == DB_IAM_PROVIDER_AZURE:
        from litellm.proxy.auth.azure_entra_db_token import (
            generate_azure_entra_db_token,
        )

        return generate_azure_entra_db_token(
            db_host=db_host, db_port=db_port, db_user=db_user
        )

    # Default: AWS RDS / Aurora IAM auth.
    from litellm.proxy.auth.rds_iam_token import generate_iam_auth_token

    return generate_iam_auth_token(db_host=db_host, db_port=db_port, db_user=db_user)


def build_postgres_url(
    *,
    user: Optional[str],
    token: str,
    host: Optional[str],
    port: str,
    name: Optional[str],
    schema: Optional[str] = None,
) -> str:
    """Assemble a ``postgresql://`` URL for IAM / identity-token DB auth.

    Single source of truth for all DB-IAM URL assembly (CLI startup, the
    ``DatabaseURLSettings`` writer/reader, and the ``PrismaWrapper`` refresh),
    so the encoding can't drift between connect and refresh, or between the
    writer and reader endpoints.

    URL-encodes the principal (``user``), database (``name``), and ``schema``,
    so Azure Entra user principals (UPNs containing ``@``) and other reserved
    characters can't corrupt the URL. This is a no-op for conventional AWS RDS
    IAM identifiers. ``token`` is assumed already URL-encoded by the provider
    mint.
    """
    user_q = urllib.parse.quote(str(user), safe="")
    name_q = urllib.parse.quote(str(name), safe="")
    url = f"postgresql://{user_q}:{token}@{host}:{port}/{name_q}"
    if schema:
        url += f"?schema={urllib.parse.quote(schema, safe='')}"
    return url
