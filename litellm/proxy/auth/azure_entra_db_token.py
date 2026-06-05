"""Mint Microsoft Entra ID access tokens for Azure Database for PostgreSQL.

Azure counterpart to ``rds_iam_token.py``. Whereas AWS RDS mints a
host/port/user-scoped presigned IAM token, Azure issues a generic Microsoft
Entra ID access token for the Azure Database for PostgreSQL data-plane scope.
The access token is used as the Postgres *password*; the database username is
the Entra principal (managed identity / service principal) name.

Auth is resolved by ``azure-identity``'s ``DefaultAzureCredential``, so this
works keyless on AKS Microsoft Entra Workload Identity (the projected
federated token at ``AZURE_FEDERATED_TOKEN_FILE``), on Azure VMs / Container
Apps via managed identity, and locally via the Azure CLI — no static secret
required. The credential object is built once and reused so the Azure SDK's
internal token cache + silent refresh apply across calls.
"""

import threading
import urllib.parse
from typing import Any, Optional

# Microsoft Entra ID scope for the Azure Database for PostgreSQL / MySQL data
# plane. The issued access token is presented to Postgres as the password.
AZURE_DB_SCOPE = "https://ossrdbms-aad.database.windows.net/.default"

# Built once and reused: DefaultAzureCredential caches tokens internally and
# refreshes them silently, so a single instance avoids rebuilding the
# credential chain on every proactive token refresh. The lock guards the lazy
# init: the writer and reader refresh loops can mint concurrently, and the mint
# runs in a worker thread (via asyncio.to_thread), so two threads could
# otherwise build (and one silently discard) separate credentials.
_cached_credential: Optional[Any] = None
_credential_lock = threading.Lock()


def _get_default_credential() -> Any:
    global _cached_credential
    if _cached_credential is None:
        with _credential_lock:
            if _cached_credential is None:
                try:
                    from azure.identity import DefaultAzureCredential
                except ImportError as e:
                    raise ImportError(
                        "azure-identity is required for Azure Database for "
                        "PostgreSQL identity-token auth. Install it with: "
                        "pip install azure-identity"
                    ) from e
                _cached_credential = DefaultAzureCredential()
    return _cached_credential


def generate_azure_entra_db_token(
    db_host: Optional[str] = None,
    db_port: Optional[str] = None,
    db_user: Optional[str] = None,
    credential: Optional[Any] = None,
) -> str:
    """Return a URL-encoded Entra ID access token for use as the DB password.

    ``db_host`` / ``db_port`` / ``db_user`` are accepted for signature parity
    with ``rds_iam_token.generate_iam_auth_token`` but are not required: unlike
    an AWS RDS presigned token, an Entra access token is scoped to the identity
    and the OSS RDBMS audience, not to a specific host. ``credential`` may be
    injected (primarily for testing); otherwise a cached
    ``DefaultAzureCredential`` is used.
    """
    cred = credential if credential is not None else _get_default_credential()
    access_token = cred.get_token(AZURE_DB_SCOPE).token
    # Quote for parity/safety with the AWS path. Entra tokens are JWTs using the
    # URL-safe base64 alphabet, so quoting is effectively a no-op here, but it
    # guarantees the token can never corrupt the assembled postgresql:// URL.
    return urllib.parse.quote(access_token, safe="")
