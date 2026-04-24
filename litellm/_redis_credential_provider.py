import asyncio
import threading
import time
from typing import Dict, Tuple

from redis.credentials import CredentialProvider  # type: ignore[attr-defined]

# GCP IAM tokens are valid for 1 hour. Cache for 55 minutes to refresh before expiry.
_GCP_IAM_TOKEN_TTL_SECONDS = 3300

# Module-level cache shared across all GCPIAMCredentialProvider instances for the
# same service account, so multiple Redis connections on the same pod share one token.
# Keyed by service_account → (token, expiry_monotonic_timestamp).
_token_cache: Dict[str, Tuple[str, float]] = {}
_token_cache_lock = threading.Lock()


def _generate_gcp_iam_access_token(service_account: str) -> str:
    """
    Generate GCP IAM access token for Redis authentication.

    Args:
        service_account: GCP service account in format 'projects/-/serviceAccounts/name@project.iam.gserviceaccount.com'

    Returns:
        Access token string for GCP IAM authentication
    """
    try:
        from google.cloud import iam_credentials_v1
    except ImportError:
        raise ImportError(
            "google-cloud-iam is required for GCP IAM Redis authentication. "
            "Install it with: pip install google-cloud-iam"
        )

    client = iam_credentials_v1.IAMCredentialsClient()
    request = iam_credentials_v1.GenerateAccessTokenRequest(
        name=service_account,
        scope=["https://www.googleapis.com/auth/cloud-platform"],
    )
    response = client.generate_access_token(request=request)
    return str(response.access_token)


def _get_cached_gcp_iam_token(service_account: str) -> str:
    """
    Return a cached GCP IAM token, refreshing only when expired.

    Uses a module-level cache shared across all GCPIAMCredentialProvider
    instances for the same service account. The threading.Lock ensures only
    one thread performs the network round-trip on expiry; all others wait
    briefly and read the fresh token (double-checked locking pattern).

    This avoids N concurrent blocking IAM refreshes when N Redis connections
    are established simultaneously (e.g. during health checks or pool warm-up),
    which would otherwise serialise inside Python's async event loop and cause
    cascading request latency.
    """
    cached = _token_cache.get(service_account)
    if cached is not None:
        token, expiry = cached
        if time.monotonic() < expiry:
            return token

    with _token_cache_lock:
        # Re-check inside the lock: another thread may have refreshed already.
        cached = _token_cache.get(service_account)
        if cached is not None:
            token, expiry = cached
            if time.monotonic() < expiry:
                return token

        token = _generate_gcp_iam_access_token(service_account)
        _token_cache[service_account] = (
            token,
            time.monotonic() + _GCP_IAM_TOKEN_TTL_SECONDS,
        )
        return token


class GCPIAMCredentialProvider(CredentialProvider):
    """
    redis.credentials.CredentialProvider implementation that supplies GCP IAM tokens
    for Redis authentication, with module-level caching per service account.

    Tokens are cached for _GCP_IAM_TOKEN_TTL_SECONDS (55 min) so that repeated
    connection establishments — e.g. during connection pool warm-up or health checks —
    do not each trigger a synchronous network round-trip that would block Python's
    async event loop and cause cascading request latency.
    """

    def __init__(self, gcp_service_account: str) -> None:
        self._gcp_service_account = gcp_service_account

    def get_credentials(self) -> Tuple[str]:
        token = _get_cached_gcp_iam_token(self._gcp_service_account)
        return (token,)

    async def get_credentials_async(self) -> Tuple[str]:
        token = await asyncio.to_thread(
            _get_cached_gcp_iam_token, self._gcp_service_account
        )
        return (token,)
