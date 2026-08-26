import asyncio
import threading
import time
from typing import Any, Final, Protocol
from urllib.parse import quote

from redis.credentials import CredentialProvider

# Azure AD scope for Redis Cache for Azure.
AZURE_REDIS_SCOPE: Final = "https://redis.azure.com/.default"

# GCP IAM tokens are valid for 1 hour. Cache for 55 minutes to refresh before expiry.
_GCP_IAM_TOKEN_TTL_SECONDS: Final = 3300

# Module-level cache shared across all GCPIAMCredentialProvider instances for the
# same service account, so multiple Redis connections on the same pod share one token.
# Keyed by service_account → (token, expiry_monotonic_timestamp).
_token_cache: Final[dict[str, tuple[str, float]]] = {}
_token_cache_lock: Final = threading.Lock()


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

    client: Final = iam_credentials_v1.IAMCredentialsClient()
    request: Final = iam_credentials_v1.GenerateAccessTokenRequest(
        name=service_account,
        scope=["https://www.googleapis.com/auth/cloud-platform"],
    )
    response: Final = client.generate_access_token(request=request)
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

    def get_credentials(self) -> tuple[str]:
        token: Final = _get_cached_gcp_iam_token(self._gcp_service_account)
        return (token,)

    async def get_credentials_async(self) -> tuple[str]:
        token: Final = await asyncio.to_thread(_get_cached_gcp_iam_token, self._gcp_service_account)
        return (token,)


_ELASTICACHE_SERVICE_NAME: Final = "elasticache"
_ELASTICACHE_TOKEN_TTL_SECONDS: Final = 900


class _FrozenBotocoreCredentials(Protocol):
    access_key: str
    secret_key: str
    token: str | None


class _BotocoreCredentials(Protocol):
    def get_frozen_credentials(self) -> _FrozenBotocoreCredentials: ...


class _BotocoreCredentialsResolver(Protocol):
    def __call__(self) -> _BotocoreCredentials | None: ...


class ElastiCacheIAMCredentialProvider(CredentialProvider):
    def __init__(
        self,
        user_name: str,
        cache_name: str,
        region: str,
        credentials_resolver: _BotocoreCredentialsResolver | None = None,
        token_lifetime_seconds: int = _ELASTICACHE_TOKEN_TTL_SECONDS,
    ) -> None:
        self._user_name = user_name
        self._cache_name = cache_name
        self._region = region
        self._credentials_resolver = credentials_resolver or self._resolve_credentials
        self._credentials: _BotocoreCredentials | None = None
        self._token_lifetime_seconds = token_lifetime_seconds

    @staticmethod
    def _resolve_credentials() -> Any:
        try:
            import botocore.session
        except ImportError as e:
            raise ImportError(
                "botocore is required for ElastiCache IAM Redis authentication. Install it with: pip install boto3"
            ) from e

        return botocore.session.get_session().get_credentials()

    def _get_credentials(self) -> tuple[str, str]:
        credentials: Final = self._credentials if self._credentials is not None else self._credentials_resolver()
        if credentials is None:
            raise RuntimeError("Unable to resolve AWS credentials for ElastiCache IAM Redis authentication")
        self._credentials = credentials

        frozen_credentials: Final = credentials.get_frozen_credentials()
        if frozen_credentials is None:
            raise RuntimeError("Unable to resolve AWS credentials for ElastiCache IAM Redis authentication")

        try:
            from botocore.auth import SigV4QueryAuth
            from botocore.awsrequest import AWSRequest
        except ImportError as e:
            raise ImportError(
                "botocore is required for ElastiCache IAM Redis authentication. Install it with: pip install boto3"
            ) from e

        request: Final = AWSRequest(
            method="GET",
            url=(f"https://{self._cache_name}/?Action=connect&User={quote(self._user_name, safe='')}"),
        )
        SigV4QueryAuth(
            frozen_credentials,
            _ELASTICACHE_SERVICE_NAME,
            self._region,
            expires=self._token_lifetime_seconds,
        ).add_auth(request)
        return self._user_name, request.url.removeprefix("https://")

    def get_credentials(self) -> tuple[str, str]:
        return self._get_credentials()

    async def get_credentials_async(self) -> tuple[str, str]:
        return await asyncio.to_thread(self._get_credentials)


class AzureADCredentialProvider(CredentialProvider):
    """
    redis.credentials.CredentialProvider implementation that supplies Azure AD
    tokens for Redis authentication.

    Wraps an azure-identity credential object so the Azure SDK's internal token
    cache and silent refresh are honoured on every Redis connection. This avoids
    the static-token-baked-in-pool issue where pool-managed connections would
    fail authentication after the initial token expired (~1 hour TTL).
    """

    def __init__(self, credential: Any, username: str | None = None) -> None:
        self._credential = credential
        self._username = username

    def get_credentials(self) -> tuple[str] | tuple[str, str]:
        token: Final = self._credential.get_token(AZURE_REDIS_SCOPE).token
        if self._username:
            return (self._username, token)
        return (token,)

    async def get_credentials_async(self) -> tuple[str] | tuple[str, str]:
        token_obj: Final = await asyncio.to_thread(self._credential.get_token, AZURE_REDIS_SCOPE)
        if self._username:
            return (self._username, token_obj.token)
        return (token_obj.token,)
