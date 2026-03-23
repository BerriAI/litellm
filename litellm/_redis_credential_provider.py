import asyncio
from typing import Tuple


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


class GCPIAMCredentialProvider:
    """
    redis.credentials.CredentialProvider implementation that generates a fresh GCP IAM
    token on every new connection. This fixes the 1-hour token expiry issue for async
    Redis cluster clients, which previously generated the token once at startup and
    cached it as a static password.
    """

    def __init__(self, gcp_service_account: str) -> None:
        self._gcp_service_account = gcp_service_account

    def get_credentials(self) -> Tuple[str]:
        token = _generate_gcp_iam_access_token(self._gcp_service_account)
        return (token,)

    async def get_credentials_async(self) -> Tuple[str]:
        token = await asyncio.to_thread(
            _generate_gcp_iam_access_token, self._gcp_service_account
        )
        return (token,)
