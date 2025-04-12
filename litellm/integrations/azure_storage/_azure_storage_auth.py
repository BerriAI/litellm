import base64
import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Optional
from httpx import Request
from litellm._logging import verbose_logger
from urllib.parse import urlparse, parse_qs

from litellm.llms.azure.common_utils import get_azure_ad_token_from_entrata_id


def _get_now_utc_str():
    return datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")


class AzureAuthSharedKey:
    def __init__(
        self,
        account_name: str,
        account_key: str,
        get_date_str: Callable[[], str] = _get_now_utc_str,
    ):
        self.account_name = account_name
        try:
            self.account_key = base64.b64decode(account_key)
        except Exception:
            raise ValueError(
                f"Invalid account key: '{account_key}' - must be a valid base64 encoded string"
            )
        self.get_date_str = get_date_str

    def __call__(self, request: Request) -> Request:
        return sign_httpx_request_with_shared_key(
            request, self.account_name, self.account_key, self.get_date_str
        )


class AzureADTokenAuth:
    def __init__(self, tenant_id: str, client_id: str, client_secret: str):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        # Internal variables used for Token based authentication
        self.azure_auth_token: Optional[
            str
        ] = None  # the Azure AD token to use for Azure Storage API requests
        self.token_expiry: Optional[
            datetime
        ] = None  # the expiry time of the currentAzure AD token

    def __call__(self, request: Request) -> Request:
        self.set_valid_azure_ad_token()
        if not self.azure_auth_token:
            raise ValueError("Failed to get Azure AD token")
        request.headers["Authorization"] = f"Bearer {self.azure_auth_token}"
        return request

    ####### Helper methods to managing Authentication to Azure Storage #######
    ##########################################################################

    def set_valid_azure_ad_token(self):
        """
        Wrapper to set self.azure_auth_token to a valid Azure AD token, refreshing if necessary

        Refreshes the token when:
        - Token is expired
        - Token is not set
        """
        # Check if token needs refresh
        if self._azure_ad_token_is_expired() or self.azure_auth_token is None:
            verbose_logger.debug("Azure AD token needs refresh")
            self.azure_auth_token = self.get_azure_ad_token_from_azure_storage(
                tenant_id=self.tenant_id,
                client_id=self.client_id,
                client_secret=self.client_secret,
            )
            # Token typically expires in 1 hour
            self.token_expiry = datetime.now() + timedelta(hours=1)
            verbose_logger.debug(f"New token will expire at {self.token_expiry}")

    def get_azure_ad_token_from_azure_storage(
        self,
        tenant_id: Optional[str],
        client_id: Optional[str],
        client_secret: Optional[str],
    ) -> str:
        """
        Gets Azure AD token to use for Azure Storage API requests
        """
        verbose_logger.debug("Getting Azure AD Token from Azure Storage")
        verbose_logger.debug(
            "tenant_id %s, client_id %s, client_secret %s",
            tenant_id,
            client_id,
            client_secret,
        )
        if tenant_id is None:
            raise ValueError(
                "Missing required environment variable: AZURE_STORAGE_TENANT_ID"
            )
        if client_id is None:
            raise ValueError(
                "Missing required environment variable: AZURE_STORAGE_CLIENT_ID"
            )
        if client_secret is None:
            raise ValueError(
                "Missing required environment variable: AZURE_STORAGE_CLIENT_SECRET"
            )

        token_provider = get_azure_ad_token_from_entrata_id(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
            scope="https://storage.azure.com/.default",
        )
        token = token_provider()

        verbose_logger.debug("azure auth token %s", token)

        return token

    def _azure_ad_token_is_expired(self):
        """
        Returns True if Azure AD token is expired, False otherwise
        """
        if self.azure_auth_token and self.token_expiry:
            if datetime.now() + timedelta(minutes=5) >= self.token_expiry:
                verbose_logger.debug("Azure AD token is expired. Requesting new token")
                return True
        return False


def build_shared_key_signature(
    account_name: str,
    account_key: bytes,
    method: str,
    url_resource: str,
    headers: Dict[str, str],
) -> str:
    """
    Stateless function to compute Shared Key authorization header for Azure Data Lake Storage Gen2
    using the signature format for SharedKey (not SharedKeyLite).
    """

    # Ensure headers are case-insensitive
    headers = {k.lower(): v for k, v in headers.items()}

    # Required headers for SharedKey
    content_encoding = headers.get("content-encoding", "")
    content_language = headers.get("content-language", "")
    content_length = headers.get("content-length", "")
    if_modified_since = headers.get("if-modified-since", "")
    if_match = headers.get("if-match", "")
    if_none_match = headers.get("if-none-match", "")
    if_unmodified_since = headers.get("if-unmodified-since", "")
    range = headers.get("range", "")
    if content_length == "0":
        content_length = ""
    content_md5 = headers.get("content-md5", "")
    content_type = headers.get("content-type", "")
    date = "" if "x-ms-date" in headers else headers.get("date", "")
    if not date and "x-ms-date" not in headers:
        raise ValueError("Date header or x-ms-date header is required")

    # Construct CanonicalizedHeaders
    canonicalized_headers = ""
    ms_headers = [
        (k.lower(), v.strip())
        for k, v in headers.items()
        if k.lower().startswith("x-ms-")
    ]

    for key, value in sorted(ms_headers):
        canonicalized_headers += f"{key}:{value}\n"

    # Construct CanonicalizedResource
    parsed_url = urlparse(url_resource)
    url_params = parse_qs(parsed_url.query)
    url_resource = parsed_url.path

    canonicalized_resource = f"/{account_name}{url_resource}"
    for key in sorted(url_params.keys()):
        values = url_params[key]
        sorted_values = sorted(values)
        canonicalized_resource += f"\n{key}:{','.join(sorted_values)}"

    # Build the signature string as per SharedKey documentation
    string_to_sign = (
        f"{method}\n"
        f"{content_encoding}\n"
        f"{content_language}\n"
        f"{content_length}\n"
        f"{content_md5}\n"
        f"{content_type}\n"
        f"{date}\n"
        f"{if_modified_since}\n"
        f"{if_match}\n"
        f"{if_none_match}\n"
        f"{if_unmodified_since}\n"
        f"{range}\n"
        f"{canonicalized_headers}"
        f"{canonicalized_resource}"
    )

    signed_hmac_sha256 = hmac.new(
        account_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).digest()
    signature = base64.b64encode(signed_hmac_sha256).decode()

    auth_header = f"SharedKey {account_name}:{signature}"
    return auth_header


def sign_httpx_request_with_shared_key(
    request: Request,
    account_name: str,
    account_key: bytes,
    get_date_str: Callable[[], str] = _get_now_utc_str,
) -> Request:
    """
    Mutates the given httpx.Request to include a Shared Key Authorization header.
    """
    # Ensure required headers
    request.headers.setdefault("x-ms-date", get_date_str())
    request.headers.setdefault("x-ms-version", "2021-04-10")

    # Add content-length header if not present
    if "content-length" not in request.headers:
        request.headers["content-length"] = "0"

    url_resource = request.url.raw_path.decode("utf-8")

    signature = build_shared_key_signature(
        account_name=account_name,
        account_key=account_key,
        method=request.method.upper(),
        url_resource=url_resource,
        headers=dict(request.headers),
    )
    request.headers["Authorization"] = signature
    return request
