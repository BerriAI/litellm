"""Anthropic Workload Identity Federation token exchange.

See https://platform.claude.com/docs/en/manage-claude/wif-reference.
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from litellm._logging import verbose_logger

ANTHROPIC_WORKLOAD_IDENTITY_FEDERATION_GRANT_TYPE = (
    "urn:ietf:params:oauth:grant-type:jwt-bearer"
)
ANTHROPIC_WORKLOAD_IDENTITY_FEDERATION_TOKEN_PATH = "/v1/oauth/token"

# Two-tier refresh thresholds, matching the official Anthropic SDKs.
ANTHROPIC_WORKLOAD_IDENTITY_FEDERATION_ADVISORY_REFRESH_SECONDS = 120
ANTHROPIC_WORKLOAD_IDENTITY_FEDERATION_MANDATORY_REFRESH_SECONDS = 30


class AnthropicWorkloadIdentityFederationError(Exception):
    pass


@dataclass
class AnthropicWorkloadIdentityFederationCredentials:
    federation_rule_id: str
    organization_id: str
    service_account_id: str
    workspace_id: Optional[str] = None
    identity_token: Optional[str] = None
    identity_token_file: Optional[str] = None

    def __post_init__(self) -> None:
        required = {
            "federation_rule_id": self.federation_rule_id,
            "organization_id": self.organization_id,
            "service_account_id": self.service_account_id,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise AnthropicWorkloadIdentityFederationError(
                f"Anthropic workload identity federation credentials missing required fields: {missing}"
            )
        if not self.identity_token and not self.identity_token_file:
            raise AnthropicWorkloadIdentityFederationError(
                "Anthropic workload identity federation credentials must set either "
                "`identity_token` or `identity_token_file`."
            )


def get_workload_identity_federation_credentials_from_env() -> (
    Optional[AnthropicWorkloadIdentityFederationCredentials]
):
    from litellm.secret_managers.main import get_secret_str

    federation_rule_id = get_secret_str("ANTHROPIC_FEDERATION_RULE_ID")
    organization_id = get_secret_str("ANTHROPIC_ORGANIZATION_ID")
    service_account_id = get_secret_str("ANTHROPIC_SERVICE_ACCOUNT_ID")
    identity_token = get_secret_str("ANTHROPIC_IDENTITY_TOKEN")
    identity_token_file = get_secret_str("ANTHROPIC_IDENTITY_TOKEN_FILE")
    workspace_id = get_secret_str("ANTHROPIC_WORKSPACE_ID")

    if not all((federation_rule_id, organization_id, service_account_id)):
        return None
    if not identity_token and not identity_token_file:
        return None

    return AnthropicWorkloadIdentityFederationCredentials(
        federation_rule_id=federation_rule_id,  # type: ignore[arg-type]
        organization_id=organization_id,  # type: ignore[arg-type]
        service_account_id=service_account_id,  # type: ignore[arg-type]
        workspace_id=workspace_id,
        identity_token=identity_token,
        identity_token_file=identity_token_file,
    )


class AnthropicWorkloadIdentityFederationTokenProvider:
    def __init__(
        self,
        credentials: AnthropicWorkloadIdentityFederationCredentials,
        api_base: str = "https://api.anthropic.com",
        http_client: Optional[httpx.Client] = None,
        timeout: float = 30.0,
    ) -> None:
        self._credentials = credentials
        self._api_base = api_base.rstrip("/")
        self._http_client = http_client
        self._timeout = timeout
        self._lock = threading.Lock()
        self._access_token: Optional[str] = None
        self._expires_at: float = 0.0

    @property
    def credentials(self) -> AnthropicWorkloadIdentityFederationCredentials:
        return self._credentials

    def get_token(self, assertion: Optional[str] = None) -> str:
        with self._lock:
            now = time.time()
            time_to_expiry = self._expires_at - now

            if (
                assertion is None
                and self._access_token
                and time_to_expiry
                > ANTHROPIC_WORKLOAD_IDENTITY_FEDERATION_ADVISORY_REFRESH_SECONDS
            ):
                return self._access_token

            mandatory = (
                assertion is not None
                or self._access_token is None
                or time_to_expiry
                <= ANTHROPIC_WORKLOAD_IDENTITY_FEDERATION_MANDATORY_REFRESH_SECONDS
            )

            try:
                self._refresh_locked(assertion=assertion)
            except Exception as exc:
                if mandatory:
                    raise
                verbose_logger.warning(
                    "Anthropic workload identity federation advisory refresh failed; reusing cached token: %s",
                    exc,
                )

            assert self._access_token is not None
            return self._access_token

    def _resolve_assertion(self, override: Optional[str]) -> str:
        if override:
            return override
        if self._credentials.identity_token:
            return self._credentials.identity_token
        path = self._credentials.identity_token_file
        if not path:
            raise AnthropicWorkloadIdentityFederationError(
                "Anthropic workload identity federation credentials have neither "
                "`identity_token` nor `identity_token_file` set."
            )
        # Re-read on each exchange: cloud workload identity providers
        # rotate the projected JWT in place, so caching its contents
        # would silently send a stale assertion.
        try:
            with open(path, "r", encoding="utf-8") as f:
                assertion = f.read().strip()
        except OSError as exc:
            raise AnthropicWorkloadIdentityFederationError(
                f"Failed to read ANTHROPIC_IDENTITY_TOKEN_FILE at {path!r}: {exc}"
            ) from exc
        if not assertion:
            raise AnthropicWorkloadIdentityFederationError(
                f"ANTHROPIC_IDENTITY_TOKEN_FILE at {path!r} is empty"
            )
        return assertion

    def _refresh_locked(self, assertion: Optional[str] = None) -> None:
        resolved_assertion = self._resolve_assertion(assertion)
        url = f"{self._api_base}{ANTHROPIC_WORKLOAD_IDENTITY_FEDERATION_TOKEN_PATH}"
        data = {
            "grant_type": ANTHROPIC_WORKLOAD_IDENTITY_FEDERATION_GRANT_TYPE,
            "assertion": resolved_assertion,
            "federation_rule_id": self._credentials.federation_rule_id,
            "organization_id": self._credentials.organization_id,
            "service_account_id": self._credentials.service_account_id,
        }
        if self._credentials.workspace_id:
            data["workspace_id"] = self._credentials.workspace_id
        headers = {
            "content-type": "application/x-www-form-urlencoded",
            "accept": "application/json",
        }

        try:
            if self._http_client is not None:
                response = self._http_client.post(url=url, data=data, headers=headers)
            else:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(url=url, data=data, headers=headers)
        except httpx.HTTPError as exc:
            raise AnthropicWorkloadIdentityFederationError(
                f"Anthropic workload identity federation token exchange failed: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise AnthropicWorkloadIdentityFederationError(
                f"Anthropic workload identity federation token exchange returned HTTP {response.status_code}: {response.text}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise AnthropicWorkloadIdentityFederationError(
                f"Anthropic workload identity federation token exchange returned non-JSON response: {response.text}"
            ) from exc

        access_token = payload.get("access_token")
        if not access_token:
            raise AnthropicWorkloadIdentityFederationError(
                f"Anthropic workload identity federation token exchange response missing 'access_token': {payload}"
            )

        # Fall back to a short window if expires_in is missing/invalid —
        # otherwise we'd cache forever on a misbehaving response.
        expires_in = payload.get("expires_in")
        try:
            expires_in_int = int(expires_in) if expires_in is not None else 0
        except (TypeError, ValueError):
            expires_in_int = 0
        if expires_in_int <= 0:
            expires_in_int = (
                ANTHROPIC_WORKLOAD_IDENTITY_FEDERATION_MANDATORY_REFRESH_SECONDS * 2
            )

        self._access_token = access_token
        self._expires_at = time.time() + expires_in_int


_PROVIDER_CACHE: dict = {}
_PROVIDER_CACHE_LOCK = threading.Lock()


def _provider_cache_key(
    credentials: AnthropicWorkloadIdentityFederationCredentials, api_base: str
) -> tuple:
    return (
        credentials.federation_rule_id,
        credentials.organization_id,
        credentials.service_account_id,
        credentials.workspace_id,
        credentials.identity_token,
        credentials.identity_token_file,
        api_base,
    )


def get_or_create_workload_identity_federation_provider(
    credentials: AnthropicWorkloadIdentityFederationCredentials,
    api_base: str = "https://api.anthropic.com",
) -> AnthropicWorkloadIdentityFederationTokenProvider:
    key = _provider_cache_key(credentials, api_base)
    with _PROVIDER_CACHE_LOCK:
        provider = _PROVIDER_CACHE.get(key)
        if provider is None:
            provider = AnthropicWorkloadIdentityFederationTokenProvider(
                credentials=credentials, api_base=api_base
            )
            _PROVIDER_CACHE[key] = provider
    return provider


def exchange_anthropic_workload_identity_federation_token(
    api_base: Optional[str] = None,
    credentials: Optional[AnthropicWorkloadIdentityFederationCredentials] = None,
    assertion: Optional[str] = None,
) -> Optional[str]:
    resolved = credentials or get_workload_identity_federation_credentials_from_env()
    if resolved is None:
        return None
    provider = get_or_create_workload_identity_federation_provider(
        credentials=resolved,
        api_base=(api_base or "https://api.anthropic.com").rstrip("/"),
    )
    return provider.get_token(assertion=assertion)


def reset_workload_identity_federation_provider_cache() -> None:
    # Intended for tests — prevents stale providers leaking across cases.
    with _PROVIDER_CACHE_LOCK:
        _PROVIDER_CACHE.clear()
