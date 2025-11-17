import os
import time
import uuid
from typing import Optional, Union, Any

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class GigaChatError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[Union[dict, httpx.Headers]] = None,
    ) -> None:
        super().__init__(status_code=status_code, message=message, headers=headers)


class BaseGigaChat:
    def __init__(self):
        self._token_cache: dict[str, Any] = {"token": None, "expires_at": 0}
        self._ssl_verify = False if os.environ.get("GIGACHAT_VERIFY_SSL_CERTS", "").lower() == "false" else True

    @staticmethod
    def _check_timestamp_unit(timestamp):
        """In login+password auth expires_at is in seconds, while in scope+cred in milliseconds"""
        if len(str(timestamp)) == 10:
            return "seconds"
        else:
            return "milliseconds"

    def _is_token_expired(self) -> bool:
        """Check if cached OAuth token is expired or missing."""
        now = (
            time.time()
            if self._check_timestamp_unit(self._token_cache["expires_at"]) == "seconds"
            else time.time() * 1000
        )
        return not self._token_cache["token"] or now >= self._token_cache["expires_at"]

    def _get_oauth_token(self) -> Optional[str]:
        """
        Get or refresh OAuth token using either:
        - username/password (returns tok + exp)
        - client credentials (returns access_token + expires_at)
        """

        # Reuse cached token if valid
        if not self._is_token_expired():
            return self._token_cache["token"]

        import httpx
        import litellm

        auth_url = (
            litellm.get_secret_str("GIGACHAT_AUTH_URL")
            or "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        )

        username = litellm.get_secret_str("GIGACHAT_USERNAME")
        password = litellm.get_secret_str("GIGACHAT_PASSWORD")
        scope = litellm.get_secret_str("GIGACHAT_SCOPE") or "GIGACHAT_API_PERS"
        credentials = litellm.get_secret_str("GIGACHAT_CREDENTIALS")

        if username and password:
            # Username/password OAuth flow
            response = httpx.post(
                auth_url,
                auth=(username, password),
                timeout=30,
                verify=self._ssl_verify,
            )
            response.raise_for_status()
            data = response.json()
            token = data.get("tok")
            expires_at = float(data.get("exp", 0))

        else:
            # Client credentials flow
            if not credentials:
                raise ValueError("Missing GIGACHAT_CREDENTIALS or username/password")
            headers = {
                "User-Agent": "GigaChat-python-lib",
                "RqUID": str(uuid.uuid4()),
                "Authorization": f"Basic {credentials}",
            }
            data = {"scope": scope}
            response = httpx.post(
                auth_url, headers=headers, data=data, timeout=30, verify=self._ssl_verify
            )
            response.raise_for_status()
            data = response.json()
            token = data.get("access_token")
            expires_at = float(data.get("expires_at", 0))

        if not token:
            raise ValueError("OAuth did not return a token")

        # Cache the token
        self._token_cache["token"] = token
        self._token_cache["expires_at"] = expires_at or (time.time() + 1800)

        return token