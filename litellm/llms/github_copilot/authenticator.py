import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import _get_httpx_client

from .common_utils import (
    APIKeyExpiredError,
    GetAccessTokenError,
    GetAPIKeyError,
    GetDeviceCodeError,
    RefreshAPIKeyError,
)

# Constants
GITHUB_CLIENT_ID = "Iv1.b507a08c87ecfe98"

# Module-level cache for copilot inference tokens in credential mode.
# Key = GitHub access_token, value = api_key_info dict (token + expires_at).
# Mirrors the file-based api-key.json pattern but kept in memory so that
# per-request Authenticator instances share the cached token.
_credential_api_key_cache: Dict[str, Dict[str, Any]] = {}
GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_API_KEY_URL = "https://api.github.com/copilot_internal/v2/token"


class Authenticator:
    def __init__(self, access_token: Optional[str] = None) -> None:
        """Initialize the GitHub Copilot authenticator.

        Args:
            access_token: If provided, the authenticator operates in
                *credential mode* — it uses this token directly instead of
                the file-based device-code flow.  When ``None`` (the
                default), the existing file-based behaviour is preserved.
        """
        self._injected_access_token = access_token

        if access_token is None:
            # File-based mode (backward compatible)
            self.token_dir = os.getenv(
                "GITHUB_COPILOT_TOKEN_DIR",
                os.path.expanduser("~/.config/litellm/github_copilot"),
            )
            self.access_token_file = os.path.join(
                self.token_dir,
                os.getenv("GITHUB_COPILOT_ACCESS_TOKEN_FILE", "access-token"),
            )
            self.api_key_file = os.path.join(
                self.token_dir,
                os.getenv("GITHUB_COPILOT_API_KEY_FILE", "api-key.json"),
            )
            self._ensure_token_dir()

    def get_access_token(self) -> str:
        """
        Return the GitHub access token.

        In credential mode, returns the injected token directly.
        In file-based mode (SDK), reads from the access-token file on disk.

        Returns:
            str: The GitHub access token.

        Raises:
            GetAccessTokenError: If no access token is available.
        """
        if self._injected_access_token is not None:
            return self._injected_access_token

        try:
            with open(self.access_token_file, "r") as f:
                access_token = f.read().strip()
                if access_token:
                    return access_token
        except IOError:
            pass  # No file — fall through to auth error below

        raise GetAccessTokenError(
            message=(
                "No GitHub Copilot access token configured. "
                "Run `litellm --login github_copilot` (SDK) or create a named credential "
                "via the LiteLLM proxy UI. "
                "See: https://docs.litellm.ai/docs/providers/github_copilot"
            ),
            status_code=401,
        )

    def get_api_key(self) -> str:
        """
        Get the API key, refreshing if necessary.

        Returns:
            str: The GitHub Copilot API key.

        Raises:
            GetAPIKeyError: If unable to obtain an API key.
        """
        if self._injected_access_token is not None:
            return self._get_api_key_credential_mode()

        try:
            with open(self.api_key_file, "r") as f:
                api_key_info = json.load(f)
                if api_key_info.get("expires_at", 0) > datetime.now().timestamp():
                    return api_key_info.get("token")
                else:
                    verbose_logger.warning("API key expired, refreshing")
                    raise APIKeyExpiredError(
                        message="API key expired",
                        status_code=401,
                    )
        except IOError:
            verbose_logger.warning("No API key file found or error opening file")
        except (json.JSONDecodeError, KeyError) as e:
            verbose_logger.warning(f"Error reading API key from file: {str(e)}")
        except APIKeyExpiredError:
            pass  # Already logged in the try block

        try:
            api_key_info = self._refresh_api_key()
            with open(self.api_key_file, "w") as f:
                json.dump(api_key_info, f)
            token = api_key_info.get("token")
            if token:
                return token
            else:
                raise GetAPIKeyError(
                    message="API key response missing token",
                    status_code=401,
                )
        except IOError as e:
            verbose_logger.error(f"Error saving API key to file: {str(e)}")
            raise GetAPIKeyError(
                message=f"Failed to save API key: {str(e)}",
                status_code=500,
            )
        except RefreshAPIKeyError as e:
            raise GetAPIKeyError(
                message=f"Failed to refresh API key: {str(e)}",
                status_code=401,
            )

    def get_api_base(self) -> Optional[str]:
        """
        Get the API endpoint from the api-key.json file.

        Returns:
            Optional[str]: The GitHub Copilot API endpoint, or None if not found.
        """
        if self._injected_access_token is not None:
            cached = _credential_api_key_cache.get(self._injected_access_token)
            if cached:
                endpoints = cached.get("endpoints", {})
                return endpoints.get("api")
            return None

        try:
            with open(self.api_key_file, "r") as f:
                api_key_info = json.load(f)
                endpoints = api_key_info.get("endpoints", {})
                api_endpoint = endpoints.get("api")
                return api_endpoint
        except (IOError, json.JSONDecodeError, KeyError) as e:
            verbose_logger.warning(f"Error reading API endpoint from file: {str(e)}")
            return None

    def _get_api_key_credential_mode(self) -> str:
        """Get API key when operating in credential mode (injected access token).

        Uses a module-level cache keyed by access_token so that multiple
        per-request Authenticator instances share the same cached copilot
        inference token and avoid redundant GitHub API calls.
        """
        cached = _credential_api_key_cache.get(self._injected_access_token)  # type: ignore[arg-type]
        if cached and cached.get("expires_at", 0) > datetime.now().timestamp():
            token = cached.get("token")
            if token:
                return token

        try:
            api_key_info = self._refresh_api_key()
            _credential_api_key_cache[self._injected_access_token] = api_key_info  # type: ignore[index]
            token = api_key_info.get("token")
            if not token:
                raise GetAPIKeyError(
                    message="API key response missing token",
                    status_code=401,
                )
            return token
        except RefreshAPIKeyError as e:
            raise GetAPIKeyError(
                message=f"Failed to refresh API key: {str(e)}",
                status_code=401,
            )

    def _refresh_api_key(self) -> Dict[str, Any]:
        """
        Refresh the API key using the access token.

        Returns:
            Dict[str, Any]: The API key information including token and expiration.

        Raises:
            RefreshAPIKeyError: If unable to refresh the API key.
        """
        access_token = self.get_access_token()
        headers = Authenticator.get_github_headers(access_token)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                sync_client = _get_httpx_client()
                response = sync_client.get(GITHUB_API_KEY_URL, headers=headers)
                response.raise_for_status()

                response_json = response.json()

                if "token" in response_json:
                    return response_json
                else:
                    verbose_logger.warning(
                        f"API key response missing token: {response_json}"
                    )
            except GetAccessTokenError:
                raise  # Re-raise with the original helpful message (docs link etc.)
            except httpx.HTTPStatusError as e:
                verbose_logger.error(
                    f"HTTP error refreshing API key (attempt {attempt+1}/{max_retries}): {str(e)}"
                )
            except Exception as e:
                verbose_logger.error(f"Unexpected error refreshing API key: {str(e)}")

        raise RefreshAPIKeyError(
            message="Failed to refresh API key after maximum retries",
            status_code=401,
        )

    def _ensure_token_dir(self) -> None:
        """Ensure the token directory exists."""
        if not os.path.exists(self.token_dir):
            os.makedirs(self.token_dir, exist_ok=True)

    @staticmethod
    def get_github_headers(access_token: Optional[str] = None) -> Dict[str, str]:
        """
        Generate standard GitHub headers for API requests.

        This is a static method so it can be imported and used by the SSO
        endpoint module without instantiating an Authenticator.

        Args:
            access_token: Optional access token to include in the headers.

        Returns:
            Dict[str, str]: Headers for GitHub API requests.
        """
        headers = {
            "accept": "application/json",
            "editor-version": "vscode/1.85.1",
            "editor-plugin-version": "copilot/1.155.0",
            "user-agent": "GithubCopilot/1.155.0",
            "accept-encoding": "gzip,deflate,br",
            "content-type": "application/json",
        }

        if access_token:
            headers["authorization"] = f"token {access_token}"

        return headers

    def _get_device_code(self) -> Dict[str, str]:
        """
        Get a device code for GitHub authentication.

        Returns:
            Dict[str, str]: Device code information.

        Raises:
            GetDeviceCodeError: If unable to get a device code.
        """
        try:
            sync_client = _get_httpx_client()
            resp = sync_client.post(
                GITHUB_DEVICE_CODE_URL,
                headers=Authenticator.get_github_headers(),
                json={"client_id": GITHUB_CLIENT_ID, "scope": "read:user"},
            )
            resp.raise_for_status()
            resp_json = resp.json()

            required_fields = ["device_code", "user_code", "verification_uri"]
            if not all(field in resp_json for field in required_fields):
                verbose_logger.error(f"Response missing required fields: {resp_json}")
                raise GetDeviceCodeError(
                    message="Response missing required fields",
                    status_code=400,
                )

            return resp_json
        except httpx.HTTPStatusError as e:
            verbose_logger.error(f"HTTP error getting device code: {str(e)}")
            raise GetDeviceCodeError(
                message=f"Failed to get device code: {str(e)}",
                status_code=400,
            )
        except json.JSONDecodeError as e:
            verbose_logger.error(f"Error decoding JSON response: {str(e)}")
            raise GetDeviceCodeError(
                message=f"Failed to decode device code response: {str(e)}",
                status_code=400,
            )
        except Exception as e:
            verbose_logger.error(f"Unexpected error getting device code: {str(e)}")
            raise GetDeviceCodeError(
                message=f"Failed to get device code: {str(e)}",
                status_code=400,
            )

    def _poll_for_access_token(self, device_code: str) -> str:
        """
        Poll for an access token after user authentication.

        Args:
            device_code: The device code to use for polling.

        Returns:
            str: The access token.

        Raises:
            GetAccessTokenError: If unable to get an access token.
        """
        sync_client = _get_httpx_client()
        max_attempts = 12  # 1 minute (12 * 5 seconds)

        for attempt in range(max_attempts):
            try:
                resp = sync_client.post(
                    GITHUB_ACCESS_TOKEN_URL,
                    headers=Authenticator.get_github_headers(),
                    json={
                        "client_id": GITHUB_CLIENT_ID,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                )
                resp.raise_for_status()
                resp_json = resp.json()

                if "access_token" in resp_json:
                    verbose_logger.info("Authentication successful!")
                    return resp_json["access_token"]
                elif (
                    "error" in resp_json
                    and resp_json.get("error") == "authorization_pending"
                ):
                    verbose_logger.debug(
                        f"Authorization pending (attempt {attempt+1}/{max_attempts})"
                    )
                else:
                    verbose_logger.warning(f"Unexpected response: {resp_json}")
            except httpx.HTTPStatusError as e:
                verbose_logger.error(f"HTTP error polling for access token: {str(e)}")
                raise GetAccessTokenError(
                    message=f"Failed to get access token: {str(e)}",
                    status_code=400,
                )
            except json.JSONDecodeError as e:
                verbose_logger.error(f"Error decoding JSON response: {str(e)}")
                raise GetAccessTokenError(
                    message=f"Failed to decode access token response: {str(e)}",
                    status_code=400,
                )
            except Exception as e:
                verbose_logger.error(
                    f"Unexpected error polling for access token: {str(e)}"
                )
                raise GetAccessTokenError(
                    message=f"Failed to get access token: {str(e)}",
                    status_code=400,
                )

            time.sleep(5)

        raise GetAccessTokenError(
            message="Timed out waiting for user to authorize the device",
            status_code=400,
        )

    def _login(self) -> str:
        """
        Login to GitHub Copilot using device code flow.

        Returns:
            str: The GitHub access token.

        Raises:
            GetDeviceCodeError: If unable to get a device code.
            GetAccessTokenError: If unable to get an access token.
        """
        device_code_info = self._get_device_code()

        device_code = device_code_info["device_code"]
        user_code = device_code_info["user_code"]
        verification_uri = device_code_info["verification_uri"]

        print(  # noqa: T201
            f"Please visit {verification_uri} and enter code {user_code} to authenticate.",
            # When this is running in docker, it may not be flushed immediately
            # so we force flush to ensure the user sees the message
            flush=True,
        )

        return self._poll_for_access_token(device_code)
