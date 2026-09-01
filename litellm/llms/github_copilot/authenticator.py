import json
import os
import time
from datetime import datetime
from typing import Any, Final

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

# Constants (default values — overridable via environment variables at call time)
DEFAULT_GITHUB_CLIENT_ID: Final = "Iv1.b507a08c87ecfe98"
DEFAULT_GITHUB_DEVICE_CODE_URL: Final = "https://github.com/login/device/code"
DEFAULT_GITHUB_ACCESS_TOKEN_URL: Final = "https://github.com/login/oauth/access_token"
DEFAULT_GITHUB_API_KEY_URL: Final = "https://api.github.com/copilot_internal/v2/token"


class Authenticator:
    def __init__(self) -> None:
        """Initialize the GitHub Copilot authenticator with configurable token paths."""
        # Token storage paths
        self.token_dir = os.getenv(
            "GITHUB_COPILOT_TOKEN_DIR",
            os.path.expanduser("~/.config/litellm/github_copilot"),
        )
        self.access_token_file = os.path.join(
            self.token_dir,
            os.getenv("GITHUB_COPILOT_ACCESS_TOKEN_FILE", "access-token"),
        )
        self.api_key_file = os.path.join(self.token_dir, os.getenv("GITHUB_COPILOT_API_KEY_FILE", "api-key.json"))
        self._ensure_token_dir()

    def get_access_token(self) -> str:
        """
        Login to Copilot with retry 3 times.

        Returns:
            str: The GitHub access token.

        Raises:
            GetAccessTokenError: If unable to obtain an access token after retries.
        """
        try:
            with open(self.access_token_file, "r") as f:
                access_token = f.read().strip()
                if access_token:
                    return access_token
        except OSError:
            verbose_logger.warning("No existing access token found or error reading file")

        for attempt in range(3):
            verbose_logger.debug("Access token acquisition attempt %s/3", attempt + 1)
            try:
                access_token = self._login()
                try:
                    with open(self.access_token_file, "w") as f:
                        f.write(access_token)
                except OSError:
                    verbose_logger.error("Error saving access token to file")
                return access_token
            except (GetDeviceCodeError, GetAccessTokenError, RefreshAPIKeyError) as e:
                verbose_logger.warning("Failed attempt %s: %s", attempt + 1, e)
                continue

        raise GetAccessTokenError(
            message="Failed to get access token after 3 attempts",
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
        except OSError:
            verbose_logger.warning("No API key file found or error opening file")
        except (json.JSONDecodeError, KeyError) as e:
            verbose_logger.warning("Error reading API key from file: %s", e)
        except APIKeyExpiredError:
            pass  # Already logged in the try block

        try:
            api_key_info = self._refresh_api_key()
            with open(self.api_key_file, "w") as f:
                json.dump(api_key_info, f)
            token: Final = api_key_info.get("token")
            if token:
                return token
            else:
                raise GetAPIKeyError(
                    message="API key response missing token",
                    status_code=401,
                )
        except OSError as e:
            verbose_logger.error("Error saving API key to file: %s", e)
            raise GetAPIKeyError(
                message=f"Failed to save API key: {e}",
                status_code=500,
            )
        except RefreshAPIKeyError as e:
            raise GetAPIKeyError(
                message=f"Failed to refresh API key: {e}",
                status_code=401,
            )

    def get_api_base(self) -> str | None:
        """
        Get the API endpoint from the api-key.json file.

        Returns:
            Optional[str]: The GitHub Copilot API endpoint, or None if not found.
        """
        try:
            with open(self.api_key_file, "r") as f:
                api_key_info: Final = json.load(f)
                endpoints: Final = api_key_info.get("endpoints", {})
                api_endpoint: Final = endpoints.get("api")
                return api_endpoint
        except (OSError, json.JSONDecodeError, KeyError) as e:
            verbose_logger.warning("Error reading API endpoint from file: %s", e)
            return None

    def _refresh_api_key(self) -> dict[str, Any]:
        """
        Refresh the API key using the access token.

        Returns:
            Dict[str, Any]: The API key information including token and expiration.

        Raises:
            RefreshAPIKeyError: If unable to refresh the API key.
        """
        access_token: Final = self.get_access_token()
        headers: Final = self._get_github_headers(access_token)
        api_key_url: Final = os.getenv("GITHUB_COPILOT_API_KEY_URL", DEFAULT_GITHUB_API_KEY_URL)

        max_retries: Final = 3
        for attempt in range(max_retries):
            try:
                sync_client = _get_httpx_client()
                response = sync_client.get(api_key_url, headers=headers)
                response.raise_for_status()

                response_json = response.json()

                if "token" in response_json:
                    return response_json
                else:
                    verbose_logger.warning("API key response missing token: %s", response_json)
            except httpx.HTTPStatusError as e:
                verbose_logger.error("HTTP error refreshing API key (attempt %s/%s): %s", attempt + 1, max_retries, e)
            except Exception as e:
                verbose_logger.error("Unexpected error refreshing API key: %s", e)

        raise RefreshAPIKeyError(
            message="Failed to refresh API key after maximum retries",
            status_code=401,
        )

    def _ensure_token_dir(self) -> None:
        """Ensure the token directory exists."""
        if not os.path.exists(self.token_dir):
            os.makedirs(self.token_dir, exist_ok=True)

    def _get_github_headers(self, access_token: str | None = None) -> dict[str, str]:
        """
        Generate standard GitHub headers for API requests.

        Args:
            access_token: Optional access token to include in the headers.

        Returns:
            Dict[str, str]: Headers for GitHub API requests.
        """
        headers: Final = {
            "accept": "application/json",
            "editor-version": "vscode/1.85.1",
            "editor-plugin-version": "copilot/1.155.0",
            "user-agent": "GithubCopilot/1.155.0",
            "accept-encoding": "gzip,deflate,br",
        }

        if access_token:
            headers["authorization"] = f"token {access_token}"

        if "content-type" not in headers:
            headers["content-type"] = "application/json"

        return headers

    def _get_device_code(self) -> dict[str, str]:
        """
        Get a device code for GitHub authentication.

        Returns:
            Dict[str, str]: Device code information.

        Raises:
            GetDeviceCodeError: If unable to get a device code.
        """
        try:
            sync_client: Final = _get_httpx_client()
            device_code_url: Final = os.getenv("GITHUB_COPILOT_DEVICE_CODE_URL", DEFAULT_GITHUB_DEVICE_CODE_URL)
            client_id: Final = os.getenv("GITHUB_COPILOT_CLIENT_ID", DEFAULT_GITHUB_CLIENT_ID)
            resp: Final = sync_client.post(
                device_code_url,
                headers=self._get_github_headers(),
                json={"client_id": client_id, "scope": "read:user"},
            )
            resp.raise_for_status()
            resp_json: Final = resp.json()

            required_fields: Final = ["device_code", "user_code", "verification_uri"]
            if not all(field in resp_json for field in required_fields):
                verbose_logger.error("Response missing required fields: %s", resp_json)
                raise GetDeviceCodeError(
                    message="Response missing required fields",
                    status_code=400,
                )

            return resp_json
        except httpx.HTTPStatusError as e:
            verbose_logger.error("HTTP error getting device code: %s", e)
            raise GetDeviceCodeError(
                message=f"Failed to get device code: {e}",
                status_code=400,
            )
        except json.JSONDecodeError as e:
            verbose_logger.error("Error decoding JSON response: %s", e)
            raise GetDeviceCodeError(
                message=f"Failed to decode device code response: {e}",
                status_code=400,
            )
        except Exception as e:
            verbose_logger.error("Unexpected error getting device code: %s", e)
            raise GetDeviceCodeError(
                message=f"Failed to get device code: {e}",
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
        sync_client: Final = _get_httpx_client()
        max_attempts: Final = 12  # 1 minute (12 * 5 seconds)

        access_token_url: Final = os.getenv("GITHUB_COPILOT_ACCESS_TOKEN_URL", DEFAULT_GITHUB_ACCESS_TOKEN_URL)
        client_id: Final = os.getenv("GITHUB_COPILOT_CLIENT_ID", DEFAULT_GITHUB_CLIENT_ID)

        for attempt in range(max_attempts):
            try:
                resp = sync_client.post(
                    access_token_url,
                    headers=self._get_github_headers(),
                    json={
                        "client_id": client_id,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                )
                resp.raise_for_status()
                resp_json = resp.json()

                if "access_token" in resp_json:
                    verbose_logger.info("Authentication successful!")
                    return resp_json["access_token"]
                elif "error" in resp_json and resp_json.get("error") == "authorization_pending":
                    verbose_logger.debug("Authorization pending (attempt %s/%s)", attempt + 1, max_attempts)
                else:
                    verbose_logger.warning("Unexpected response: %s", resp_json)
            except httpx.HTTPStatusError as e:
                verbose_logger.error("HTTP error polling for access token: %s", e)
                raise GetAccessTokenError(
                    message=f"Failed to get access token: {e}",
                    status_code=400,
                )
            except json.JSONDecodeError as e:
                verbose_logger.error("Error decoding JSON response: %s", e)
                raise GetAccessTokenError(
                    message=f"Failed to decode access token response: {e}",
                    status_code=400,
                )
            except Exception as e:
                verbose_logger.error("Unexpected error polling for access token: %s", e)
                raise GetAccessTokenError(
                    message=f"Failed to get access token: {e}",
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
        device_code_info: Final = self._get_device_code()

        device_code: Final = device_code_info["device_code"]
        user_code: Final = device_code_info["user_code"]
        verification_uri: Final = device_code_info["verification_uri"]

        print(  # noqa: T201
            f"Please visit {verification_uri} and enter code {user_code} to authenticate.",
            # When this is running in docker, it may not be flushed immediately
            # so we force flush to ensure the user sees the message
            flush=True,
        )

        return self._poll_for_access_token(device_code)
