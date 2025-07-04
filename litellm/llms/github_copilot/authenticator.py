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
GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_API_KEY_URL = "https://api.github.com/copilot_internal/v2/token"


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
        self.api_key_file = os.path.join(
            self.token_dir, os.getenv("GITHUB_COPILOT_API_KEY_FILE", "api-key.json")
        )
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
        except IOError:
            verbose_logger.warning(
                "No existing access token found or error reading file"
            )

        for attempt in range(3):
            verbose_logger.debug(f"Access token acquisition attempt {attempt + 1}/3")
            try:
                access_token = self._login()
                try:
                    with open(self.access_token_file, "w") as f:
                        f.write(access_token)
                except IOError:
                    verbose_logger.error("Error saving access token to file")
                return access_token
            except (GetDeviceCodeError, GetAccessTokenError, RefreshAPIKeyError) as e:
                verbose_logger.warning(f"Failed attempt {attempt + 1}: {str(e)}")
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

    def _refresh_api_key(self) -> Dict[str, Any]:
        """
        Refresh the API key using the access token.

        Returns:
            Dict[str, Any]: The API key information including token and expiration.

        Raises:
            RefreshAPIKeyError: If unable to refresh the API key.
        """
        access_token = self.get_access_token()
        headers = self._get_github_headers(access_token)

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

    def _get_github_headers(self, access_token: Optional[str] = None) -> Dict[str, str]:
        """
        Generate standard GitHub headers for API requests.

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
        }

        if access_token:
            headers["authorization"] = f"token {access_token}"

        if "content-type" not in headers:
            headers["content-type"] = "application/json"

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
                headers=self._get_github_headers(),
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
                    headers=self._get_github_headers(),
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
            f"Please visit {verification_uri} and enter code {user_code} to authenticate."
        )

        return self._poll_for_access_token(device_code)
