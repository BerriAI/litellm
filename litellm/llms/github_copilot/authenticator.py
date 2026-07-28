import json
import os
import time
from typing import Final
from urllib.parse import urlsplit

import httpx
from pydantic import ConfigDict, TypeAdapter, with_config
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.asyncify import can_block_current_thread
from litellm.llms.custom_httpx.http_handler import get_httpx_client

from .common_utils import (
    GetAccessTokenError,
    GetAPIKeyError,
    GetDeviceCodeError,
    get_copilot_auth_headers,
)


# Constants (default values — overridable via environment variables at call time)
DEFAULT_GITHUB_CLIENT_ID: Final = "Iv1.b507a08c87ecfe98"
DEFAULT_GITHUB_DEVICE_CODE_URL: Final = "https://github.com/login/device/code"
DEFAULT_GITHUB_ACCESS_TOKEN_URL: Final = "https://github.com/login/oauth/access_token"
DEFAULT_GITHUB_API_KEY_URL: Final = "https://api.github.com/copilot_internal/v2/token"


@with_config(ConfigDict(extra="allow", strict=True))
class _DeviceCode(TypedDict):
    device_code: ReadOnly[str]
    user_code: ReadOnly[object]
    verification_uri: ReadOnly[object]


@with_config(ConfigDict(extra="allow", strict=True))
class _AccessTokenPoll(TypedDict):
    access_token: ReadOnly[NotRequired[str]]
    error: ReadOnly[NotRequired[object]]


_JSON_OBJECT: Final = TypeAdapter(dict[str, object], config=ConfigDict(strict=True))
_DEVICE_CODE: Final = TypeAdapter(_DeviceCode)
_ACCESS_TOKEN_POLL: Final = TypeAdapter(_AccessTokenPoll)


def github_api_headers(
    access_token: str | None = None,
) -> dict[str, str]:  # mutable-ok: returned straight into httpx handlers whose headers params require dict
    headers: Final = {
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


def _https_hostname(url: str) -> str | None:
    parsed_url = urlsplit(url)
    if (
        parsed_url.scheme.lower() != "https"
        or parsed_url.hostname is None
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.query
        or parsed_url.fragment
    ):
        return None
    return parsed_url.hostname.lower()


def _is_secure_api_base(api_base: str) -> bool:
    return _https_hostname(api_base) is not None


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

        if not can_block_current_thread():
            raise GetAccessTokenError(
                message=(
                    "GitHub Copilot device-code login needs a human and cannot run inside a running event loop "
                    "or a worker thread (for example the LiteLLM proxy). Log in once outside the proxy with "
                    '`python -c "from litellm.llms.github_copilot.authenticator import Authenticator; '
                    'Authenticator().get_access_token()"` and mount the resulting access-token file into '
                    "the proxy, or set GITHUB_COPILOT_TOKEN_DIR to a directory that already holds it."
                ),
                status_code=401,
            )

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
            except (GetDeviceCodeError, GetAccessTokenError) as e:
                verbose_logger.warning("Failed attempt %s: %s", attempt + 1, e)
                continue

        raise GetAccessTokenError(
            message="Failed to get access token after 3 attempts",
            status_code=401,
        )

    def get_api_key(self) -> str:
        try:
            return self.get_access_token()
        except GetAccessTokenError as e:
            raise GetAPIKeyError(
                message=f"Failed to get OAuth access token: {str(e)}",
                status_code=401,
            )

    def get_api_base(self, api_base: str | None = None) -> str | None:
        candidates = (
            ("deployment api_base", api_base),
            ("GITHUB_COPILOT_API_BASE", os.getenv("GITHUB_COPILOT_API_BASE")),
        )
        for source, candidate in candidates:
            if candidate is None:
                continue
            if _is_secure_api_base(candidate):
                return candidate
            verbose_logger.warning(
                f"Ignoring {source} because it must be an HTTPS URL without credentials, query, or fragment"
            )
        return None

    def _ensure_token_dir(self) -> None:
        """Ensure the token directory exists."""
        if not os.path.exists(self.token_dir):
            os.makedirs(self.token_dir, exist_ok=True)

    def _get_github_headers(self) -> dict[str, str]:
        return get_copilot_auth_headers()

    def _get_device_code(self) -> _DeviceCode:
        """
        Get a device code for GitHub authentication.

        Returns:
            Dict[str, str]: Device code information.

        Raises:
            GetDeviceCodeError: If unable to get a device code.
        """
        try:
            sync_client: Final = get_httpx_client()
            device_code_url: Final = os.getenv("GITHUB_COPILOT_DEVICE_CODE_URL", DEFAULT_GITHUB_DEVICE_CODE_URL)
            client_id: Final = os.getenv("GITHUB_COPILOT_CLIENT_ID", DEFAULT_GITHUB_CLIENT_ID)
            resp: Final = sync_client.post(
                device_code_url,
                headers=self._get_github_headers(),
                json={"client_id": client_id, "scope": "read:user"},
            )
            resp.raise_for_status()
            resp_json: Final = _JSON_OBJECT.validate_python(resp.json())

            required_fields: Final = ["device_code", "user_code", "verification_uri"]
            if not all(field in resp_json for field in required_fields):
                verbose_logger.error("Response missing required fields: %s", resp_json)
                raise GetDeviceCodeError(
                    message="Response missing required fields",
                    status_code=400,
                )

            return _DEVICE_CODE.validate_python(resp_json)
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
        sync_client: Final = get_httpx_client()
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
                resp_json = _JSON_OBJECT.validate_python(resp.json())
                poll_result = _ACCESS_TOKEN_POLL.validate_python(resp_json)

                if "access_token" in poll_result:
                    verbose_logger.info("Authentication successful!")
                    return poll_result["access_token"]
                elif "error" in poll_result and poll_result.get("error") == "authorization_pending":
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
