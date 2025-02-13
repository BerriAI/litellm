import os

import httpx
import json

import time
from datetime import datetime
from typing import Optional

from litellm._logging import verbose_logger
from litellm.caching import InMemoryCache
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
    httpxSpecialProvider,
)

from .constants import (
    GetAccessTokenError,
    GetDeviceCodeError,
    RefreshAPIKeyError,
    GetAPIKeyError,
    APIKeyExpiredError,
)

class Authenticator:
    def __init__(self) -> None:
        # Token storage paths
        self.token_dir = os.getenv("GITHUB_COPILOT_TOKEN_DIR", os.path.expanduser("~/.config/litellm/github_copilot"))
        self.access_token_file = os.path.join(self.token_dir, os.getenv("GITHUB_COPILOT_ACCESS_TOKEN_FILE", "access-token"))
        self.api_key_file = os.path.join(self.token_dir, os.getenv("GITHUB_COPILOT_API_KEY_FILE", "api-key.json"))
        self._ensure_token_dir()

        self.get_access_token()

    def get_access_token(self) -> str:
        """
        Login to Copilot with retry 3 times

        Returns:
        access_token: str

        """
        try:
            with open(self.access_token_file, 'r') as f:
                access_token = f.read().strip()
                return access_token
        except IOError:
            verbose_logger.warning("Error loading access token from file")
        
        for _ in range(3):
            try:
                access_token = self._login()
            except GetDeviceCodeError | GetAccessTokenError | RefreshAPIKeyError:
                continue
            else:
                try:
                    with open(self.access_token_file, 'w') as f:
                        f.write(access_token)
                except IOError:
                    verbose_logger.error("Error saving access token to file")
                return access_token

        raise GetAccessTokenError("Failed to get access token")

    def get_api_key(self) -> str:
        """Get the API key"""
        try:
            with open(self.api_key_file, 'r') as f:
                api_key_info = json.load(f)
                if api_key_info.get('expires_at') > datetime.now().timestamp():
                    return api_key_info.get('token')
                else:
                    raise APIKeyExpiredError("API key expired")
        except IOError:
            verbose_logger.warning("Error opening API key file")
        except (json.JSONDecodeError, KeyError):
            verbose_logger.warning("Error reading API key from file")
        except APIKeyExpiredError:
            verbose_logger.warning("API key expired")
        
        try:
            api_key_info = self._refresh_api_key()
            with open(self.api_key_file, 'w') as f:
                json.dump(api_key_info, f)
        except IOError:
            verbose_logger.error("Error saving API key to file")
        except RefreshAPIKeyError:
            raise GetAPIKeyError("Failed to refresh API key")
        
        return api_key_info.get('token')

    def _refresh_api_key(self) -> dict:
        """
        Refresh the API key using the access token

        Returns:
        api_key_info: dict
        """

        access_token = self.get_access_token()
        headers = {
            'authorization': f'token {access_token}',
            'editor-version': 'vscode/1.85.1',
            'editor-plugin-version': 'copilot/1.155.0',
            'user-agent': 'GithubCopilot/1.155.0'
        }

        max_retries = 3
        for _ in range(max_retries):
            try:
                sync_client = _get_httpx_client()
                response = sync_client.get(
                    'https://api.github.com/copilot_internal/v2/token',
                    headers=headers
                )
                response.raise_for_status()

                response_json = response.json()

                if 'token' in response_json:
                    return response_json
            except httpx.HTTPStatusError as e:
                verbose_logger.error(f"Error refreshing API key: {str(e)}")

        raise RefreshAPIKeyError("Failed to refresh API key")
            

    def _ensure_token_dir(self) -> None:
        """Ensure the token directory exists"""
        if not os.path.exists(self.token_dir):
            os.makedirs(self.token_dir, exist_ok=True)

    def _login(self) -> str:
        """
        Login to GitHub Copilot using device code flow

        Returns:
        access_token: str
        """

        try:
            sync_client = _get_httpx_client()
            # Get device code
            resp = sync_client.post(
                'https://github.com/login/device/code',
                headers={
                    'accept': 'application/json',
                    'editor-version': 'vscode/1.85.1',
                    'editor-plugin-version': 'copilot/1.155.0',
                    'content-type': 'application/json',
                    'user-agent': 'GithubCopilot/1.155.0',
                    'accept-encoding': 'gzip,deflate,br'
                },
                json={"client_id": "Iv1.b507a08c87ecfe98", "scope": "read:user"}
            )
            resp.raise_for_status()
            resp_json = resp.json()

            device_code = resp_json.get('device_code')
            user_code = resp_json.get('user_code')
            verification_uri = resp_json.get('verification_uri')

            if not all([device_code, user_code, verification_uri]):
                verbose_logger.error("Response missing required fields")
                return None
        except httpx.HttpError as e:
            verbose_logger.error(f"Error getting device code: {str(e)}")
            raise GetDeviceCodeError("Failed to get device code")
        except json.JSONDecodeError as e:
            verbose_logger.error(f"Error decoding JSON response: {str(e)}")
            raise GetDeviceCodeError("Failed to get device code")
        except RuntimeError as e:
            verbose_logger.error(f"Error getting device code: {str(e)}")
            raise GetDeviceCodeError("Failed to get device code")
        
        print(f'Please visit {verification_uri} and enter code {user_code} to authenticate.')

        while True:
            time.sleep(5)

            # Get access token
            try:
                resp = sync_client.post(
                    'https://github.com/login/oauth/access_token',
                    headers={
                        'accept': 'application/json',
                        'editor-version': 'vscode/1.85.1',
                        'editor-plugin-version': 'copilot/1.155.0',
                        'content-type': 'application/json',
                        'user-agent': 'GithubCopilot/1.155.0',
                        'accept-encoding': 'gzip,deflate,br'
                    },
                    json={
                        "client_id": "Iv1.b507a08c87ecfe98",
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code"
                    }
                )
                resp.raise_for_status()
                resp_json = resp.json()

                if "access_token" in resp_json:
                    verbose_logger.info("Authentication success!")
                    return resp_json['access_token']
                else:
                    continue
            except httpx.HTTPStatusError as e:
                verbose_logger.error(f"Error getting access token: {str(e)}")
                raise GetAccessTokenError("Failed to get access token")
            except json.JSONDecodeError as e:
                verbose_logger.error(f"Error decoding JSON response: {str(e)}")
                raise GetAccessTokenError("Failed to get access token")




