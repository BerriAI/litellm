from typing import Optional

from litellm.litellm_core_utils.cli_token_utils import get_litellm_gateway_api_key

from .chat import ChatClient
from .credentials import CredentialsManagementClient
from .http_client import HTTPClient
from .keys import KeysManagementClient
from .model_groups import ModelGroupsManagementClient
from .models import ModelsManagementClient
from .teams import TeamsManagementClient


class Client:
    """Main client for interacting with the LiteLLM proxy API."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize the LiteLLM proxy client.

        Args:
            base_url (str): The base URL of the LiteLLM proxy server (e.g., "http://localhost:4000")
            api_key (Optional[str]): API key for authentication. If provided, it will be sent as a Bearer token.
            timeout: Request timeout in seconds (default: 30)
        """
        self._base_url = base_url.rstrip("/")  # Remove trailing slash if present
        self._api_key = get_litellm_gateway_api_key() or api_key

        # Initialize resource clients

        self.http = HTTPClient(base_url=base_url, api_key=api_key, timeout=timeout)
        self.models = ModelsManagementClient(base_url=self._base_url, api_key=self._api_key)
        self.model_groups = ModelGroupsManagementClient(base_url=self._base_url, api_key=self._api_key)
        self.chat = ChatClient(base_url=self._base_url, api_key=self._api_key)
        self.keys = KeysManagementClient(base_url=self._base_url, api_key=self._api_key)
        self.credentials = CredentialsManagementClient(base_url=self._base_url, api_key=self._api_key)
        self.teams = TeamsManagementClient(base_url=self._base_url, api_key=self._api_key)
