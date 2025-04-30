from typing import Optional
from .models import ModelsManagementClient
from .model_groups import ModelGroupsManagementClient
from .chat import ChatClient
from .keys import KeysManagementClient
from .credentials import CredentialsManagementClient


class Client:
    """Main client for interacting with the LiteLLM proxy API."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the LiteLLM proxy client.

        Args:
            base_url (str): The base URL of the LiteLLM proxy server (e.g., "http://localhost:8000")
            api_key (Optional[str]): API key for authentication. If provided, it will be sent as a Bearer token.
        """
        self._base_url = base_url.rstrip("/")  # Remove trailing slash if present
        self._api_key = api_key

        # Initialize resource clients
        self.models = ModelsManagementClient(base_url=self._base_url, api_key=self._api_key)
        self.model_groups = ModelGroupsManagementClient(base_url=self._base_url, api_key=self._api_key)
        self.chat = ChatClient(base_url=self._base_url, api_key=self._api_key)
        self.keys = KeysManagementClient(base_url=self._base_url, api_key=self._api_key)
        self.credentials = CredentialsManagementClient(base_url=self._base_url, api_key=self._api_key)
