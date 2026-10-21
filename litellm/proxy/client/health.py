from typing import Any, Final

from .http_client import HTTPClient


class HealthManagementClient:
    """
    Client for interacting with the health endpoints of the LiteLLM proxy server.
    """

    def __init__(self, base_url: str, api_key: str | None = None, timeout: int = 30):
        """
        Initialize the HealthManagementClient.

        Args:
            base_url (str): The base URL of the LiteLLM proxy server (e.g., "http://localhost:4000")
            api_key (Optional[str]): API key for authentication. If provided, it will be sent as a Bearer token.
            timeout (int): Request timeout in seconds (default: 30)
        """
        self._http = HTTPClient(base_url=base_url, api_key=api_key, timeout=timeout)

    def get_readiness(self) -> dict[str, Any]:
        """
        Check the readiness of the LiteLLM proxy server.

        Returns:
            Dict[str, Any]: The readiness status and details from the server.

        Raises:
            requests.exceptions.RequestException: If the request fails
            ValueError: If the response is not valid JSON
        """
        return self._http.request("GET", "/health/readiness")

    def get_server_version(self) -> str | None:
        """
        Get the LiteLLM server version from the readiness endpoint.

        Returns:
            Optional[str]: The server version if available, otherwise None.
        """
        readiness: Final = self.get_readiness()
        return readiness.get("litellm_version")
