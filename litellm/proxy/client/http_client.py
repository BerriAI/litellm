"""HTTP client for making requests to the LiteLLM proxy server."""

from typing import Any, Dict, Optional, Union
import requests


class HTTPClient:
    """HTTP client for making requests to the LiteLLM proxy server."""

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 30):
        """Initialize the HTTP client.

        Args:
            base_url: Base URL of the LiteLLM proxy server
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds (default: 30)
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def request(
        self,
        method: str,
        uri: str,
        *,
        data: Optional[Union[Dict[str, Any], list, bytes]] = None,
        json: Optional[Union[Dict[str, Any], list]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Make an HTTP request to the LiteLLM proxy server.

        This method is used to make generic requests to the LiteLLM proxy
        server, when there is not a specific client or method for the request.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            uri: URI path (will be appended to base_url) (e.g., "/credentials")
            data: (optional) Dictionary, list of tuples, bytes, or file-like
                object to send in the body of the request.
            json: (optional) A JSON serializable Python object to send in the body
                of the request.
            headers: (optional) Dictionary of HTTP headers to send with the request.
            **kwargs: Additional keyword arguments to pass to the request.

        Returns:
            Parsed JSON response from the server

        Raises:
            requests.exceptions.RequestException: If the request fails
            ValueError: If the response is not valid JSON

        Example:
            >>> client.http.request("POST", "/health/test_connection", json={
                "litellm_params": {
                    "model": "gpt-4",
                    "custom_llm_provider": "azure_ai",
                    "litellm_credential_name": None,
                    "api_key": "6xxxxxxx",
                    "api_base": "https://litellm8397336933.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21",
                },
                "mode": "chat",
            })
            {'status': 'error',
             'result': {'model': 'gpt-4',
             'custom_llm_provider': 'azure_ai',
             'litellm_credential_name': None,
             ...
        """
        # Build complete URL
        url = f"{self._base_url}/{uri.lstrip('/')}"

        # Prepare headers
        request_headers = {}
        if headers:
            request_headers.update(headers)
        if self._api_key:
            request_headers["Authorization"] = f"Bearer {self._api_key}"

        response = requests.request(
            method=method,
            url=url,
            data=data,
            json=json,
            headers=request_headers,
            timeout=self._timeout,
            **kwargs,
        )

        # Raise for HTTP errors
        response.raise_for_status()

        # Parse and return JSON response
        return response.json()
