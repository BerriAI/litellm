import requests
from typing import List, Dict, Any, Optional, Union
from .exceptions import UnauthorizedError


class ModelGroupsManagementClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the ModelGroupsManagementClient.

        Args:
            base_url (str): The base URL of the LiteLLM proxy server (e.g., "http://localhost:8000")
            api_key (Optional[str]): API key for authentication. If provided, it will be sent as a Bearer token.
        """
        self._base_url = base_url.rstrip("/")  # Remove trailing slash if present
        self._api_key = api_key

    def _get_headers(self) -> Dict[str, str]:
        """
        Get the headers for API requests, including authorization if api_key is set.

        Returns:
            Dict[str, str]: Headers to use for API requests
        """
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def info(self, return_request: bool = False) -> Union[List[Dict[str, Any]], requests.Request]:
        """
        Get detailed information about all model groups from the server.

        Args:
            return_request (bool): If True, returns the prepared request object instead of executing it

        Returns:
            Union[List[Dict[str, Any]], requests.Request]: Either a list of model group information dictionaries
            or a prepared request object if return_request is True

        Raises:
            UnauthorizedError: If the request fails with a 401 status code
            requests.exceptions.RequestException: If the request fails with any other error
        """
        url = f"{self._base_url}/model_group/info"
        request = requests.Request("GET", url, headers=self._get_headers())

        if return_request:
            return request

        # Prepare and send the request
        session = requests.Session()
        try:
            response = session.send(request.prepare())
            response.raise_for_status()
            return response.json()["data"]
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise UnauthorizedError(e)
            raise
