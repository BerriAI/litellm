import requests
from typing import Dict, Any, Optional, Union

from .exceptions import UnauthorizedError


class CredentialsManagementClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the CredentialsManagementClient.

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
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def list(
        self,
        return_request: bool = False,
    ) -> Union[Dict[str, Any], requests.Request]:
        """
        List all credentials.

        Args:
            return_request (bool): If True, returns the prepared request object instead of executing it

        Returns:
            Union[Dict[str, Any], requests.Request]: Either the response from the server or
            a prepared request object if return_request is True

        Raises:
            UnauthorizedError: If the request fails with a 401 status code
            requests.exceptions.RequestException: If the request fails with any other error
        """
        url = f"{self._base_url}/credentials"

        request = requests.Request("GET", url, headers=self._get_headers())

        if return_request:
            return request

        session = requests.Session()
        try:
            response = session.send(request.prepare())
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise UnauthorizedError(e)
            raise

    def create(
        self,
        credential_name: str,
        credential_info: Dict[str, Any],
        credential_values: Dict[str, Any],
        return_request: bool = False,
    ) -> Union[Dict[str, Any], requests.Request]:
        """
        Create a new credential.

        Args:
            credential_name (str): Name of the credential
            credential_info (Dict[str, Any]): Additional information about the credential
            credential_values (Dict[str, Any]): Values for the credential
            return_request (bool): If True, returns the prepared request object instead of executing it

        Returns:
            Union[Dict[str, Any], requests.Request]: Either the response from the server or
            a prepared request object if return_request is True

        Raises:
            UnauthorizedError: If the request fails with a 401 status code
            requests.exceptions.RequestException: If the request fails with any other error
        """
        url = f"{self._base_url}/credentials"

        data = {
            "credential_name": credential_name,
            "credential_info": credential_info,
            "credential_values": credential_values,
        }

        request = requests.Request("POST", url, headers=self._get_headers(), json=data)

        if return_request:
            return request

        session = requests.Session()
        try:
            response = session.send(request.prepare())
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise UnauthorizedError(e)
            raise

    def delete(
        self,
        credential_name: str,
        return_request: bool = False,
    ) -> Union[Dict[str, Any], requests.Request]:
        """
        Delete a credential by name.

        Args:
            credential_name (str): Name of the credential to delete
            return_request (bool): If True, returns the prepared request object instead of executing it

        Returns:
            Union[Dict[str, Any], requests.Request]: Either the response from the server or
            a prepared request object if return_request is True

        Raises:
            UnauthorizedError: If the request fails with a 401 status code
            requests.exceptions.RequestException: If the request fails with any other error
        """
        url = f"{self._base_url}/credentials/{credential_name}"

        request = requests.Request("DELETE", url, headers=self._get_headers())

        if return_request:
            return request

        session = requests.Session()
        try:
            response = session.send(request.prepare())
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise UnauthorizedError(e)
            raise

    def get(
        self,
        credential_name: str,
        return_request: bool = False,
    ) -> Union[Dict[str, Any], requests.Request]:
        """
        Get a credential by name.

        Args:
            credential_name (str): Name of the credential to retrieve
            return_request (bool): If True, returns the prepared request object instead of executing it

        Returns:
            Union[Dict[str, Any], requests.Request]: Either the response from the server or
            a prepared request object if return_request is True

        Raises:
            UnauthorizedError: If the request fails with a 401 status code
            requests.exceptions.RequestException: If the request fails with any other error
        """
        url = f"{self._base_url}/credentials/by_name/{credential_name}"

        request = requests.Request("GET", url, headers=self._get_headers())

        if return_request:
            return request

        session = requests.Session()
        try:
            response = session.send(request.prepare())
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise UnauthorizedError(e)
            raise
