from typing import Any, Dict, List, Optional, Union

import requests

from .exceptions import UnauthorizedError


class KeysManagementClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the KeysManagementClient.

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
        page: Optional[int] = None,
        size: Optional[int] = None,
        user_id: Optional[str] = None,
        team_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        key_hash: Optional[str] = None,
        key_alias: Optional[str] = None,
        return_full_object: Optional[bool] = None,
        include_team_keys: Optional[bool] = None,
        return_request: bool = False,
    ) -> Union[Dict[str, Any], requests.Request]:
        """
        List all API keys with optional filtering and pagination.

        Args:
            page (Optional[int]): Page number for pagination
            size (Optional[int]): Number of items per page
            user_id (Optional[str]): Filter keys by user ID
            team_id (Optional[str]): Filter keys by team ID
            organization_id (Optional[str]): Filter keys by organization ID
            key_hash (Optional[str]): Filter by specific key hash
            key_alias (Optional[str]): Filter by key alias
            return_full_object (Optional[bool]): Whether to return the full key object
            include_team_keys (Optional[bool]): Whether to include team keys in the response
            return_request (bool): If True, returns the prepared request object instead of executing it

        Returns:
            Union[Dict[str, Any], requests.Request]: Either the response from the server or
            a prepared request object if return_request is True. The response contains a list
            of API keys with their configurations.

        Raises:
            UnauthorizedError: If the request fails with a 401 status code
            requests.exceptions.RequestException: If the request fails with any other error
        """
        url = f"{self._base_url}/key/list"
        params: Dict[str, Any] = {}

        # Add optional query parameters
        if page is not None:
            params["page"] = page
        if size is not None:
            params["size"] = size
        if user_id is not None:
            params["user_id"] = user_id
        if team_id is not None:
            params["team_id"] = team_id
        if organization_id is not None:
            params["organization_id"] = organization_id
        if key_hash is not None:
            params["key_hash"] = key_hash
        if key_alias is not None:
            params["key_alias"] = key_alias
        if return_full_object is not None:
            params["return_full_object"] = str(return_full_object).lower()
        if include_team_keys is not None:
            params["include_team_keys"] = str(include_team_keys).lower()

        request = requests.Request("GET", url, headers=self._get_headers(), params=params)

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

    def generate(
        self,
        models: Optional[List[str]] = None,
        aliases: Optional[Dict[str, str]] = None,
        spend: Optional[float] = None,
        duration: Optional[str] = None,
        key_alias: Optional[str] = None,
        team_id: Optional[str] = None,
        user_id: Optional[str] = None,
        budget_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        return_request: bool = False,
    ) -> Union[Dict[str, Any], requests.Request]:
        """
        Generate an API key based on the provided data.

        Docs: https://docs.litellm.ai/docs/proxy/virtual_keys

        Args:
            models (Optional[List[str]]): List of allowed models for this key
            aliases (Optional[Dict[str, str]]): Model alias mappings
            spend (Optional[float]): Maximum spend limit for this key
            duration (Optional[str]): Duration for which the key is valid (e.g. "24h", "7d")
            key_alias (Optional[str]): Alias/name for the key for easier identification
            team_id (Optional[str]): Team ID to associate the key with
            user_id (Optional[str]): User ID to associate the key with
            budget_id (Optional[str]): Budget ID to associate the key with
            config (Optional[Dict[str, Any]]): Additional configuration parameters
            return_request (bool): If True, returns the prepared request object instead of executing it

        Returns:
            Union[Dict[str, Any], requests.Request]: Either the response from the server or
            a prepared request object if return_request is True

        Raises:
            UnauthorizedError: If the request fails with a 401 status code
            requests.exceptions.RequestException: If the request fails with any other error
        """
        url = f"{self._base_url}/key/generate"

        data: Dict[str, Any] = {}
        if models is not None:
            data["models"] = models
        if aliases is not None:
            data["aliases"] = aliases
        if spend is not None:
            data["spend"] = spend
        if duration is not None:
            data["duration"] = duration
        if key_alias is not None:
            data["key_alias"] = key_alias
        if team_id is not None:
            data["team_id"] = team_id
        if user_id is not None:
            data["user_id"] = user_id
        if budget_id is not None:
            data["budget_id"] = budget_id
        if config is not None:
            data["config"] = config

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
        keys: Optional[List[str]] = None,
        key_aliases: Optional[List[str]] = None,
        return_request: bool = False,
    ) -> Union[Dict[str, Any], requests.Request]:
        """
        Delete existing keys

        Args:
            keys (List[str]): List of API keys to delete
            key_aliases (List[str]): List of key aliases to delete
            return_request (bool): If True, returns the prepared request object instead of executing it

        Returns:
            Union[Dict[str, Any], requests.Request]: Either the response from the server or
            a prepared request object if return_request is True

        Raises:
            UnauthorizedError: If the request fails with a 401 status code
            requests.exceptions.RequestException: If the request fails with any other error
        """
        url = f"{self._base_url}/key/delete"

        data = {
            "keys": keys,
            "key_aliases": key_aliases,
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

    def update(
        self,
        key: str,
        models: Optional[List[str]] = None,
        aliases: Optional[Dict[str, str]] = None,
        spend: Optional[float] = None,
        duration: Optional[str] = None,
        key_alias: Optional[str] = None,
        team_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Union[Dict[str, Any], requests.Request]:
        """
        Update an existing API key's parameters.

        Args:
            models: Optional[List[str]] = None,
            aliases: Optional[Dict[str, str]] = None,
            spend: Optional[float] = None,
            duration: Optional[str] = None,
            key_alias: Optional[str] = None,
            team_id: Optional[str] = None,
            user_id: Optional[str] = None,

        Returns:
            Union[Dict[str, Any], requests.Request]: Either the response from the server or
            a prepared request object if return_request is True

        Raises:
            UnauthorizedError: If the request fails with a 401 status code
            requests.exceptions.RequestException: If the request fails with any other error
        """
        url = f"{self._base_url}/key/update"

        data: Dict[str, Any] = {"key": key}
        
        if key_alias is not None:
            data["key_alias"] = key_alias
        if user_id is not None:
            data["user_id"] = user_id
        if team_id is not None:
            data["team_id"] = team_id
        if models is not None:
            data["models"] = models
        if spend is not None:
            data["spend"] = spend
        if duration is not None:
            data["duration"] = duration
        if aliases is not None:
            data["aliases"] = aliases
        request = requests.Request("POST", url, headers=self._get_headers(), json=data)
        session = requests.Session()
        response_text: Optional[str] = None
        try:
            response = session.send(request.prepare())
            response_text = response.text
            response.raise_for_status()
            return response.json()
        except Exception:
            raise Exception(f"Error updating key: {response_text}")


    def info(self, key: str, return_request: bool = False) -> Union[Dict[str, Any], requests.Request]:
        """
        Get information about API keys.

        Args:
            key (str): The key hash to get information about
            return_request (bool): If True, returns the prepared request object instead of executing it

        Returns:
            Union[Dict[str, Any], requests.Request]: Either the response from the server or a prepared request object if return_request is True

        Raises:
            UnauthorizedError: If the request fails with a 401 status code
            requests.exceptions.RequestException: If the request fails with any other error
        """
        url = f"{self._base_url}/key/info?key={key}"
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
