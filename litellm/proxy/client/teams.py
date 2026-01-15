"""Teams management client for LiteLLM proxy."""

from typing import Any, Dict, List, Optional, Union

import requests

from .exceptions import UnauthorizedError


class TeamsManagementClient:
    """Client for managing teams in LiteLLM proxy."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the TeamsManagementClient.

        Args:
            base_url (str): The base URL of the LiteLLM proxy server (e.g., "http://localhost:4000")
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
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List teams that the user belongs to.

        Args:
            user_id (Optional[str]): Only return teams which this user belongs to
            organization_id (Optional[str]): Only return teams which belong to this organization

        Returns:
            List[Dict[str, Any]]: List of team objects

        Raises:
            requests.exceptions.HTTPError: If the request fails
            UnauthorizedError: If authentication fails
        """
        url = f"{self._base_url}/team/list"
        params = {}
        if user_id:
            params["user_id"] = user_id
        if organization_id:
            params["organization_id"] = organization_id

        response = requests.get(url, headers=self._get_headers(), params=params)
        
        if response.status_code == 401:
            raise UnauthorizedError("Authentication failed. Check your API key.")
        
        response.raise_for_status()
        return response.json()

    def list_v2(
        self,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        team_id: Optional[str] = None,
        team_alias: Optional[str] = None,
        page: int = 1,
        page_size: int = 10,
        sort_by: Optional[str] = None,
        sort_order: str = "asc",
    ) -> Dict[str, Any]:
        """
        Get a paginated list of teams with filtering and sorting options.

        Args:
            user_id (Optional[str]): Only return teams which this user belongs to
            organization_id (Optional[str]): Only return teams which belong to this organization
            team_id (Optional[str]): Filter teams by exact team_id match
            team_alias (Optional[str]): Filter teams by partial team_alias match
            page (int): Page number for pagination
            page_size (int): Number of teams per page
            sort_by (Optional[str]): Column to sort by (e.g. 'team_id', 'team_alias', 'created_at')
            sort_order (str): Sort order ('asc' or 'desc')

        Returns:
            Dict[str, Any]: Paginated response containing teams and pagination info

        Raises:
            requests.exceptions.HTTPError: If the request fails
            UnauthorizedError: If authentication fails
        """
        url = f"{self._base_url}/v2/team/list"
        params: Dict[str, Union[str, int]] = {
            "page": page,
            "page_size": page_size,
            "sort_order": sort_order,
        }
        
        if user_id:
            params["user_id"] = user_id
        if organization_id:
            params["organization_id"] = organization_id
        if team_id:
            params["team_id"] = team_id
        if team_alias:
            params["team_alias"] = team_alias
        if sort_by:
            params["sort_by"] = sort_by

        response = requests.get(url, headers=self._get_headers(), params=params)
        
        if response.status_code == 401:
            raise UnauthorizedError("Authentication failed. Check your API key.")
        
        response.raise_for_status()
        return response.json()

    def get_available(self) -> List[Dict[str, Any]]:
        """
        Get list of available teams that the user can join.

        Returns:
            List[Dict[str, Any]]: List of available team objects

        Raises:
            requests.exceptions.HTTPError: If the request fails
            UnauthorizedError: If authentication fails
        """
        url = f"{self._base_url}/team/available"
        
        response = requests.get(url, headers=self._get_headers())
        
        if response.status_code == 401:
            raise UnauthorizedError("Authentication failed. Check your API key.")
        
        response.raise_for_status()
        return response.json()
