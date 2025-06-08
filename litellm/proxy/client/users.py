import requests
from typing import List, Dict, Any, Optional
from .exceptions import UnauthorizedError, NotFoundError


class UsersManagementClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def list_users(self, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List users (GET /user/list)"""
        url = f"{self.base_url}/user/list"
        response = requests.get(url, headers=self._get_headers(), params=params)
        if response.status_code == 401:
            raise UnauthorizedError(response.text)
        response.raise_for_status()
        return response.json().get("users", response.json())

    def get_user(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get user info (GET /user/info)"""
        url = f"{self.base_url}/user/info"
        params = {"user_id": user_id} if user_id else {}
        response = requests.get(url, headers=self._get_headers(), params=params)
        if response.status_code == 401:
            raise UnauthorizedError(response.text)
        if response.status_code == 404:
            raise NotFoundError(response.text)
        response.raise_for_status()
        return response.json()

    def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user (POST /user/new)"""
        url = f"{self.base_url}/user/new"
        response = requests.post(url, headers=self._get_headers(), json=user_data)
        if response.status_code == 401:
            raise UnauthorizedError(response.text)
        response.raise_for_status()
        return response.json()

    def delete_user(self, user_ids: List[str]) -> Dict[str, Any]:
        """Delete users (POST /user/delete)"""
        url = f"{self.base_url}/user/delete"
        response = requests.post(url, headers=self._get_headers(), json={"user_ids": user_ids})
        if response.status_code == 401:
            raise UnauthorizedError(response.text)
        response.raise_for_status()
        return response.json()
