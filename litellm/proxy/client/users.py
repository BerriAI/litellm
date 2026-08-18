from typing import Any, Final

import requests

from .exceptions import NotFoundError, UnauthorizedError


class UsersManagementClient:
    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _get_headers(self) -> dict[str, str]:
        headers: Final = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def list_users(self, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """List users (GET /user/list)"""
        url: Final = f"{self.base_url}/user/list"
        response: Final = requests.get(url, headers=self._get_headers(), params=params)
        if response.status_code == 401:
            raise UnauthorizedError(response.text)
        response.raise_for_status()
        return response.json().get("users", response.json())

    def get_user(self, user_id: str | None = None) -> dict[str, Any]:
        """Get user info (GET /user/info)"""
        url: Final = f"{self.base_url}/user/info"
        params: Final = {"user_id": user_id} if user_id else {}
        response: Final = requests.get(url, headers=self._get_headers(), params=params)
        if response.status_code == 401:
            raise UnauthorizedError(response.text)
        if response.status_code == 404:
            raise NotFoundError(response.text)
        response.raise_for_status()
        return response.json()

    def get_user_v2(self, user_id: str | None = None) -> dict[str, Any]:
        """Get user info v2 - lightweight, returns only user object (GET /v2/user/info)"""
        url: Final = f"{self.base_url}/v2/user/info"
        params: Final = {"user_id": user_id} if user_id else {}
        response: Final = requests.get(url, headers=self._get_headers(), params=params)
        if response.status_code == 401:
            raise UnauthorizedError(response.text)
        if response.status_code == 404:
            raise NotFoundError(response.text)
        response.raise_for_status()
        return response.json()

    def create_user(self, user_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new user (POST /user/new)"""
        url: Final = f"{self.base_url}/user/new"
        response: Final = requests.post(url, headers=self._get_headers(), json=user_data)
        if response.status_code == 401:
            raise UnauthorizedError(response.text)
        response.raise_for_status()
        return response.json()

    def delete_user(self, user_ids: list[str]) -> dict[str, Any]:
        """Delete users (POST /user/delete)"""
        url: Final = f"{self.base_url}/user/delete"
        response: Final = requests.post(url, headers=self._get_headers(), json={"user_ids": user_ids})
        if response.status_code == 401:
            raise UnauthorizedError(response.text)
        response.raise_for_status()
        return response.json()
