"""
BitBucket API client for fetching .prompt files from BitBucket repositories.
"""

import base64
from typing import Any, Dict, List, Optional

from litellm.llms.custom_httpx.http_handler import HTTPHandler


class BitBucketClient:
    """
    Client for interacting with BitBucket API to fetch .prompt files.

    Supports:
    - Authentication with access tokens
    - Fetching file contents from repositories
    - Team-based access control through BitBucket permissions
    - Branch-specific file fetching
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the BitBucket client.

        Args:
            config: Dictionary containing:
                - workspace: BitBucket workspace name
                - repository: Repository name
                - access_token: BitBucket access token (or app password)
                - branch: Branch to fetch from (default: main)
                - base_url: Custom BitBucket API base URL (optional)
                - auth_method: Authentication method ('token' or 'basic', default: 'token')
                - username: Username for basic auth (optional)
        """
        self.workspace = config.get("workspace")
        self.repository = config.get("repository")
        self.access_token = config.get("access_token")
        self.branch = config.get("branch", "main")
        self.base_url = config.get("", "https://api.bitbucket.org/2.0")
        self.auth_method = config.get("auth_method", "token")
        self.username = config.get("username")

        if not all([self.workspace, self.repository, self.access_token]):
            raise ValueError("workspace, repository, and access_token are required")

        # Set up authentication headers
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        if self.auth_method == "basic" and self.username:
            # Use basic auth with username and app password
            credentials = f"{self.username}:{self.access_token}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            self.headers["Authorization"] = f"Basic {encoded_credentials}"
        else:
            # Use token-based authentication (default)
            self.headers["Authorization"] = f"Bearer {self.access_token}"

        # Initialize HTTPHandler
        self.http_handler = HTTPHandler()

    def get_file_content(self, file_path: str) -> Optional[str]:
        """
        Fetch the content of a file from the BitBucket repository.

        Args:
            file_path: Path to the file in the repository

        Returns:
            File content as string, or None if file not found
        """
        url = f"{self.base_url}/repositories/{self.workspace}/{self.repository}/src/{self.branch}/{file_path}"

        try:
            response = self.http_handler.get(url, headers=self.headers)
            response.raise_for_status()

            # BitBucket returns file content as base64 encoded
            if response.headers.get("content-type", "").startswith("text/"):
                return response.text
            else:
                # For binary files or when content-type is not text, try to decode as base64
                try:
                    return base64.b64decode(response.content).decode("utf-8")
                except Exception:
                    return response.text

        except Exception as e:
            # Check if it's an HTTP error
            if hasattr(e, "response") and hasattr(e.response, "status_code"):
                if e.response.status_code == 404:
                    return None
                elif e.response.status_code == 403:
                    raise Exception(
                        f"Access denied to file '{file_path}'. Check your BitBucket permissions for workspace '{self.workspace}' and repository '{self.repository}'."
                    )
                elif e.response.status_code == 401:
                    raise Exception(
                        "Authentication failed. Check your BitBucket access token and permissions."
                    )
                else:
                    raise Exception(f"Failed to fetch file '{file_path}': {e}")
            else:
                raise Exception(f"Error fetching file '{file_path}': {e}")

    def list_files(
        self, directory_path: str = "", file_extension: str = ".prompt"
    ) -> List[str]:
        """
        List files in a directory with a specific extension.

        Args:
            directory_path: Directory path in the repository (empty for root)
            file_extension: File extension to filter by (default: .prompt)

        Returns:
            List of file paths
        """
        url = f"{self.base_url}/repositories/{self.workspace}/{self.repository}/src/{self.branch}/{directory_path}"

        try:
            response = self.http_handler.get(url, headers=self.headers)
            response.raise_for_status()

            data = response.json()
            files = []

            for item in data.get("values", []):
                if item.get("type") == "commit_file":
                    file_path = item.get("path", "")
                    if file_path.endswith(file_extension):
                        files.append(file_path)

            return files

        except Exception as e:
            # Check if it's an HTTP error
            if hasattr(e, "response") and hasattr(e.response, "status_code"):
                if e.response.status_code == 404:
                    return []
                elif e.response.status_code == 403:
                    raise Exception(
                        f"Access denied to directory '{directory_path}'. Check your BitBucket permissions for workspace '{self.workspace}' and repository '{self.repository}'."
                    )
                elif e.response.status_code == 401:
                    raise Exception(
                        "Authentication failed. Check your BitBucket access token and permissions."
                    )
                else:
                    raise Exception(f"Failed to list files in '{directory_path}': {e}")
            else:
                raise Exception(f"Error listing files in '{directory_path}': {e}")

    def get_repository_info(self) -> Dict[str, Any]:
        """
        Get information about the repository.

        Returns:
            Dictionary containing repository information
        """
        url = f"{self.base_url}/repositories/{self.workspace}/{self.repository}"

        try:
            response = self.http_handler.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to get repository info: {e}")

    def test_connection(self) -> bool:
        """
        Test the connection to the BitBucket repository.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            self.get_repository_info()
            return True
        except Exception:
            return False

    def get_branches(self) -> List[Dict[str, Any]]:
        """
        Get list of branches in the repository.

        Returns:
            List of branch information dictionaries
        """
        url = f"{self.base_url}/repositories/{self.workspace}/{self.repository}/refs/branches"

        try:
            response = self.http_handler.get(url, headers=self.headers)
            response.raise_for_status()

            data = response.json()
            return data.get("values", [])
        except Exception as e:
            raise Exception(f"Failed to get branches: {e}")

    def get_file_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a file (size, last modified, etc.).

        Args:
            file_path: Path to the file in the repository

        Returns:
            Dictionary containing file metadata, or None if file not found
        """
        url = f"{self.base_url}/repositories/{self.workspace}/{self.repository}/src/{self.branch}/{file_path}"

        try:
            # Use GET with Range header to get just the headers (HEAD equivalent)
            headers = self.headers.copy()
            headers["Range"] = "bytes=0-0"  # Request only first byte to get headers

            response = self.http_handler.get(url, headers=headers)
            response.raise_for_status()

            return {
                "content_type": response.headers.get("content-type"),
                "content_length": response.headers.get("content-length"),
                "last_modified": response.headers.get("last-modified"),
            }
        except Exception as e:
            # Check if it's an HTTP error
            if hasattr(e, "response") and hasattr(e.response, "status_code"):
                if e.response.status_code == 404:
                    return None
                raise Exception(f"Failed to get file metadata for '{file_path}': {e}")
            else:
                raise Exception(f"Error getting file metadata for '{file_path}': {e}")

    def close(self):
        """Close the HTTP handler to free resources."""
        if hasattr(self, "http_handler"):
            self.http_handler.close()
