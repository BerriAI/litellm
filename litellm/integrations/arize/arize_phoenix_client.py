"""
Arize Phoenix API client for fetching prompt versions from Arize Phoenix.
"""

from typing import Any, Dict, Optional

from litellm.llms.custom_httpx.http_handler import HTTPHandler


class ArizePhoenixClient:
    """
    Client for interacting with Arize Phoenix API to fetch prompt versions.

    Supports:
    - Authentication with Bearer tokens
    - Fetching prompt versions
    - Direct API base URL configuration
    """

    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None):
        """
        Initialize the Arize Phoenix client.

        Args:
            api_key: Arize Phoenix API token
            api_base: Base URL for the Arize Phoenix API (e.g., 'https://app.phoenix.arize.com/s/workspace/v1')
        """
        self.api_key = api_key
        self.api_base = api_base

        if not self.api_key:
            raise ValueError("api_key is required")

        if not self.api_base:
            raise ValueError("api_base is required")

        # Set up authentication headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

        # Initialize HTTPHandler
        self.http_handler = HTTPHandler(disable_default_headers=True)

    def get_prompt_version(self, prompt_version_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a prompt version from Arize Phoenix.

        Args:
            prompt_version_id: The ID of the prompt version to fetch

        Returns:
            Dictionary containing prompt version data, or None if not found
        """
        url = f"{self.api_base}/v1/prompt_versions/{prompt_version_id}"

        try:
            # Use the underlying httpx client directly to avoid query param extraction
            response = self.http_handler.get(url, headers=self.headers)
            response.raise_for_status()

            data = response.json()
            return data.get("data")

        except Exception as e:
            # Check if it's an HTTP error
            response = getattr(e, "response", None)
            if response is not None and hasattr(response, "status_code"):
                if response.status_code == 404:
                    return None
                elif response.status_code == 403:
                    raise Exception(
                        f"Access denied to prompt version '{prompt_version_id}'. Check your Arize Phoenix permissions."
                    )
                elif response.status_code == 401:
                    raise Exception(
                        "Authentication failed. Check your Arize Phoenix API key and permissions."
                    )
                else:
                    raise Exception(
                        f"Failed to fetch prompt version '{prompt_version_id}': {e}"
                    )
            else:
                raise Exception(
                    f"Error fetching prompt version '{prompt_version_id}': {e}"
                )

    def test_connection(self) -> bool:
        """
        Test the connection to the Arize Phoenix API.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Try to access the prompt_versions endpoint to test connection
            url = f"{self.api_base}/prompt_versions"
            response = self.http_handler.client.get(url, headers=self.headers)
            response.raise_for_status()
            return True
        except Exception:
            return False

    def close(self):
        """Close the HTTP handler to free resources."""
        if hasattr(self, "http_handler"):
            self.http_handler.close()
