"""
HTTP client for the Qualifire Studio API.
Handles sync and async calls to the /compile endpoint.
"""

from typing import Any, Dict, List, Optional

import httpx

from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.llms.custom_http import httpxSpecialProvider


class QualifireClient:
    """
    Low-level HTTP client for the Qualifire API.

    Uses the /api/v1/studio/prompts/{promptId}/compile endpoint
    to compile prompts with variable substitution server-side.
    """

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.qualifire.ai",
    ):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Qualifire-API-Key": self.api_key,
        }

    def _build_compile_url(self, prompt_id: str) -> str:
        """Build the compile endpoint URL for a given prompt ID."""
        return f"{self.api_base}/api/v1/studio/prompts/{prompt_id}/compile"

    def _build_request_body(
        self,
        variables: Optional[Dict[str, Any]] = None,
        revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the request body for the compile endpoint."""
        body: Dict[str, Any] = {}
        if variables:
            body["variables"] = variables
        if revision:
            body["revision"] = revision
        return body

    def _handle_error_response(self, response: httpx.Response, prompt_id: str) -> None:
        """Handle HTTP error responses with specific messages."""
        if response.status_code == 401:
            raise Exception(
                f"Authentication failed for Qualifire API. "
                f"Please check your API key."
            )
        elif response.status_code == 403:
            raise Exception(
                f"Access denied to prompt '{prompt_id}'. "
                f"Please check your permissions."
            )
        elif response.status_code == 404:
            raise Exception(
                f"Prompt '{prompt_id}' not found in Qualifire. "
                f"Please check the prompt ID."
            )
        response.raise_for_status()

    def compile_prompt(
        self,
        prompt_id: str,
        variables: Optional[Dict[str, Any]] = None,
        revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compile a prompt by calling the Qualifire API synchronously.

        Args:
            prompt_id: The Qualifire prompt CUID
            variables: Variables for template substitution
            revision: Optional revision CUID to pin a specific version

        Returns:
            The compiled prompt response from Qualifire
        """
        url = self._build_compile_url(prompt_id)
        body = self._build_request_body(variables, revision)

        http_client = _get_httpx_client()

        try:
            response = http_client.post(
                url,
                json=body,
                headers=self._get_headers(),
            )

            if response.status_code >= 400:
                self._handle_error_response(response, prompt_id)

            return response.json()
        except httpx.HTTPError as e:
            raise Exception(
                f"Failed to compile prompt '{prompt_id}' from Qualifire: {e}"
            )

    async def async_compile_prompt(
        self,
        prompt_id: str,
        variables: Optional[Dict[str, Any]] = None,
        revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compile a prompt by calling the Qualifire API asynchronously.

        Args:
            prompt_id: The Qualifire prompt CUID
            variables: Variables for template substitution
            revision: Optional revision CUID to pin a specific version

        Returns:
            The compiled prompt response from Qualifire
        """
        url = self._build_compile_url(prompt_id)
        body = self._build_request_body(variables, revision)

        http_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.PromptManagement,
        )

        try:
            response = await http_client.post(
                url,
                json=body,
                headers=self._get_headers(),
            )

            if response.status_code >= 400:
                self._handle_error_response(response, prompt_id)

            return response.json()
        except httpx.HTTPError as e:
            raise Exception(
                f"Failed to compile prompt '{prompt_id}' from Qualifire: {e}"
            )
