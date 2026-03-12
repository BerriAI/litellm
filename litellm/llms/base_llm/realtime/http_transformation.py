"""
Base transformation class for realtime HTTP endpoints (client_secrets, realtime_calls).

These are HTTP (not WebSocket) endpoints used by the WebRTC flow:
  POST /v1/realtime/client_secrets  — obtains a short-lived ephemeral key
  POST /v1/realtime/calls           — exchanges an SDP offer using that key
"""

from abc import ABC, abstractmethod
from typing import Optional, Union

import httpx


class BaseRealtimeHTTPConfig(ABC):
    """
    Abstract base for provider-specific realtime HTTP credential / URL logic.

    Implement one subclass per provider (OpenAI, Azure, …).
    """

    # ------------------------------------------------------------------ #
    # Credential resolution                                                #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_api_base(
        self,
        api_base: Optional[str],
        **kwargs,
    ) -> str:
        """
        Resolve the provider API base URL.

        Resolution order (provider-specific):
          explicit api_base → litellm.api_base → env var → hard-coded default
        """

    @abstractmethod
    def get_api_key(
        self,
        api_key: Optional[str],
        **kwargs,
    ) -> str:
        """
        Resolve the provider API key.

        Resolution order (provider-specific):
          explicit api_key → litellm.api_key → env var → ""
        """

    # ------------------------------------------------------------------ #
    # client_secrets endpoint                                              #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_complete_url(
        self, api_base: Optional[str], model: str, api_version: Optional[str] = None
    ) -> str:
        """Return the full URL for POST /realtime/client_secrets."""

    @abstractmethod
    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        """
        Build and return the request headers for the client_secrets call.

        Merge `headers` (caller-supplied extras) with auth / content-type
        headers required by this provider.
        """

    # ------------------------------------------------------------------ #
    # realtime_calls endpoint                                              #
    # ------------------------------------------------------------------ #

    def get_realtime_calls_url(
        self, api_base: Optional[str], model: str, api_version: Optional[str] = None
    ) -> str:
        """Return the full URL for POST /realtime/calls (SDP exchange)."""
        base = (api_base or "").rstrip("/")
        return f"{base}/v1/realtime/calls"

    def get_realtime_calls_headers(self, ephemeral_key: str) -> dict:
        """
        Build headers for the realtime_calls POST.

        The Bearer token here is the ephemeral key obtained from
        client_secrets, not the long-lived provider key.
        """
        return {
            "Authorization": f"Bearer {ephemeral_key}",
        }

    # ------------------------------------------------------------------ #
    # Error handling                                                      #
    # ------------------------------------------------------------------ #

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ):
        """
        Map HTTP errors to LiteLLM exception types.

        Default: generic exception. Override in subclasses for provider-specific
        error mapping (e.g., Azure uses different error codes).
        """
        from litellm.llms.base_llm.chat.transformation import BaseLLMException

        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
