"""
Base transformation class for provider-side Agents API.

Providers that have a native agents CRUD API (e.g. Gemini v1beta/agents)
subclass BaseAgentsAPIConfig and implement the abstract methods.

The HTTP calls are handled by AgentsHTTPHandler — this class is pure
transform logic (same separation as BaseInteractionsAPIConfig /
InteractionsHTTPHandler).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import httpx

from litellm.types.agents import (
    AgentCreateResponse,
    AgentDeleteResult,
    AgentListResponse,
    AgentVersionsResponse,
)


class BaseAgentsAPIConfig(ABC):
    """
    Minimal interface for providers that expose a native agents CRUD API.
    """

    # ------------------------------------------------------------------ #
    # CREATE                                                               #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: Dict[str, Any],
    ) -> str:
        """Return the full URL for POST /agents (create)."""

    @abstractmethod
    def validate_environment(
        self,
        headers: Dict[str, str],
        litellm_params: Dict[str, Any],
    ) -> Dict[str, str]:
        """Validate credentials and return auth headers."""

    @abstractmethod
    def transform_create_request(
        self,
        name: str,
        litellm_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Map name + litellm_params to the provider's create-agent body."""

    @abstractmethod
    def transform_create_response(
        self,
        raw_response: httpx.Response,
        name: str,
    ) -> AgentCreateResponse:
        """Parse create response. Raise on non-2xx."""

    # ------------------------------------------------------------------ #
    # LIST                                                                 #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def transform_list_request(
        self,
        api_base: Optional[str],
        litellm_params: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Return (url, query_params) for GET /agents."""

    @abstractmethod
    def transform_list_response(
        self,
        raw_response: httpx.Response,
    ) -> AgentListResponse:
        """Parse list-agents response. Raise on non-2xx."""

    # ------------------------------------------------------------------ #
    # GET                                                                  #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def transform_get_request(
        self,
        name: str,
        api_base: Optional[str],
        litellm_params: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Return (url, query_params) for GET /agents/{name}."""

    @abstractmethod
    def transform_get_response(
        self,
        raw_response: httpx.Response,
        name: str,
    ) -> AgentCreateResponse:
        """Parse get-agent response. Raise on non-2xx."""

    # ------------------------------------------------------------------ #
    # DELETE                                                               #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def transform_delete_request(
        self,
        name: str,
        api_base: Optional[str],
        litellm_params: Dict[str, Any],
    ) -> str:
        """Return the URL for DELETE /agents/{name}."""

    @abstractmethod
    def transform_delete_response(
        self,
        raw_response: httpx.Response,
        name: str,
    ) -> AgentDeleteResult:
        """Parse delete-agent response. Raise on non-2xx."""

    # ------------------------------------------------------------------ #
    # LIST VERSIONS                                                        #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def transform_list_versions_request(
        self,
        name: str,
        api_base: Optional[str],
        litellm_params: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Return (url, query_params) for GET /agents/{name}/versions."""

    @abstractmethod
    def transform_list_versions_response(
        self,
        raw_response: httpx.Response,
        name: str,
    ) -> AgentVersionsResponse:
        """Parse list-versions response. Raise on non-2xx."""

    # ------------------------------------------------------------------ #
    # ERROR HANDLING                                                       #
    # ------------------------------------------------------------------ #

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> Exception:
        """Map HTTP error status codes to provider-specific exceptions."""
        from litellm.llms.base_llm.chat.transformation import BaseLLMException

        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
