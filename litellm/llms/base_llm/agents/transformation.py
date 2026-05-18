"""
Base transformation class for provider-side Agents API.

Providers that have a native "create agent" API (e.g. Gemini v1beta/agents)
subclass BaseAgentsAPIConfig and implement the abstract methods.

The HTTP calls are handled by AgentsHTTPHandler — this class is pure
transform logic (same separation as BaseInteractionsAPIConfig /
InteractionsHTTPHandler).

If get_provider_agents_api_config() returns None for a given provider,
the create_agent endpoint falls through to the plain DB-storage path so
all existing providers (Vertex AI, LangGraph, A2A, etc.) are unaffected.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import httpx

from litellm.types.agents import AgentCreateResponse


class BaseAgentsAPIConfig(ABC):
    """
    Minimal interface for providers that expose a native agent-creation API.

    Implementations are responsible for:
    - Building the correct endpoint URL
    - Adding authentication headers
    - Serialising the create-agent request into the provider's body format
    - Deserialising the raw HTTP response into an AgentCreateResponse
    """

    @abstractmethod
    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: Dict[str, Any],
    ) -> str:
        """Return the full URL for the create-agent endpoint."""

    @abstractmethod
    def validate_environment(
        self,
        headers: Dict[str, str],
        litellm_params: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Validate credentials and return the headers dict to use for the
        HTTP request (must include auth headers).
        """

    @abstractmethod
    def transform_create_request(
        self,
        name: str,
        litellm_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Map the agent name + litellm_params to the provider's
        create-agent request body.
        """

    @abstractmethod
    def transform_create_response(
        self,
        raw_response: httpx.Response,
        name: str,
    ) -> AgentCreateResponse:
        """
        Parse the raw HTTP response into an AgentCreateResponse.
        Raise an appropriate exception on non-2xx status.
        """

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
