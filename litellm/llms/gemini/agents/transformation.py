"""
Google AI Studio Agents API configuration.

Proxies the Gemini v1beta Agents API:
  POST   /v1beta/agents                         create
  GET    /v1beta/agents                         list
  GET    /v1beta/agents/{name}                  get
  DELETE /v1beta/agents/{name}                  delete
  GET    /v1beta/agents/{name}/versions         list versions
"""

from typing import Any, Dict, Optional, Tuple, Union

import httpx

from litellm._logging import verbose_logger
from litellm.llms.base_llm.agents.transformation import BaseAgentsAPIConfig
from litellm.llms.gemini.common_utils import GeminiError, GeminiModelInfo
from litellm.types.agents import (
    AgentCreateResponse,
    AgentDeleteResult,
    AgentListResponse,
    AgentVersionsResponse,
)


# Keys inside litellm_params that should be forwarded to the Gemini
# create-agent body verbatim.
_GEMINI_AGENT_BODY_KEYS = ("base_agent", "instructions", "base_environment")

# LiteLLM-internal keys that must never be forwarded to Gemini.
_LITELLM_INTERNAL_KEYS = frozenset(
    {
        "custom_llm_provider",
        "api_key",
        "api_base",
        "make_public",
        "cost_per_query",
        "input_cost_per_token",
        "output_cost_per_token",
        "require_trace_id_on_calls_to_agent",
        "require_trace_id_on_calls_by_agent",
        "max_iterations",
        "max_budget_per_session",
        "guardrails",
        "is_public",
        "agent_name",
        "agent_id",
        "agent_card_params",
        "provider_agent_response",
    }
)


class GeminiAgentsConfig(BaseAgentsAPIConfig):
    """
    Configuration for the Google AI Studio (Gemini) native Agents API.

    Authentication uses x-goog-api-key, resolved from (in order):
      1. litellm_params["api_key"]
      2. GOOGLE_API_KEY env var
      3. GEMINI_API_KEY env var
    """

    @property
    def api_version(self) -> str:
        return "v1beta"

    def _base_url(self, api_base: Optional[str]) -> str:
        return f"{GeminiModelInfo.get_api_base(api_base)}/{self.api_version}"

    # ------------------------------------------------------------------ #
    # Shared helpers                                                       #
    # ------------------------------------------------------------------ #

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> Exception:
        return GeminiError(
            message=error_message,
            status_code=status_code,
            headers=dict(headers),
        )

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: Dict[str, Any],
    ) -> str:
        return f"{self._base_url(api_base)}/agents"

    def validate_environment(
        self,
        headers: Dict[str, str],
        litellm_params: Dict[str, Any],
    ) -> Dict[str, str]:
        headers = dict(headers)
        headers["Content-Type"] = "application/json"
        explicit_api_key = litellm_params.get("api_key")
        # SECURITY: when the caller overrides ``api_base``, refuse to fall back
        # to the process-wide GOOGLE_API_KEY / GEMINI_API_KEY env vars. Otherwise
        # an authenticated proxy user could set ``api_base`` to an attacker-
        # controlled host and have the proxy ship its shared Gemini key in the
        # ``x-goog-api-key`` header.
        if litellm_params.get("api_base") and not explicit_api_key:
            raise ValueError(
                "When overriding api_base for Gemini agents, you must also "
                "supply an explicit api_key. Falling back to GOOGLE_API_KEY / "
                "GEMINI_API_KEY env vars with a custom api_base is refused "
                "to prevent leaking the shared provider key to arbitrary hosts."
            )
        api_key = GeminiModelInfo.get_api_key(explicit_api_key)
        if not api_key:
            raise ValueError(
                "Google API key is required. "
                "Set GOOGLE_API_KEY or GEMINI_API_KEY, or pass api_key."
            )
        headers["x-goog-api-key"] = api_key
        return headers

    def _raise_for_status(self, raw_response: httpx.Response) -> None:
        if not (200 <= raw_response.status_code < 300):
            raise GeminiError(
                message=raw_response.text,
                status_code=raw_response.status_code,
                headers=dict(raw_response.headers),
            )

    # ------------------------------------------------------------------ #
    # CREATE                                                               #
    # ------------------------------------------------------------------ #

    def transform_create_request(
        self,
        name: str,
        litellm_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"name": name}
        for key in _GEMINI_AGENT_BODY_KEYS:
            value = litellm_params.get(key)
            if value is not None:
                body[key] = value
        verbose_logger.debug("GeminiAgentsConfig create body: %s", body)
        return body

    def transform_create_response(
        self,
        raw_response: httpx.Response,
        name: str,
    ) -> AgentCreateResponse:
        """
        Gemini returns:
          {"id": "my-agent", "base_agent": "waverunner",
           "system_instruction": "...", "base_environment": {...}}
        """
        self._raise_for_status(raw_response)
        try:
            data: Dict[str, Any] = raw_response.json()
        except Exception:
            verbose_logger.warning(
                "GeminiAgentsConfig: non-JSON create response (status=%d).",
                raw_response.status_code,
            )
            data = {"id": name}
        # Gemini uses "id" as the identifier; normalise to both fields.
        data.setdefault("id", name)
        data.setdefault("name", data["id"])
        verbose_logger.debug("GeminiAgentsConfig create response: %s", data)
        return AgentCreateResponse(**data)

    # ------------------------------------------------------------------ #
    # LIST                                                                 #
    # ------------------------------------------------------------------ #

    def transform_list_request(
        self,
        api_base: Optional[str],
        litellm_params: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        url = f"{self._base_url(api_base)}/agents"
        params: Dict[str, Any] = {}
        if litellm_params.get("page_size"):
            params["pageSize"] = litellm_params["page_size"]
        if litellm_params.get("page_token"):
            params["pageToken"] = litellm_params["page_token"]
        return url, params

    def transform_list_response(
        self,
        raw_response: httpx.Response,
    ) -> AgentListResponse:
        self._raise_for_status(raw_response)
        try:
            data = raw_response.json()
        except Exception:
            data = {}
        verbose_logger.debug("GeminiAgentsConfig list response: %s", data)
        return AgentListResponse(
            agents=data.get("agents", []),
            next_page_token=data.get("nextPageToken"),
        )

    # ------------------------------------------------------------------ #
    # GET                                                                  #
    # ------------------------------------------------------------------ #

    def transform_get_request(
        self,
        name: str,
        api_base: Optional[str],
        litellm_params: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        url = f"{self._base_url(api_base)}/agents/{name}"
        return url, {}

    def transform_get_response(
        self,
        raw_response: httpx.Response,
        name: str,
    ) -> AgentCreateResponse:
        """Same shape as create response — Gemini returns "id" as identifier."""
        self._raise_for_status(raw_response)
        try:
            data = raw_response.json()
        except Exception:
            data = {"id": name}
        data.setdefault("id", name)
        data.setdefault("name", data["id"])
        verbose_logger.debug("GeminiAgentsConfig get response: %s", data)
        return AgentCreateResponse(**data)

    # ------------------------------------------------------------------ #
    # DELETE                                                               #
    # ------------------------------------------------------------------ #

    def transform_delete_request(
        self,
        name: str,
        api_base: Optional[str],
        litellm_params: Dict[str, Any],
    ) -> str:
        return f"{self._base_url(api_base)}/agents/{name}"

    def transform_delete_response(
        self,
        raw_response: httpx.Response,
        name: str,
    ) -> AgentDeleteResult:
        """Gemini returns an empty body ``{}`` with HTTP 200 on success."""
        self._raise_for_status(raw_response)
        verbose_logger.debug(
            "GeminiAgentsConfig delete (status=%d) agent '%s'",
            raw_response.status_code,
            name,
        )
        return AgentDeleteResult(name=name, deleted=True)

    # ------------------------------------------------------------------ #
    # LIST VERSIONS                                                        #
    # ------------------------------------------------------------------ #

    def transform_list_versions_request(
        self,
        name: str,
        api_base: Optional[str],
        litellm_params: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        url = f"{self._base_url(api_base)}/agents/{name}/versions"
        params: Dict[str, Any] = {}
        if litellm_params.get("page_size"):
            params["pageSize"] = litellm_params["page_size"]
        if litellm_params.get("page_token"):
            params["pageToken"] = litellm_params["page_token"]
        return url, params

    def transform_list_versions_response(
        self,
        raw_response: httpx.Response,
        name: str,
    ) -> AgentVersionsResponse:
        """
        Gemini returns:
          {"agentVersions": [{"agent": "waverunner", "name": "agents/.../versions/uuid", ...}]}
        """
        self._raise_for_status(raw_response)
        try:
            data = raw_response.json()
        except Exception:
            data = {}
        verbose_logger.debug(
            "GeminiAgentsConfig list_versions response for '%s': %s", name, data
        )
        return AgentVersionsResponse(
            agent_versions=data.get("agentVersions", []),
            next_page_token=data.get("nextPageToken"),
        )
