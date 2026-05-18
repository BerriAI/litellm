"""
Google AI Studio Agents API configuration.

Proxies the Gemini v1beta Agents API:
  POST   https://generativelanguage.googleapis.com/v1beta/agents
  GET    https://generativelanguage.googleapis.com/v1beta/agents/{name}
  DELETE https://generativelanguage.googleapis.com/v1beta/agents/{name}

The exact response schema is not publicly documented; we accept whatever
fields the server returns and surface them verbatim under
litellm_params["provider_agent_response"].
"""

from typing import Any, Dict, Optional, Union

import httpx

from litellm._logging import verbose_logger
from litellm.llms.base_llm.agents.transformation import BaseAgentsAPIConfig
from litellm.llms.gemini.common_utils import GeminiError, GeminiModelInfo
from litellm.types.agents import AgentCreateResponse


# Keys inside litellm_params that are Gemini-specific agent fields and should
# be forwarded to the provider create-agent body verbatim.
_GEMINI_AGENT_BODY_KEYS = ("base_agent", "instructions", "base_environment")

# Keys that are LiteLLM internal and must never be forwarded to Gemini.
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

    # ------------------------------------------------------------------ #
    # BaseAgentsAPIConfig interface                                        #
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
        resolved = GeminiModelInfo.get_api_base(api_base)
        return f"{resolved}/{self.api_version}/agents"

    def validate_environment(
        self,
        headers: Dict[str, str],
        litellm_params: Dict[str, Any],
    ) -> Dict[str, str]:
        headers = dict(headers)
        headers["Content-Type"] = "application/json"
        api_key = GeminiModelInfo.get_api_key(litellm_params.get("api_key"))
        if not api_key:
            raise ValueError(
                "Google API key is required to create a Gemini agent. "
                "Set GOOGLE_API_KEY or GEMINI_API_KEY, or pass api_key in litellm_params."
            )
        headers["x-goog-api-key"] = api_key
        return headers

    def transform_create_request(
        self,
        name: str,
        litellm_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Map LiteLLM fields to the Gemini create-agent body.

        Required:
          name        ← name

        Optional (from litellm_params):
          base_agent        ← litellm_params["base_agent"]
          instructions      ← litellm_params["instructions"]
          base_environment  ← litellm_params["base_environment"]
        """
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
        Parse Gemini's create-agent response.

        The exact schema is not published; all returned fields are preserved
        via AgentCreateResponse's extra="allow" config.  On non-2xx we raise
        GeminiError so the outer handler can surface a clean HTTP error.
        """
        if not (200 <= raw_response.status_code < 300):
            raise GeminiError(
                message=raw_response.text,
                status_code=raw_response.status_code,
                headers=dict(raw_response.headers),
            )

        try:
            data: Dict[str, Any] = raw_response.json()
        except Exception:
            verbose_logger.warning(
                "GeminiAgentsConfig: could not parse JSON from create-agent response "
                "(status=%d). Using name as fallback.",
                raw_response.status_code,
            )
            data = {"name": name}

        data.setdefault("name", name)

        verbose_logger.debug("GeminiAgentsConfig create response: %s", data)
        return AgentCreateResponse(**data)
