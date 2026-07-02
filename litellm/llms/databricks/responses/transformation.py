"""
Databricks Responses API configurations (Unity AI Gateway + serving-endpoints).

`litellm.responses(model="databricks/...")` routes by model family to one of three
surfaces (all OpenAI-Responses wire format, so transforms are inherited from
:class:`OpenAIResponsesAPIConfig`; only the URL/surface differs):

* gpt-N (non-oss)  -> native OpenAI Responses  ``<host>/ai-gateway/openai/v1/responses``
                      (:class:`DatabricksResponsesAPIConfig`)
* Claude           -> Supervisor API           ``<host>/ai-gateway/mlflow/v1/responses``
                      (:class:`DatabricksSupervisorResponsesAPIConfig`) — allowlisted
                      (Claude + gpt-5 full + qwen35).
* Gemini / gpt-oss -> Open Responses (unified) ``<host>/serving-endpoints/open-responses``
  / qwen / llama /   (:class:`DatabricksOpenResponsesAPIConfig`)
  gemma / others

Surface scopes were verified live against a Databricks workspace: the unified Open
Responses API lives only on ``/serving-endpoints`` (there is no
``/ai-gateway/open-responses``), Supervisor is allowlisted, and the native OpenAI
Responses path is GPT-only. An explicit ``/serving-endpoints`` or custom-path base is
honored for backward compatibility.

Reference: https://docs.databricks.com/aws/en/machine-learning/model-serving/query-open-responses-models
"""

import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from litellm.llms.databricks.common_utils import DatabricksBase
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import ResponseInputParam
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

from ..ai_gateway import (
    AI_GATEWAY_PATHS,
    has_explicit_custom_path,
    normalize_gateway_base,
    parse_use_ai_gateway_flag,
    resolve_surface,
    workspace_host_from_base,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class _DatabricksResponsesBase(DatabricksBase, OpenAIResponsesAPIConfig):
    """Shared Databricks Responses behavior. Subclasses set the surface via
    ``gateway_path`` / ``serving_url_suffix`` / ``force_surface``."""

    # AI Gateway path appended to ``<host>/ai-gateway`` when on the gateway surface.
    gateway_path: str = AI_GATEWAY_PATHS["openai_responses"]
    # Full suffix appended to ``<host>`` when on the serving-endpoints surface.
    serving_url_suffix: str = "/serving-endpoints/responses"
    # Pin a surface ("ai_gateway" | "serving_endpoints"); ``None`` = resolve via the
    # optimistic gateway-first cache lookup (no network probe).
    force_surface: Optional[str] = None

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.DATABRICKS

    def get_responses_surface_fallbacks(self, model: str) -> list:
        """Per-family ordered responses-surface chain (see ``fallback.py``).

        No single Databricks responses surface serves every model, so
        ``litellm.responses()`` tries each surface in turn and, when all reject
        the model, falls through to chat-completions emulation.
        """
        from .fallback import databricks_responses_config_chain

        return databricks_responses_config_chain(model)

    def should_fallback_on_responses_error(self, exc: Exception) -> bool:
        """True when an error means "this surface does not serve this model"."""
        from .fallback import is_surface_unavailable_error

        return is_surface_unavailable_error(exc)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = litellm_params.api_key or os.getenv("DATABRICKS_API_KEY")
        api_base = litellm_params.api_base or os.getenv("DATABRICKS_API_BASE")

        # Reuse the shared Databricks auth core (M2M / PAT / PROFILE / SDK unified).
        _, headers = self.databricks_resolve_auth(
            api_key=api_key,
            api_base=api_base,
            custom_endpoint=False,
            headers=headers,
            databricks_profile=self.resolve_databricks_profile(litellm_params),
        )
        headers["Content-Type"] = "application/json"
        headers = self.apply_request_tags_header(headers, litellm_params=litellm_params)
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        user_api_base = api_base or os.getenv("DATABRICKS_API_BASE")

        # Explicit custom (non-surface) base: preserve the legacy responses URL.
        if user_api_base and has_explicit_custom_path(user_api_base):
            return f"{user_api_base.rstrip('/')}/responses"

        resolved_base = self._get_api_base(user_api_base)
        host = workspace_host_from_base(resolved_base)

        if self.force_surface in ("ai_gateway", "serving_endpoints"):
            surface = self.force_surface
        else:
            params = litellm_params if isinstance(litellm_params, dict) else {}
            use_ai_gateway = parse_use_ai_gateway_flag(params, None)
            # Pure, no-network resolution: optimistic gateway-first unless the host
            # is cached gateway-absent. When the gateway surface rejects the model,
            # the responses() surface-fallback chain retries the next surface.
            surface = resolve_surface(
                api_base=user_api_base,
                use_ai_gateway=use_ai_gateway,
                host=host,
            )
        if surface == "ai_gateway":
            return normalize_gateway_base(host) + self.gateway_path
        return f"{host}{self.serving_url_suffix}"

    def transform_responses_api_request(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        response_api_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        # Strip the provider prefix (e.g. "databricks/databricks-gpt-5" ->
        # "databricks-gpt-5") then delegate to the OpenAI Responses transform.
        if model.startswith("databricks/"):
            model = model[len("databricks/") :]
        return super().transform_responses_api_request(
            model=model,
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def supports_native_websocket(self) -> bool:
        """Databricks does not support native WebSocket for Responses API."""
        return False


class DatabricksResponsesAPIConfig(_DatabricksResponsesBase):
    """Native OpenAI Responses for gpt-N (non-oss) models on the AI Gateway:
    ``<host>/ai-gateway/openai/v1/responses`` (serving fallback
    ``<host>/serving-endpoints/responses``)."""

    gateway_path: str = AI_GATEWAY_PATHS["openai_responses"]
    serving_url_suffix: str = "/serving-endpoints/responses"
    force_surface: Optional[str] = None


class DatabricksSupervisorResponsesAPIConfig(_DatabricksResponsesBase):
    """Supervisor Responses API on the AI Gateway:
    ``<host>/ai-gateway/mlflow/v1/responses``.

    Primary surface for **Claude** (verified). Allowlisted (Claude + gpt-5 full +
    qwen35); exists only on the AI Gateway, so the surface is pinned to the gateway.
    """

    gateway_path: str = AI_GATEWAY_PATHS["supervisor_responses"]
    force_surface: Optional[str] = "ai_gateway"


class DatabricksOpenResponsesAPIConfig(_DatabricksResponsesBase):
    """Unified Open Responses API on serving-endpoints:
    ``<host>/serving-endpoints/open-responses``.

    Primary surface for **Gemini** and open models (gpt-oss, qwen3-next, …). As of
    the live verification this cross-provider unified surface exists only on
    serving-endpoints (``/ai-gateway/open-responses`` returns 404 "Invalid path"),
    so the surface is pinned accordingly.

    TODO(ai-gateway-open-responses): the AI Gateway is expected to expose Open
    Responses soon. When it lands (and the exact path is confirmed), set
    ``gateway_path`` to the real gateway path and ``force_surface = None`` so this
    resolves gateway-first with the serving-endpoints path below as the fallback —
    no other change needed. Until then keep it pinned to serving-endpoints.
    """

    serving_url_suffix: str = "/serving-endpoints/open-responses"
    force_surface: Optional[str] = "serving_endpoints"
