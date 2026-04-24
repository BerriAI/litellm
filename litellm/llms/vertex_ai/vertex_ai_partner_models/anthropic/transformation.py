# What is this?
## Handler file for calling claude-3 on vertex ai
from typing import Any, List, Optional

import httpx

import litellm
from litellm.llms.base_llm.chat.transformation import LiteLLMLoggingObj
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ....anthropic.chat.transformation import AnthropicConfig

# Keys inside ``output_config`` that Vertex AI Claude does not accept.
# Today only ``effort`` triggers "Extra inputs are not permitted"; add new
# entries here as Vertex parity drifts. Keep this list narrow — anything
# Vertex DOES accept (e.g. ``format`` for structured outputs) must be
# preserved so callers can rely on Anthropic-native features.
_VERTEX_UNSUPPORTED_OUTPUT_CONFIG_KEYS = frozenset({"effort"})


def _sanitize_vertex_anthropic_output_params(data: dict) -> None:
    """
    Strip Vertex-unsupported keys from ``output_config`` / ``output_format``
    in-place; forward whatever remains.

    Behavior:
      * ``output_config`` containing only unsupported keys (e.g. ``effort``
        alone) is removed entirely so the request body has no empty dict.
      * ``output_config`` containing a mix of supported + unsupported keys has
        the unsupported subset filtered out and the rest forwarded.
      * ``output_config`` that is supported in full passes through unchanged.
      * ``output_format`` is forwarded as-is (Vertex AI Claude accepts it).
      * Non-dict values for ``output_config`` are dropped to avoid sending
        malformed payloads downstream.
    """
    output_config = data.get("output_config")
    if output_config is None:
        return
    if not isinstance(output_config, dict):
        data.pop("output_config", None)
        return
    sanitized = {
        k: v
        for k, v in output_config.items()
        if k not in _VERTEX_UNSUPPORTED_OUTPUT_CONFIG_KEYS
    }
    if sanitized:
        data["output_config"] = sanitized
    else:
        data.pop("output_config", None)


class VertexAIError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url=" https://cloud.google.com/vertex-ai/"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class VertexAIAnthropicConfig(AnthropicConfig):
    """
    Reference:https://docs.anthropic.com/claude/reference/messages_post

    Note that the API for Claude on Vertex differs from the Anthropic API documentation in the following ways:

    - `model` is not a valid parameter. The model is instead specified in the Google Cloud endpoint URL.
    - `anthropic_version` is a required parameter and must be set to "vertex-2023-10-16".

    The class `VertexAIAnthropicConfig` provides configuration for the VertexAI's Anthropic API interface. Below are the parameters:

    - `max_tokens` Required (integer) max tokens,
    - `anthropic_version` Required (string) version of anthropic for bedrock - e.g. "bedrock-2023-05-31"
    - `system` Optional (string) the system prompt, conversion from openai format to this is handled in factory.py
    - `temperature` Optional (float) The amount of randomness injected into the response
    - `top_p` Optional (float) Use nucleus sampling.
    - `top_k` Optional (int) Only sample from the top K options for each subsequent token
    - `stop_sequences` Optional (List[str]) Custom text sequences that cause the model to stop generating

    Note: Please make sure to modify the default parameters as required for your use case.
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "vertex_ai"

    def _add_context_management_beta_headers(
        self, beta_set: set, context_management: dict
    ) -> None:
        """
        Add context_management beta headers to the beta_set.

        - If any edit has type "compact_20260112", add compact-2026-01-12 header
        - For all other edits, add context-management-2025-06-27 header

        Args:
            beta_set: Set of beta headers to modify in-place
            context_management: The context_management dict from optional_params
        """
        from litellm.types.llms.anthropic import ANTHROPIC_BETA_HEADER_VALUES

        edits = context_management.get("edits", [])
        has_compact = False
        has_other = False

        for edit in edits:
            edit_type = edit.get("type", "")
            if edit_type == "compact_20260112":
                has_compact = True
            else:
                has_other = True

        # Add compact header if any compact edits exist
        if has_compact:
            beta_set.add(ANTHROPIC_BETA_HEADER_VALUES.COMPACT_2026_01_12.value)

        # Add context management header if any other edits exist
        if has_other:
            beta_set.add(
                ANTHROPIC_BETA_HEADER_VALUES.CONTEXT_MANAGEMENT_2025_06_27.value
            )

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        data = super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        data.pop("model", None)  # vertex anthropic doesn't accept 'model' parameter

        # Vertex AI Claude accepts ``output_config.format`` (structured outputs /
        # JSON Schema) but NOT ``output_config.effort`` — sending ``effort`` to
        # Vertex returns 400 "Extra inputs are not permitted". Sanitize in place:
        # forward the structured-output bits, drop the unsupported keys.
        # Same treatment for the legacy top-level ``output_format`` field.
        _sanitize_vertex_anthropic_output_params(data)

        tools = optional_params.get("tools")
        tool_search_used = self.is_tool_search_used(tools)
        auto_betas = self.get_anthropic_beta_list(
            model=model,
            optional_params=optional_params,
            computer_tool_used=self.is_computer_tool_used(tools),
            prompt_caching_set=self.is_cache_control_set(messages),
            file_id_used=self.is_file_id_used(messages),
            mcp_server_used=self.is_mcp_server_used(optional_params.get("mcp_servers")),
        )

        beta_set = set(auto_betas)
        if tool_search_used:
            beta_set.add(
                "tool-search-tool-2025-10-19"
            )  # Vertex requires this header for tool search

        # Add context_management beta headers (compact and/or context-management)
        context_management = optional_params.get("context_management")
        if context_management:
            self._add_context_management_beta_headers(beta_set, context_management)

        extra_headers = optional_params.get("extra_headers") or {}
        anthropic_beta_value = extra_headers.get("anthropic-beta", "")
        if isinstance(anthropic_beta_value, str) and anthropic_beta_value:
            for beta in anthropic_beta_value.split(","):
                beta = beta.strip()
                if beta:
                    beta_set.add(beta)
        elif isinstance(anthropic_beta_value, list):
            beta_set.update(anthropic_beta_value)

        data.pop("extra_headers", None)

        if beta_set:
            data["anthropic_beta"] = list(beta_set)
            headers["anthropic-beta"] = ",".join(beta_set)

        return data

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Override parent method to ensure VertexAI always uses tool-based structured outputs.
        VertexAI doesn't support the output_format parameter, so we force all models
        to use the tool-based approach for structured outputs.
        """
        # Temporarily override model name to force tool-based approach
        # This ensures Claude Sonnet 4.5 uses tools instead of output_format
        original_model = model
        if "response_format" in non_default_params:
            model = "claude-3-sonnet-20240229"  # Use a model that will use tool-based approach

        # Call parent method with potentially modified model name
        optional_params = super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )

        # Restore original model name for any other processing
        model = original_model

        return optional_params

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        response = super().transform_response(
            model,
            raw_response,
            model_response,
            logging_obj,
            request_data,
            messages,
            optional_params,
            litellm_params,
            encoding,
            api_key,
            json_mode,
        )
        response.model = model

        return response

    @classmethod
    def is_supported_model(cls, model: str, custom_llm_provider: str) -> bool:
        """
        Check if the model is supported by the VertexAI Anthropic API.
        """
        if (
            custom_llm_provider != "vertex_ai"
            and custom_llm_provider != "vertex_ai_beta"
        ):
            return False
        if "claude" in model.lower():
            return True
        elif model in litellm.vertex_anthropic_models:
            return True
        return False
