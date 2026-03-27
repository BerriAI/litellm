from typing import TYPE_CHECKING, Any, List, Optional

import httpx

from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
)
from litellm.llms.bedrock.common_utils import (
    add_additional_properties_to_schema,
    get_anthropic_beta_from_headers,
    remove_custom_field_from_tools,
)
from litellm.types.llms.anthropic import ANTHROPIC_TOOL_SEARCH_BETA_HEADER
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

# Anthropic Claude models that support native structured outputs on Bedrock InvokeModel.
# Maintained separately from the Converse path's BEDROCK_NATIVE_STRUCTURED_OUTPUT_MODELS
# because Invoke and Converse have independent feature rollouts.
# Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html
BEDROCK_INVOKE_NATIVE_STRUCTURED_OUTPUT_MODELS = {
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-5",
    "claude-opus-4-6",
}


class AmazonAnthropicClaudeConfig(AmazonInvokeConfig, AnthropicConfig):
    """
    Reference:
        https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=claude
        https://docs.anthropic.com/claude/docs/models-overview#model-comparison
        https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-request-response.html

    Supported Params for the Amazon / Anthropic Claude models (Claude 3, Claude 4, etc.):
    Supports anthropic_beta parameter for beta features like:
    - computer-use-2025-01-24 (Claude 3.7 Sonnet)
    - computer-use-2024-10-22 (Claude 3.5 Sonnet v2)
    - token-efficient-tools-2025-02-19 (Claude 3.7 Sonnet)
    - interleaved-thinking-2025-05-14 (Claude 4 models)
    - output-128k-2025-02-19 (Claude 3.7 Sonnet)
    - dev-full-thinking-2025-05-14 (Claude 4 models)
    - context-1m-2025-08-07 (Claude Sonnet 4)
    """

    anthropic_version: str = "bedrock-2023-05-31"

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "bedrock"

    def get_supported_openai_params(self, model: str) -> List[str]:
        return AnthropicConfig.get_supported_openai_params(self, model)

    @staticmethod
    def _supports_native_structured_outputs(model: str) -> bool:
        """Check if the Bedrock Invoke model supports native structured outputs."""
        return any(
            substring in model
            for substring in BEDROCK_INVOKE_NATIVE_STRUCTURED_OUTPUT_MODELS
        )

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        response_format = non_default_params.get("response_format")

        # Native path: build output_format directly for Bedrock-supported models
        # (includes haiku-4-5 which the Anthropic parent doesn't know about).
        if isinstance(
            response_format, dict
        ) and self._supports_native_structured_outputs(model):
            _output_format = self.map_response_format_to_anthropic_output_format(
                response_format
            )
            if _output_format is not None:
                optional_params["output_format"] = _output_format
                optional_params["json_mode"] = True
                remaining = {
                    k: v
                    for k, v in non_default_params.items()
                    if k != "response_format"
                }
                return AnthropicConfig.map_openai_params(
                    self,
                    remaining,
                    optional_params,
                    model,
                    drop_params,
                )

        # Fallback: force tool-based structured outputs for unsupported models
        # (or json_object without schema on a supported model).
        if response_format is not None:
            model = "claude-3-sonnet-20240229"

        return AnthropicConfig.map_openai_params(
            self,
            non_default_params,
            optional_params,
            model,
            drop_params,
        )

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        # Filter out AWS authentication parameters before passing to Anthropic transformation
        # AWS params should only be used for signing requests, not included in request body
        filtered_params = {
            k: v
            for k, v in optional_params.items()
            if k not in self.aws_authentication_params
        }
        filtered_params = self._normalize_bedrock_tool_search_tools(filtered_params)

        _anthropic_request = AnthropicConfig.transform_request(
            self,
            model=model,
            messages=messages,
            optional_params=filtered_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        _anthropic_request.pop("model", None)
        _anthropic_request.pop("stream", None)

        # Convert Anthropic output_format to Bedrock InvokeModel output_config.format
        output_format = _anthropic_request.pop("output_format", None)
        if (
            output_format
            and isinstance(output_format, dict)
            and output_format.get("type") == "json_schema"
        ):
            schema = output_format.get("schema", {})
            normalized_schema = add_additional_properties_to_schema(schema)
            # Preserve existing output_config keys (e.g. effort from reasoning_effort)
            output_config = _anthropic_request.get("output_config") or {}
            output_config["format"] = {
                "type": "json_schema",
                "schema": normalized_schema,
            }
            _anthropic_request["output_config"] = output_config
        else:
            # Non-native path: strip output_config entirely.
            # Bedrock Invoke rejects the key itself (not just sub-keys) with
            # "extraneous key [output_config] is not permitted" for models
            # that don't support native structured outputs.
            # Fixes: https://github.com/BerriAI/litellm/issues/22797
            _anthropic_request.pop("output_config", None)
        if "anthropic_version" not in _anthropic_request:
            _anthropic_request["anthropic_version"] = self.anthropic_version

        # Remove `custom` field from tools (Bedrock doesn't support it)
        # Claude Code sends `custom: {defer_loading: true}` on tool definitions,
        # which causes Bedrock to reject the request with "Extra inputs are not permitted"
        # Ref: https://github.com/BerriAI/litellm/issues/22847
        remove_custom_field_from_tools(_anthropic_request)

        tools = optional_params.get("tools")
        tool_search_used = self.is_tool_search_used(tools)
        programmatic_tool_calling_used = self.is_programmatic_tool_calling_used(tools)
        input_examples_used = self.is_input_examples_used(tools)

        beta_set = set(get_anthropic_beta_from_headers(headers))
        auto_betas = self.get_anthropic_beta_list(
            model=model,
            optional_params=optional_params,
            computer_tool_used=self.is_computer_tool_used(tools),
            prompt_caching_set=False,
            file_id_used=self.is_file_id_used(messages),
            mcp_server_used=self.is_mcp_server_used(optional_params.get("mcp_servers")),
        )
        beta_set.update(auto_betas)

        if tool_search_used and not (
            programmatic_tool_calling_used or input_examples_used
        ):
            beta_set.discard(ANTHROPIC_TOOL_SEARCH_BETA_HEADER)
            if "opus-4" in model.lower() or "opus_4" in model.lower():
                beta_set.add("tool-search-tool-2025-10-19")

        # Filter out beta headers that Bedrock Invoke doesn't support
        # Uses centralized configuration from anthropic_beta_headers_config.json
        beta_list = list(beta_set)
        _anthropic_request["anthropic_beta"] = beta_list

        return _anthropic_request

    def _normalize_bedrock_tool_search_tools(self, optional_params: dict) -> dict:
        """
        Convert tool search entries to the format supported by the Bedrock Invoke API.
        """
        tools = optional_params.get("tools")
        if not tools or not isinstance(tools, list):
            return optional_params

        normalized_tools = []
        for tool in tools:
            tool_type = tool.get("type")
            if tool_type == "tool_search_tool_bm25_20251119":
                # Bedrock Invoke does not support the BM25 variant, so skip it.
                continue
            if tool_type == "tool_search_tool_regex_20251119":
                normalized_tool = tool.copy()
                normalized_tool["type"] = "tool_search_tool_regex"
                normalized_tool["name"] = normalized_tool.get(
                    "name", "tool_search_tool_regex"
                )
                normalized_tools.append(normalized_tool)
                continue
            normalized_tools.append(tool)

        optional_params["tools"] = normalized_tools
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
        return AnthropicConfig.transform_response(
            self,
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )
