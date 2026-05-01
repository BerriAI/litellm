from typing import TYPE_CHECKING, Any, List, Optional

import httpx

from litellm.anthropic_beta_headers_manager import filter_and_transform_beta_headers
from litellm.litellm_core_utils.prompt_templates.factory import (
    convert_to_anthropic_image_obj,
)
from litellm.litellm_core_utils.prompt_templates.image_handling import (
    async_convert_url_to_base64,
    convert_url_to_base64,
)
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
)
from litellm.llms.bedrock.common_utils import (
    get_anthropic_beta_from_headers,
    normalize_tool_input_schema_types_for_bedrock_invoke,
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

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        # Force tool-based structured outputs for Bedrock Invoke
        # (similar to VertexAI fix in #19201)
        # Bedrock Invoke doesn't support output_format parameter
        original_model = model
        if "response_format" in non_default_params:
            # Use a model name that forces tool-based approach
            model = "claude-3-sonnet-20240229"

        optional_params = AnthropicConfig.map_openai_params(
            self,
            non_default_params,
            optional_params,
            model,
            drop_params,
        )

        # Restore original model name
        model = original_model

        return optional_params

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        _anthropic_request = self._build_bedrock_anthropic_request_base(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        self._convert_document_url_sources_to_base64(_anthropic_request)
        beta_list = self._compute_bedrock_invoke_beta_headers(
            model=model,
            messages=messages,
            optional_params=optional_params,
            headers=headers,
        )
        if beta_list:
            _anthropic_request["anthropic_beta"] = beta_list

        return _anthropic_request

    async def async_transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        _anthropic_request = self._build_bedrock_anthropic_request_base(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        await self._async_convert_document_url_sources_to_base64(_anthropic_request)
        beta_list = self._compute_bedrock_invoke_beta_headers(
            model=model,
            messages=messages,
            optional_params=optional_params,
            headers=headers,
        )
        if beta_list:
            _anthropic_request["anthropic_beta"] = beta_list

        return _anthropic_request

    def _build_bedrock_anthropic_request_base(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        filtered_params = {
            k: v
            for k, v in optional_params.items()
            if k not in self.aws_authentication_params
        }
        filtered_params = self._normalize_bedrock_tool_search_tools(filtered_params)

        anthropic_request = AnthropicConfig.transform_request(
            self,
            model=model,
            messages=messages,
            optional_params=filtered_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        anthropic_request.pop("model", None)
        anthropic_request.pop("stream", None)
        anthropic_request.pop("output_format", None)
        anthropic_request.pop("output_config", None)
        if "anthropic_version" not in anthropic_request:
            anthropic_request["anthropic_version"] = self.anthropic_version

        # Remove `custom` field from tools (Bedrock doesn't support it)
        remove_custom_field_from_tools(anthropic_request)
        normalize_tool_input_schema_types_for_bedrock_invoke(anthropic_request)
        return anthropic_request

    def _compute_bedrock_invoke_beta_headers(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        headers: dict,
    ) -> List[str]:
        tools = optional_params.get("tools")
        tool_search_used = self.is_tool_search_used(tools)
        programmatic_tool_calling_used = self.is_programmatic_tool_calling_used(tools)
        input_examples_used = self.is_input_examples_used(tools)

        user_beta_set = set(get_anthropic_beta_from_headers(headers))
        beta_set = set(user_beta_set)
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

        auto_beta_list = filter_and_transform_beta_headers(
            beta_headers=list(beta_set - user_beta_set),
            provider="bedrock",
        )
        return sorted(user_beta_set.union(set(auto_beta_list)))

    def _convert_document_url_sources_to_base64(self, anthropic_request: dict) -> None:
        """
        Bedrock Invoke does not accept document URL sources. Convert to base64 payloads.
        """
        messages = anthropic_request.get("messages")
        if not isinstance(messages, list):
            return

        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict) or block.get("type") != "document":
                    continue
                source = block.get("source")
                if not isinstance(source, dict) or source.get("type") != "url":
                    continue
                source_url = source.get("url")
                if not isinstance(source_url, str):
                    continue

                inferred_format: Optional[str] = None
                if source_url.lower().endswith(".pdf"):
                    inferred_format = "application/pdf"
                base64_url = convert_url_to_base64(url=source_url)
                image_chunk = convert_to_anthropic_image_obj(
                    openai_image_url=base64_url,
                    format=inferred_format,
                )
                block["source"] = {
                    "type": "base64",
                    "media_type": image_chunk["media_type"],
                    "data": image_chunk["data"],
                }

    async def _async_convert_document_url_sources_to_base64(
        self, anthropic_request: dict
    ) -> None:
        """
        Async version of document URL conversion for async completion paths.
        """
        messages = anthropic_request.get("messages")
        if not isinstance(messages, list):
            return

        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict) or block.get("type") != "document":
                    continue
                source = block.get("source")
                if not isinstance(source, dict) or source.get("type") != "url":
                    continue
                source_url = source.get("url")
                if not isinstance(source_url, str):
                    continue

                inferred_format: Optional[str] = None
                if source_url.lower().endswith(".pdf"):
                    inferred_format = "application/pdf"
                base64_url = await async_convert_url_to_base64(url=source_url)
                image_chunk = convert_to_anthropic_image_obj(
                    openai_image_url=base64_url,
                    format=inferred_format,
                )
                block["source"] = {
                    "type": "base64",
                    "media_type": image_chunk["media_type"],
                    "data": image_chunk["data"],
                }

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
