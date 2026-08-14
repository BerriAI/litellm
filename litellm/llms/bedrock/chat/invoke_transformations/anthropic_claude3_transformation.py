from typing import TYPE_CHECKING, Any, Final

import httpx

from litellm.anthropic_beta_headers_manager import filter_and_transform_beta_headers
from litellm.litellm_core_utils.litellm_logging import verbose_logger
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
    convert_bedrock_invoke_output_format_to_inline_schema,
    get_anthropic_beta_from_headers,
    normalize_bedrock_opus_output_config_effort,
    normalize_custom_field_on_tools,
    normalize_tool_input_schema_types_for_bedrock_invoke,
    pop_bedrock_invoke_output_config_format,
)
from litellm.types.llms.anthropic import ANTHROPIC_TOOL_SEARCH_BETA_HEADER
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse
from litellm.utils import _supports_factory

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
    def custom_llm_provider(self) -> str | None:
        return "bedrock"

    def should_strip_billing_metadata(self) -> bool:
        return True

    def get_supported_openai_params(self, model: str) -> list[str]:
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
        original_model: Final = model
        if "response_format" in non_default_params:
            # Use a model name that forces tool-based approach
            model = "claude-3-sonnet-20240229"

        # Clamp ``reasoning_effort`` to the Bedrock effort ceiling before the
        # parent mapping converts it to ``output_config.effort`` and the
        # downstream effort gate runs. Mirrors the converse path's
        # ``_handle_reasoning_effort_parameter`` and the messages path's
        # ``_clamp_adaptive_reasoning_effort_for_bedrock`` so adaptive Claude
        # requests degrade ``xhigh`` -> ``max`` rather than 400-ing on
        # models like Opus 4.6 that don't natively advertise xhigh.
        self._clamp_adaptive_reasoning_effort_for_bedrock(model=original_model, params=non_default_params)

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

    @staticmethod
    def _clamp_adaptive_reasoning_effort_for_bedrock(model: str, params: dict) -> None:
        """Lower ``reasoning_effort`` to the Bedrock effort ceiling before mapping.

        Bedrock's adaptive Claude models accept the OpenAI-style
        ``reasoning_effort`` tier, but the request validator can reject tiers
        the model does not natively advertise (e.g. ``xhigh`` on Opus 4.6).
        Clamp the raw tier to the model's
        ``bedrock_output_config_effort_ceiling`` so Claude Code "goal mode"
        keeps working. Non-adaptive models and models without a ceiling are
        left untouched.
        """
        if not AnthropicConfig._is_adaptive_thinking_model(model, "bedrock"):
            return
        effort: Final = params.get("reasoning_effort")
        if not isinstance(effort, str):
            return
        clamped: Final = {"effort": effort}
        normalize_bedrock_opus_output_config_effort(model=model, output_config=clamped)
        params["reasoning_effort"] = clamped["effort"]

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        _anthropic_request: Final = self._build_bedrock_anthropic_request_base(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        self._convert_document_url_sources_to_base64(_anthropic_request)
        beta_list: Final = self._compute_bedrock_invoke_beta_headers(
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
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        _anthropic_request: Final = self._build_bedrock_anthropic_request_base(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        await self._async_convert_document_url_sources_to_base64(_anthropic_request)
        beta_list: Final = self._compute_bedrock_invoke_beta_headers(
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
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        filtered_params = {k: v for k, v in optional_params.items() if k not in self.aws_authentication_params}
        output_config: Final = filtered_params.get("output_config")
        if isinstance(output_config, dict):
            filtered_params["output_config"] = dict(output_config)
            normalize_bedrock_opus_output_config_effort(
                model=model,
                output_config=filtered_params["output_config"],
            )
        filtered_params = self._normalize_bedrock_tool_search_tools(filtered_params)

        anthropic_request: Final = AnthropicConfig.transform_request(
            self,
            model=model,
            messages=messages,
            optional_params=filtered_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        anthropic_request.pop("model", None)
        anthropic_request.pop("stream", None)
        anthropic_request.pop("stream_chunk_size", None)
        output_format: Final = anthropic_request.pop("output_format", None)
        output_config_format: Final = pop_bedrock_invoke_output_config_format(anthropic_request)
        if output_format:
            convert_bedrock_invoke_output_format_to_inline_schema(
                output_format=output_format,
                request_body=anthropic_request,
            )
        elif output_config_format:
            convert_bedrock_invoke_output_format_to_inline_schema(
                output_format=output_config_format,
                request_body=anthropic_request,
            )
        if not (
            _supports_factory(
                model=model,
                custom_llm_provider="bedrock",
                key="supports_output_config",
            )
            or AnthropicConfig._model_supports_effort_param(model, "bedrock")
        ):
            if anthropic_request.pop("output_config", None) is not None:
                verbose_logger.warning(
                    "Bedrock Invoke: stripping unsupported `output_config` for "
                    "model=%s — neither `supports_output_config` nor any "
                    "`supports_*_reasoning_effort` flag is set in "
                    "model_prices_and_context_window.json. Add the capability "
                    "flag to the model JSON entry if this model accepts "
                    "`output_config`.",
                    model,
                )
        if "anthropic_version" not in anthropic_request:
            anthropic_request["anthropic_version"] = self.anthropic_version

        # Hoist `custom.defer_loading` then drop `custom` (Bedrock doesn't support it)
        normalize_custom_field_on_tools(anthropic_request)
        normalize_tool_input_schema_types_for_bedrock_invoke(anthropic_request)
        return anthropic_request

    def _compute_bedrock_invoke_beta_headers(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        headers: dict,
    ) -> list[str]:
        tools: Final = optional_params.get("tools")
        tool_search_used: Final = self.is_tool_search_used(tools)
        programmatic_tool_calling_used: Final = self.is_programmatic_tool_calling_used(tools)
        input_examples_used: Final = self.is_input_examples_used(tools)

        user_beta_set: Final = set(get_anthropic_beta_from_headers(headers))
        beta_set: Final = set(user_beta_set)
        auto_betas: Final = self.get_anthropic_beta_list(
            model=model,
            optional_params=optional_params,
            computer_tool_used=self.is_computer_tool_used(tools),
            prompt_caching_set=False,
            file_id_used=self.is_file_id_used(messages),
            mcp_server_used=self.is_mcp_server_used(optional_params.get("mcp_servers")),
            custom_llm_provider="bedrock",
        )
        beta_set.update(auto_betas)

        if tool_search_used and not (programmatic_tool_calling_used or input_examples_used):
            beta_set.discard(ANTHROPIC_TOOL_SEARCH_BETA_HEADER)
            if "opus-4" in model.lower() or "opus_4" in model.lower():
                beta_set.add("tool-search-tool-2025-10-19")

        auto_beta_list: Final = filter_and_transform_beta_headers(
            beta_headers=list(beta_set - user_beta_set),
            provider="bedrock",
        )
        return sorted(user_beta_set.union(set(auto_beta_list)))

    def _convert_document_url_sources_to_base64(self, anthropic_request: dict) -> None:
        """
        Bedrock Invoke does not accept document URL sources. Convert to base64 payloads.
        """
        messages: Final = anthropic_request.get("messages")
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

                inferred_format: str | None = None
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

    async def _async_convert_document_url_sources_to_base64(self, anthropic_request: dict) -> None:
        """
        Async version of document URL conversion for async completion paths.
        """
        messages: Final = anthropic_request.get("messages")
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

                inferred_format: str | None = None
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
        tools: Final = optional_params.get("tools")
        if not tools or not isinstance(tools, list):
            return optional_params

        normalized_tools: Final = []
        for tool in tools:
            tool_type = tool.get("type")
            if tool_type == "tool_search_tool_bm25_20251119":
                # Bedrock Invoke does not support the BM25 variant, so skip it.
                continue
            if tool_type == "tool_search_tool_regex_20251119":
                normalized_tool = tool.copy()
                normalized_tool["type"] = "tool_search_tool_regex"
                normalized_tool["name"] = normalized_tool.get("name", "tool_search_tool_regex")
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
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: str | None = None,
        json_mode: bool | None = None,
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
