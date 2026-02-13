from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import httpx

from litellm.llms.anthropic.common_utils import AnthropicModelInfo
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)
from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
)
from litellm.llms.bedrock.common_utils import (
    get_anthropic_beta_from_headers,
    is_claude_4_5_on_bedrock,
)
from litellm.types.llms.anthropic import ANTHROPIC_TOOL_SEARCH_BETA_HEADER
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import GenericStreamingChunk
from litellm.types.utils import GenericStreamingChunk as GChunk
from litellm.types.utils import ModelResponseStream

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class AmazonAnthropicClaudeMessagesConfig(
    AnthropicMessagesConfig,
    AmazonInvokeConfig,
):
    """
    Call Claude model family in the /v1/messages API spec
    Supports anthropic_beta parameter for beta features.
    """

    DEFAULT_BEDROCK_ANTHROPIC_API_VERSION = "bedrock-2023-05-31"

    # Beta header patterns that are not supported by Bedrock Invoke API
    # These will be filtered out to prevent 400 "invalid beta flag" errors

    def __init__(self, **kwargs):
        BaseAnthropicMessagesConfig.__init__(self, **kwargs)
        AmazonInvokeConfig.__init__(self, **kwargs)

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[dict, Optional[str]]:
        return headers, api_base

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        fake_stream: Optional[bool] = None,
    ) -> Tuple[dict, Optional[bytes]]:
        return AmazonInvokeConfig.sign_request(
            self=self,
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=api_base,
            api_key=api_key,
            model=model,
            stream=stream,
            fake_stream=fake_stream,
        )

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        return AmazonInvokeConfig.get_complete_url(
            self=self,
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
            stream=stream,
        )

    def _remove_ttl_from_cache_control(
        self, anthropic_messages_request: Dict, model: Optional[str] = None
    ) -> None:
        """
        Remove `ttl` field from cache_control in messages.
        Bedrock doesn't support the ttl field in cache_control.

        Update: Bedock supports `5m` and `1h` for Claude 4.5 models.

        Args:
            anthropic_messages_request: The request dictionary to modify in-place
            model: The model name to check if it supports ttl
        """
        is_claude_4_5 = False
        if model:
            is_claude_4_5 = self._is_claude_4_5_on_bedrock(model)

        if "messages" in anthropic_messages_request:
            for message in anthropic_messages_request["messages"]:
                if isinstance(message, dict) and "content" in message:
                    content = message["content"]
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "cache_control" in item:
                                cache_control = item["cache_control"]
                                if (
                                    isinstance(cache_control, dict)
                                    and "ttl" in cache_control
                                ):
                                    ttl = cache_control["ttl"]
                                    if is_claude_4_5 and ttl in ["5m", "1h"]:
                                        continue

                                    cache_control.pop("ttl", None)

    def _supports_extended_thinking_on_bedrock(self, model: str) -> bool:
        """
        Check if the model supports extended thinking beta headers on Bedrock.

        On 3rd-party platforms (e.g., Amazon Bedrock), extended thinking is only
        supported on: Claude Opus 4.5, Claude Opus 4.1, Opus 4, or Sonnet 4.

        Ref: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking

        Args:
            model: The model name

        Returns:
            True if the model supports extended thinking on Bedrock
        """
        model_lower = model.lower()

        # Supported models on Bedrock for extended thinking
        supported_patterns = [
            "opus-4.5",
            "opus_4.5",
            "opus-4-5",
            "opus_4_5",  # Opus 4.5
            "opus-4.1",
            "opus_4.1",
            "opus-4-1",
            "opus_4_1",  # Opus 4.1
            "opus-4",
            "opus_4",  # Opus 4
            "sonnet-4",
            "sonnet_4",  # Sonnet 4
        ]

        return any(pattern in model_lower for pattern in supported_patterns)

    def _is_claude_opus_4_5(self, model: str) -> bool:
        """
        Check if the model is Claude Opus 4.5.

        Args:
            model: The model name

        Returns:
            True if the model is Claude Opus 4.5
        """
        model_lower = model.lower()
        opus_4_5_patterns = [
            "opus-4.5",
            "opus_4.5",
            "opus-4-5",
            "opus_4_5",
        ]
        return any(pattern in model_lower for pattern in opus_4_5_patterns)

    def _is_claude_4_5_on_bedrock(self, model: str) -> bool:
        """
        Check if the model is Claude 4.5 on Bedrock.

        Claude Sonnet 4.5, Haiku 4.5, and Opus 4.5 support 1-hour prompt caching.

        Args:
            model: The model name

        Returns:
            True if the model is Claude 4.5
        """
        return is_claude_4_5_on_bedrock(model)

    def _supports_tool_search_on_bedrock(self, model: str) -> bool:
        """
        Check if the model supports tool search on Bedrock.

        On Amazon Bedrock, server-side tool search is supported on Claude Opus 4.5
        and Claude Sonnet 4.5 with the tool-search-tool-2025-10-19 beta header.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool

        Args:
            model: The model name

        Returns:
            True if the model supports tool search on Bedrock
        """
        model_lower = model.lower()

        # Supported models for tool search on Bedrock
        supported_patterns = [
            # Opus 4.5
            "opus-4.5",
            "opus_4.5",
            "opus-4-5",
            "opus_4_5",
            # Sonnet 4.5
            "sonnet-4.5",
            "sonnet_4.5",
            "sonnet-4-5",
            "sonnet_4_5",
            # Opus 4.6
            "opus-4.6",
            "opus_4.6",
            "opus-4-6",
            "opus_4_6",
        ]

        return any(pattern in model_lower for pattern in supported_patterns)

    def _get_tool_search_beta_header_for_bedrock(
        self,
        model: str,
        tool_search_used: bool,
        programmatic_tool_calling_used: bool,
        input_examples_used: bool,
        beta_set: set,
    ) -> None:
        """
        Adjust tool search beta header for Bedrock.

        Bedrock requires a different beta header for tool search on Opus 4 models
        when tool search is used without programmatic tool calling or input examples.

        Note: On Amazon Bedrock, server-side tool search is only supported on Claude Opus 4
        with the `tool-search-tool-2025-10-19` beta header.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool

        Args:
            model: The model name
            tool_search_used: Whether tool search is used
            programmatic_tool_calling_used: Whether programmatic tool calling is used
            input_examples_used: Whether input examples are used
            beta_set: The set of beta headers to modify in-place
        """
        if tool_search_used and not (
            programmatic_tool_calling_used or input_examples_used
        ):
            beta_set.discard(ANTHROPIC_TOOL_SEARCH_BETA_HEADER)
            if "opus-4" in model.lower() or "opus_4" in model.lower():
                beta_set.add("tool-search-tool-2025-10-19")

    def _convert_output_format_to_inline_schema(
        self,
        output_format: Dict,
        anthropic_messages_request: Dict,
    ) -> None:
        """
        Convert Anthropic output_format to inline schema in message content.

        Bedrock Invoke doesn't support the output_format parameter, so we embed
        the schema directly into the user message content as text instructions.

        This approach adds the schema to the last user message, instructing the model
        to respond in the specified JSON format.

        Args:
            output_format: The output_format dict with 'type' and 'schema'
            anthropic_messages_request: The request dict to modify in-place

        Ref: https://aws.amazon.com/blogs/machine-learning/structured-data-response-with-amazon-bedrock-prompt-engineering-and-tool-use/
        """
        import json

        # Extract schema from output_format
        schema = output_format.get("schema")
        if not schema:
            return

        # Get messages from the request
        messages = anthropic_messages_request.get("messages", [])
        if not messages:
            return

        # Find the last user message
        last_user_message_idx = None
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                last_user_message_idx = idx
                break

        if last_user_message_idx is None:
            return

        last_user_message = messages[last_user_message_idx]
        content = last_user_message.get("content", [])

        # Ensure content is a list
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
            last_user_message["content"] = content

        # Add schema as text content to the message
        schema_text = {"type": "text", "text": json.dumps(schema)}
        content.append(schema_text)

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        anthropic_messages_request = AnthropicMessagesConfig.transform_anthropic_messages_request(
            self=self,
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        #########################################################
        ############## BEDROCK Invoke SPECIFIC TRANSFORMATION ###
        #########################################################

        # 1. anthropic_version is required for all claude models
        if "anthropic_version" not in anthropic_messages_request:
            anthropic_messages_request[
                "anthropic_version"
            ] = self.DEFAULT_BEDROCK_ANTHROPIC_API_VERSION

        # 2. `stream` is not allowed in request body for bedrock invoke
        if "stream" in anthropic_messages_request:
            anthropic_messages_request.pop("stream", None)

        # 3. `model` is not allowed in request body for bedrock invoke
        if "model" in anthropic_messages_request:
            anthropic_messages_request.pop("model", None)

        # 4. Remove `ttl` field from cache_control in messages (Bedrock doesn't support it for older models)
        self._remove_ttl_from_cache_control(
            anthropic_messages_request=anthropic_messages_request, model=model
        )

        # 5. Convert `output_format` to inline schema (Bedrock invoke doesn't support output_format)
        output_format = anthropic_messages_request.pop("output_format", None)
        if output_format:
            self._convert_output_format_to_inline_schema(
                output_format=output_format,
                anthropic_messages_request=anthropic_messages_request,
            )

        # 6. AUTO-INJECT beta headers based on features used
        anthropic_model_info = AnthropicModelInfo()
        tools = anthropic_messages_optional_request_params.get("tools")
        messages_typed = cast(List[AllMessageValues], messages)
        tool_search_used = anthropic_model_info.is_tool_search_used(tools)
        programmatic_tool_calling_used = (
            anthropic_model_info.is_programmatic_tool_calling_used(tools)
        )
        input_examples_used = anthropic_model_info.is_input_examples_used(tools)

        beta_set = set(get_anthropic_beta_from_headers(headers))
        auto_betas = anthropic_model_info.get_anthropic_beta_list(
            model=model,
            optional_params=anthropic_messages_optional_request_params,
            computer_tool_used=anthropic_model_info.is_computer_tool_used(tools),
            prompt_caching_set=False,
            file_id_used=anthropic_model_info.is_file_id_used(messages_typed),
            mcp_server_used=anthropic_model_info.is_mcp_server_used(
                anthropic_messages_optional_request_params.get("mcp_servers")
            ),
        )
        beta_set.update(auto_betas)

        self._get_tool_search_beta_header_for_bedrock(
            model=model,
            tool_search_used=tool_search_used,
            programmatic_tool_calling_used=programmatic_tool_calling_used,
            input_examples_used=input_examples_used,
            beta_set=beta_set,
        )

        # --- Custom logic: if tool-search-tool-2025-10-19 is present, add tool-examples-2025-10-29 ---
        if "tool-search-tool-2025-10-19" in beta_set:
            beta_set.add("tool-examples-2025-10-29")
        # ------------------------------------------------------------------------------
    
        if beta_set:
            anthropic_messages_request["anthropic_beta"] = list(beta_set)

        return anthropic_messages_request

    def get_async_streaming_response_iterator(
        self,
        model: str,
        httpx_response: httpx.Response,
        request_body: dict,
        litellm_logging_obj: LiteLLMLoggingObj,
    ) -> AsyncIterator:
        aws_decoder = AmazonAnthropicClaudeMessagesStreamDecoder(
            model=model,
        )
        completion_stream = aws_decoder.aiter_bytes(
            httpx_response.aiter_bytes(chunk_size=aws_decoder.DEFAULT_CHUNK_SIZE)
        )
        # Convert decoded Bedrock events to Server-Sent Events expected by Anthropic clients.
        return self.bedrock_sse_wrapper(
            completion_stream=completion_stream,
            litellm_logging_obj=litellm_logging_obj,
            request_body=request_body,
        )

    async def bedrock_sse_wrapper(
        self,
        completion_stream: AsyncIterator[
            Union[bytes, GenericStreamingChunk, ModelResponseStream, dict]
        ],
        litellm_logging_obj: LiteLLMLoggingObj,
        request_body: dict,
    ):
        """
        Bedrock invoke does not return SSE formatted data. This function is a wrapper to ensure litellm chunks are SSE formatted.
        """
        from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
            BaseAnthropicMessagesStreamingIterator,
        )

        handler = BaseAnthropicMessagesStreamingIterator(
            litellm_logging_obj=litellm_logging_obj,
            request_body=request_body,
        )

        async for chunk in handler.async_sse_wrapper(completion_stream):
            yield chunk


class AmazonAnthropicClaudeMessagesStreamDecoder(AWSEventStreamDecoder):
    def __init__(
        self,
        model: str,
    ) -> None:
        """
        Iterator to return Bedrock invoke response in anthropic /messages format
        """
        super().__init__(model=model)
        self.DEFAULT_CHUNK_SIZE = 1024

    def _chunk_parser(
        self, chunk_data: dict
    ) -> Union[GChunk, ModelResponseStream, dict]:
        """
        Parse the chunk data into anthropic /messages format

        Bedrock returns usage metrics using camelCase keys. Convert these to
        the Anthropic `/v1/messages` specification so callers receive a
        consistent response shape when streaming.
        """
        amazon_bedrock_invocation_metrics = chunk_data.pop(
            "amazon-bedrock-invocationMetrics", {}
        )
        if amazon_bedrock_invocation_metrics:
            anthropic_usage = {}
            if "inputTokenCount" in amazon_bedrock_invocation_metrics:
                anthropic_usage["input_tokens"] = amazon_bedrock_invocation_metrics[
                    "inputTokenCount"
                ]
            if "outputTokenCount" in amazon_bedrock_invocation_metrics:
                anthropic_usage["output_tokens"] = amazon_bedrock_invocation_metrics[
                    "outputTokenCount"
                ]
            chunk_data["usage"] = anthropic_usage
        return chunk_data
