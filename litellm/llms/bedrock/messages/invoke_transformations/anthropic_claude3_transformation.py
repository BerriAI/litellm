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
from litellm.llms.bedrock.common_utils import get_anthropic_beta_from_headers
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
            anthropic_messages_request["anthropic_version"] = (
                self.DEFAULT_BEDROCK_ANTHROPIC_API_VERSION
            )

        # 2. `stream` is not allowed in request body for bedrock invoke
        if "stream" in anthropic_messages_request:
            anthropic_messages_request.pop("stream", None)

        # 3. `model` is not allowed in request body for bedrock invoke
        if "model" in anthropic_messages_request:
            anthropic_messages_request.pop("model", None)
            
        # 4. AUTO-INJECT beta headers based on features used
        anthropic_model_info = AnthropicModelInfo()
        tools = anthropic_messages_optional_request_params.get("tools")
        messages_typed = cast(List[AllMessageValues], messages)
        tool_search_used = anthropic_model_info.is_tool_search_used(tools)
        programmatic_tool_calling_used = anthropic_model_info.is_programmatic_tool_calling_used(
            tools
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

        if (
            tool_search_used
            and not (programmatic_tool_calling_used or input_examples_used)
        ):
            beta_set.discard(ANTHROPIC_TOOL_SEARCH_BETA_HEADER)
            if "opus-4" in model.lower() or "opus_4" in model.lower():
                beta_set.add("tool-search-tool-2025-10-19")

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
