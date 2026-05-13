"""
Transformation for Bedrock Invoke Agent via Anthropic /v1/messages API

Allows calling Bedrock managed agents through the Anthropic Messages API spec.

Model format: bedrock/agent/<AGENT_ID>/<ALIAS_ID>

https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_InvokeAgent.html
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.chat.invoke_agent.transformation import (
    AmazonInvokeAgentConfig,
)
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
    AnthropicResponseTextBlock,
    AnthropicUsage,
)
from litellm.types.router import GenericLiteLLMParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class AmazonInvokeAgentMessagesConfig(BaseAnthropicMessagesConfig, BaseAWSLLM):
    """
    Anthropic /v1/messages interface for Bedrock managed agents (InvokeAgent API).

    Model format: bedrock/agent/<AGENT_ID>/<ALIAS_ID>

    This class reuses the existing AmazonInvokeAgentConfig for URL construction
    and AWS event stream parsing, while adapting the request/response to the
    Anthropic Messages API format.
    """

    def __init__(self, **kwargs):
        BaseAnthropicMessagesConfig.__init__(self)
        BaseAWSLLM.__init__(self, **kwargs)
        self._invoke_agent_config = AmazonInvokeAgentConfig(**kwargs)

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

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        return self._invoke_agent_config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
            stream=stream,
        )

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
        return self._sign_request(
            service_name="bedrock",
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=api_base,
            model=model,
            stream=stream,
            fake_stream=fake_stream,
            api_key=api_key,
        )

    def get_supported_anthropic_messages_params(self, model: str) -> list:
        return []

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """
        Transform Anthropic messages format to InvokeAgent request body.

        Extracts the last user message as the inputText for the agent.
        """
        last_message_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    last_message_content = content
                elif isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    last_message_content = "\n".join(text_parts)
                break

        if not last_message_content:
            last_message_content = convert_content_list_to_str(messages[-1])

        return {
            "inputText": last_message_content,
            "enableTrace": True,
        }

    def transform_anthropic_messages_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> AnthropicMessagesResponse:
        """
        Transform InvokeAgent AWS event stream response to Anthropic Messages format.
        """
        try:
            raw_content = raw_response.content
            verbose_logger.debug(
                "Processing %d bytes of AWS event stream data for agent messages response",
                len(raw_content),
            )

            events = self._invoke_agent_config._parse_aws_event_stream(raw_content)
            verbose_logger.debug("Parsed %d events from agent stream", len(events))

            content = self._invoke_agent_config._extract_response_content(events)
            usage_info = self._invoke_agent_config._extract_usage_info(events)

            response_model = usage_info.get("model") or model

            text_block = AnthropicResponseTextBlock(
                type="text",
                text=content,
            )

            usage = AnthropicUsage(
                input_tokens=usage_info.get("inputTokens", 0),
                output_tokens=usage_info.get("outputTokens", 0),
            )

            return AnthropicMessagesResponse(
                id=f"msg_{str(uuid.uuid4())}",
                type="message",
                role="assistant",
                content=[text_block],
                model=response_model,
                stop_reason="end_turn",
                usage=usage,
            )

        except Exception as e:
            verbose_logger.error(
                "Error processing Bedrock Invoke Agent messages response: %s", str(e)
            )
            raise BedrockError(
                message=f"Error processing response: {str(e)}",
                status_code=raw_response.status_code,
            )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> Any:
        return BedrockError(status_code=status_code, message=error_message)
