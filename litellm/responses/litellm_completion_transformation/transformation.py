"""
Handles transforming from Responses API -> LiteLLM completion  (Chat Completion API)
"""

import asyncio
import contextvars
from functools import partial
from typing import Any, Coroutine, Dict, Iterable, List, Literal, Optional, Union

import httpx
from openai.types.chat.completion_create_params import CompletionCreateParamsBase

import litellm
from litellm.constants import request_timeout
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.responses.streaming_iterator import (
    BaseResponsesAPIStreamingIterator,
    ResponsesAPIStreamingIterator,
    SyncResponsesAPIStreamingIterator,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionSystemMessage,
    ChatCompletionUserMessage,
    GenericChatCompletionMessage,
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager, client


class LiteLLMCompletionResponsesConfig:

    @staticmethod
    def transform_responses_api_request_to_chat_completion_request(
        model: str,
        input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """
        Transform a Responses API request into a Chat Completion request
        """
        return {
            "messages": LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
                input=input,
                responses_api_request=responses_api_request,
            ),
            "model": model,
            "tool_choice": responses_api_request.get("tool_choice"),
            "tools": responses_api_request.get("tools"),
            "top_p": responses_api_request.get("top_p"),
            "user": responses_api_request.get("user"),
            "temperature": responses_api_request.get("temperature"),
            "parallel_tool_calls": responses_api_request.get("parallel_tool_calls"),
            "max_tokens": responses_api_request.get("max_output_tokens"),
            "stream": kwargs.get("stream", False),
            "metadata": kwargs.get("metadata", {}),
            "service_tier": kwargs.get("service_tier", ""),
        }

    @staticmethod
    def transform_responses_api_input_to_messages(
        input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
    ) -> List[Union[AllMessageValues, GenericChatCompletionMessage]]:
        """
        Transform a Responses API input into a list of messages
        """
        messages: List[Union[AllMessageValues, GenericChatCompletionMessage]] = []

        # if instructions are provided, add a system message
        if responses_api_request.get("instructions"):
            messages.append(
                LiteLLMCompletionResponsesConfig.transform_instructions_to_system_message(
                    responses_api_request.get("instructions")
                )
            )

        # if input is a string, add a user message
        if isinstance(input, str):
            messages.append(ChatCompletionUserMessage(role="user", content=input))
        elif isinstance(input, list):
            for _input in input:
                messages.append(
                    GenericChatCompletionMessage(
                        role=_input.get("role") or "user",
                        content=LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(
                            _input.get("content")
                        ),
                    )
                )

        return messages

    @staticmethod
    def _transform_responses_api_content_to_chat_completion_content(
        content: Any,
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Transform a Responses API content into a Chat Completion content
        """

        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            content_list = []
            for item in content:
                if isinstance(item, str):
                    content_list.append(item)
                elif isinstance(item, dict):
                    content_list.append(
                        {
                            "type": LiteLLMCompletionResponsesConfig._get_chat_completion_request_content_type(
                                item.get("type") or "text"
                            ),
                            "text": item.get("text"),
                        }
                    )
            return content_list
        else:
            raise ValueError(f"Invalid content type: {type(content)}")

    @staticmethod
    def _get_chat_completion_request_content_type(content_type: str) -> str:
        """
        Get the Chat Completion request content type
        """
        # Responses API content has `input_` prefix, if it exists, remove it
        if content_type.startswith("input_"):
            return content_type[len("input_") :]
        else:
            return content_type

    @staticmethod
    def transform_instructions_to_system_message(
        instructions: Optional[str],
    ) -> ChatCompletionSystemMessage:
        """
        Transform a Instructions into a system message
        """
        return ChatCompletionSystemMessage(role="system", content=instructions or "")

    @staticmethod
    def transform_chat_completion_response_to_responses_api_response(
        chat_completion_response: dict,
    ) -> dict:
        """
        Transform a Chat Completion response into a Responses API response
        """
        return {}
