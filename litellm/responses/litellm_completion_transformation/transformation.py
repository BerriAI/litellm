"""
Handles transforming from Responses API -> LiteLLM completion  (Chat Completion API)
"""

from typing import Any, Dict, List, Optional, Union

from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionSystemMessage,
    ChatCompletionUserMessage,
    GenericChatCompletionMessage,
    Reasoning,
    ResponseAPIUsage,
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
    ResponseTextConfig,
)
from litellm.types.responses.main import GenericResponseOutputItem, OutputText
from litellm.types.utils import (
    Choices,
    Message,
    ModelResponse,
    ModelResponseStream,
    Usage,
)


class LiteLLMCompletionResponsesConfig:

    @staticmethod
    def transform_responses_api_request_to_chat_completion_request(
        model: str,
        input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        custom_llm_provider: Optional[str] = None,
        stream: Optional[bool] = None,
        **kwargs,
    ) -> dict:
        """
        Transform a Responses API request into a Chat Completion request
        """
        litellm_completion_request: dict = {
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
            "stream": stream,
            "metadata": kwargs.get("metadata"),
            "service_tier": kwargs.get("service_tier"),
        }

        # only pass non-None values
        litellm_completion_request = {
            k: v for k, v in litellm_completion_request.items() if v is not None
        }

        return litellm_completion_request

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
        chat_completion_response: ModelResponse,
    ) -> ResponsesAPIResponse:
        """
        Transform a Chat Completion response into a Responses API response
        """
        return ResponsesAPIResponse(
            id=chat_completion_response.id,
            created_at=chat_completion_response.created,
            model=chat_completion_response.model,
            object=chat_completion_response.object,
            error=getattr(chat_completion_response, "error", None),
            incomplete_details=getattr(
                chat_completion_response, "incomplete_details", None
            ),
            instructions=getattr(chat_completion_response, "instructions", None),
            metadata=getattr(chat_completion_response, "metadata", {}),
            output=LiteLLMCompletionResponsesConfig._transform_chat_completion_choices_to_responses_output(
                chat_completion_response=chat_completion_response,
                choices=getattr(chat_completion_response, "choices", []),
            ),
            parallel_tool_calls=getattr(
                chat_completion_response, "parallel_tool_calls", False
            ),
            temperature=getattr(chat_completion_response, "temperature", 0),
            tool_choice=getattr(chat_completion_response, "tool_choice", "auto"),
            tools=getattr(chat_completion_response, "tools", []),
            top_p=getattr(chat_completion_response, "top_p", None),
            max_output_tokens=getattr(
                chat_completion_response, "max_output_tokens", None
            ),
            previous_response_id=getattr(
                chat_completion_response, "previous_response_id", None
            ),
            reasoning=Reasoning(),
            status=getattr(chat_completion_response, "status", "completed"),
            text=ResponseTextConfig(),
            truncation=getattr(chat_completion_response, "truncation", None),
            usage=LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
                chat_completion_response=chat_completion_response
            ),
            user=getattr(chat_completion_response, "user", None),
        )

    @staticmethod
    def _transform_chat_completion_choices_to_responses_output(
        chat_completion_response: ModelResponse,
        choices: List[Choices],
    ) -> List[GenericResponseOutputItem]:
        responses_output: List[GenericResponseOutputItem] = []
        for choice in choices:
            responses_output.append(
                GenericResponseOutputItem(
                    type="message",
                    id=chat_completion_response.id,
                    status=choice.finish_reason,
                    role=choice.message.role,
                    content=[
                        LiteLLMCompletionResponsesConfig._transform_chat_message_to_response_output_text(
                            choice.message
                        )
                    ],
                )
            )
        return responses_output

    @staticmethod
    def _transform_chat_message_to_response_output_text(
        message: Message,
    ) -> OutputText:
        return OutputText(
            type="output_text",
            text=message.content,
        )

    @staticmethod
    def _transform_chat_completion_usage_to_responses_usage(
        chat_completion_response: ModelResponse,
    ) -> ResponseAPIUsage:
        usage: Optional[Usage] = getattr(chat_completion_response, "usage", None)
        if usage is None:
            return ResponseAPIUsage(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            )
        return ResponseAPIUsage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )
