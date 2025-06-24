"""
Handles transforming from Responses API -> LiteLLM completion  (Chat Completion API)
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

from openai.types.responses.tool_param import FunctionToolParam
from typing_extensions import TypedDict

from litellm._logging import verbose_logger

try:
    from litellm_enterprise.enterprise_callbacks.session_handler import (
        _ENTERPRISE_ResponsesSessionHandler,
    )
except Exception as e:
    verbose_logger.debug(
        f"[Non-Blocking] Unable to import _ENTERPRISE_ResponsesSessionHandler - LiteLLM Enterprise Feature - {str(e)}"
    )
    _ENTERPRISE_ResponsesSessionHandler = None
from litellm.caching import InMemoryCache
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionResponseMessage,
    ChatCompletionSystemMessage,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolMessage,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    ChatCompletionUserMessage,
    GenericChatCompletionMessage,
    OpenAIMcpServerTool,
    OpenAIWebSearchOptions,
    OpenAIWebSearchUserLocation,
    Reasoning,
    ResponseAPIUsage,
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
    ResponseTextConfig,
)
from litellm.types.responses.main import (
    GenericResponseOutputItem,
    GenericResponseOutputItemContentAnnotation,
    OutputFunctionToolCall,
    OutputText,
)
from litellm.types.utils import (
    ChatCompletionAnnotation,
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
    Usage,
)

########### Initialize Classes used for Responses API  ###########
TOOL_CALLS_CACHE = InMemoryCache()


class ChatCompletionSession(TypedDict, total=False):
    messages: List[
        Union[
            AllMessageValues,
            GenericChatCompletionMessage,
            ChatCompletionMessageToolCall,
            ChatCompletionResponseMessage,
            Message,
        ]
    ]
    litellm_session_id: Optional[str]


########### End of Initialize Classes used for Responses API  ###########


class LiteLLMCompletionResponsesConfig:
    @staticmethod
    def get_supported_openai_params(model: str) -> list:
        """
        LiteLLM Adapter from OpenAI Responses API to Chat Completion API supports a subset of OpenAI Responses API params
        """
        return [
            "input",
            "model",
            "instructions",
            "max_output_tokens",
            "metadata",
            "parallel_tool_calls",
            "previous_response_id",
            "stream",
            "temperature",
            "tool_choice",
            "tools",
            "top_p",
            "user",
        ]

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
        tools, web_search_options = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            responses_api_request.get("tools") or []  # type: ignore
        )
        litellm_completion_request: dict = {
            "messages": LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
                input=input,
                responses_api_request=responses_api_request,
            ),
            "model": model,
            "tool_choice": responses_api_request.get("tool_choice"),
            "tools": tools,
            "top_p": responses_api_request.get("top_p"),
            "user": responses_api_request.get("user"),
            "temperature": responses_api_request.get("temperature"),
            "parallel_tool_calls": responses_api_request.get("parallel_tool_calls"),
            "max_tokens": responses_api_request.get("max_output_tokens"),
            "stream": stream,
            "metadata": kwargs.get("metadata"),
            "service_tier": kwargs.get("service_tier"),
            "web_search_options": web_search_options,
            # litellm specific params
            "custom_llm_provider": custom_llm_provider,
        }

        # Responses API `Completed` events require usage, we pass `stream_options` to litellm.completion to include usage
        if stream is True:
            stream_options = {
                "include_usage": True,
            }
            litellm_completion_request["stream_options"] = stream_options
            litellm_logging_obj: Optional[LiteLLMLoggingObj] = kwargs.get(
                "litellm_logging_obj"
            )
            if litellm_logging_obj:
                litellm_logging_obj.stream_options = stream_options

        # only pass non-None values
        litellm_completion_request = {
            k: v for k, v in litellm_completion_request.items() if v is not None
        }

        return litellm_completion_request

    @staticmethod
    def transform_responses_api_input_to_messages(
        input: Union[str, ResponseInputParam],
        responses_api_request: Union[ResponsesAPIOptionalRequestParams, dict],
    ) -> List[
        Union[
            AllMessageValues,
            GenericChatCompletionMessage,
            ChatCompletionMessageToolCall,
            ChatCompletionResponseMessage,
            Message,
        ]
    ]:
        """
        Transform a Responses API input into a list of messages
        """
        messages: List[
            Union[
                AllMessageValues,
                GenericChatCompletionMessage,
                ChatCompletionMessageToolCall,
                ChatCompletionResponseMessage,
                Message,
            ]
        ] = []
        if responses_api_request.get("instructions"):
            messages.append(
                LiteLLMCompletionResponsesConfig.transform_instructions_to_system_message(
                    responses_api_request.get("instructions")
                )
            )

        messages.extend(
            LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
                input=input,
            )
        )

        return messages

    @staticmethod
    async def async_responses_api_session_handler(
        previous_response_id: str,
        litellm_completion_request: dict,
    ) -> dict:
        """
        Async hook to get the chain of previous input and output pairs and return a list of Chat Completion messages
        """
        if _ENTERPRISE_ResponsesSessionHandler is not None:
            chat_completion_session = ChatCompletionSession(
                messages=[], litellm_session_id=None
            )
            if previous_response_id:
                chat_completion_session = await _ENTERPRISE_ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
                    previous_response_id=previous_response_id
                )
            _messages = litellm_completion_request.get("messages") or []
            session_messages = chat_completion_session.get("messages") or []
            litellm_completion_request["messages"] = session_messages + _messages
            litellm_completion_request[
                "litellm_trace_id"
            ] = chat_completion_session.get("litellm_session_id")
        return litellm_completion_request

    @staticmethod
    def _transform_response_input_param_to_chat_completion_message(
        input: Union[str, ResponseInputParam],
    ) -> List[
        Union[
            AllMessageValues,
            GenericChatCompletionMessage,
            ChatCompletionMessageToolCall,
            ChatCompletionResponseMessage,
        ]
    ]:
        """
        Transform a ResponseInputParam into a Chat Completion message
        """
        messages: List[
            Union[
                AllMessageValues,
                GenericChatCompletionMessage,
                ChatCompletionMessageToolCall,
                ChatCompletionResponseMessage,
            ]
        ] = []
        tool_call_output_messages: List[
            Union[
                AllMessageValues,
                GenericChatCompletionMessage,
                ChatCompletionMessageToolCall,
                ChatCompletionResponseMessage,
            ]
        ] = []

        if isinstance(input, str):
            messages.append(ChatCompletionUserMessage(role="user", content=input))
        elif isinstance(input, list):
            for _input in input:
                chat_completion_messages = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
                    input_item=_input
                )
                if LiteLLMCompletionResponsesConfig._is_input_item_tool_call_output(
                    input_item=_input
                ):
                    tool_call_output_messages.extend(chat_completion_messages)
                else:
                    messages.extend(chat_completion_messages)

        messages.extend(tool_call_output_messages)
        return messages

    @staticmethod
    def _ensure_tool_call_output_has_corresponding_tool_call(
        messages: List[Union[AllMessageValues, GenericChatCompletionMessage]],
    ) -> bool:
        """
        If any tool call output is present, ensure there is a corresponding tool call/tool_use block
        """
        for message in messages:
            if message.get("role") == "tool":
                return True
        return False

    @staticmethod
    def _transform_responses_api_input_item_to_chat_completion_message(
        input_item: Any,
    ) -> List[
        Union[
            AllMessageValues,
            GenericChatCompletionMessage,
            ChatCompletionResponseMessage,
        ]
    ]:
        """
        Transform a Responses API input item into a Chat Completion message

        - EasyInputMessageParam
        - Message
        - ResponseOutputMessageParam
        - ResponseFileSearchToolCallParam
        - ResponseComputerToolCallParam
        - ComputerCallOutput
        - ResponseFunctionWebSearchParam
        - ResponseFunctionToolCallParam
        - FunctionCallOutput
        - ResponseReasoningItemParam
        - ItemReference
        """
        if LiteLLMCompletionResponsesConfig._is_input_item_tool_call_output(input_item):
            # handle executed tool call results
            return LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
                tool_call_output=input_item
            )
        else:
            return [
                GenericChatCompletionMessage(
                    role=input_item.get("role") or "user",
                    content=LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(
                        input_item.get("content")
                    ),
                )
            ]

    @staticmethod
    def _is_input_item_tool_call_output(input_item: Any) -> bool:
        """
        Check if the input item is a tool call output
        """
        return input_item.get("type") in [
            "function_call_output",
            "web_search_call",
            "computer_call_output",
        ]

    @staticmethod
    def _transform_responses_api_tool_call_output_to_chat_completion_message(
        tool_call_output: Dict[str, Any],
    ) -> List[
        Union[
            AllMessageValues,
            GenericChatCompletionMessage,
            ChatCompletionResponseMessage,
        ]
    ]:
        """
        ChatCompletionToolMessage is used to indicate the output from a tool call
        """
        tool_output_message = ChatCompletionToolMessage(
            role="tool",
            content=tool_call_output.get("output") or "",
            tool_call_id=tool_call_output.get("call_id") or "",
        )

        _tool_use_definition = TOOL_CALLS_CACHE.get_cache(
            key=tool_call_output.get("call_id") or "",
        )
        if _tool_use_definition:
            """
            Append the tool use definition to the list of messages


            Providers like Anthropic require the tool use definition to be included with the tool output

            - Input:
                {'function':
                    arguments:'{"command": ["echo","<html>\\n<head>\\n  <title>Hello</title>\\n</head>\\n<body>\\n  <h1>Hi</h1>\\n</body>\\n</html>",">","index.html"]}',
                    name='shell',
                    'id': 'toolu_018KFWsEySHjdKZPdUzXpymJ',
                    'type': 'function'
                }
            - Output:
                {
                    "id": "toolu_018KFWsEySHjdKZPdUzXpymJ",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"latitude\":48.8566,\"longitude\":2.3522}"
                        }
                }

            """
            function: dict = _tool_use_definition.get("function") or {}
            tool_call_chunk = ChatCompletionToolCallChunk(
                id=_tool_use_definition.get("id") or "",
                type=_tool_use_definition.get("type") or "function",
                function=ChatCompletionToolCallFunctionChunk(
                    name=function.get("name") or "",
                    arguments=function.get("arguments") or "",
                ),
                index=0,
            )
            chat_completion_response_message = ChatCompletionResponseMessage(
                tool_calls=[tool_call_chunk],
                role="assistant",
            )
            return [chat_completion_response_message, tool_output_message]

        return [tool_output_message]

    @staticmethod
    def _transform_input_file_item_to_file_item(item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a Responses API input_file item to a Chat Completion file item

        Args:
            item: Dictionary containing input_file type with file_id and/or file_data

        Returns:
            Dictionary with transformed file structure for Chat Completion
        """
        file_dict: Dict[str, Any] = {}
        keys = ["file_id", "file_data"]
        for key in keys:
            if item.get(key):
                file_dict[key] = item.get(key)

        new_item: Dict[str, Any] = {"type": "file", "file": file_dict}
        return new_item

    @staticmethod
    def _transform_responses_api_content_to_chat_completion_content(
        content: Any,
    ) -> Union[str, List[Union[str, Dict[str, Any]]]]:
        """
        Transform a Responses API content into a Chat Completion content
        """

        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            content_list: List[Union[str, Dict[str, Any]]] = []
            for item in content:
                if isinstance(item, str):
                    content_list.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "input_file":
                        content_list.append(
                            LiteLLMCompletionResponsesConfig._transform_input_file_item_to_file_item(
                                item
                            )
                        )
                    else:
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
    def transform_responses_api_tools_to_chat_completion_tools(
        tools: Optional[List[Union[FunctionToolParam, OpenAIMcpServerTool]]],
    ) -> Tuple[List[Union[ChatCompletionToolParam, OpenAIMcpServerTool]], Optional[OpenAIWebSearchOptions]]:
        """
        Transform a Responses API tools into a Chat Completion tools
        """
        if tools is None:
            return [], None
        chat_completion_tools: List[
            Union[ChatCompletionToolParam, OpenAIMcpServerTool]
        ] = []
        web_search_options: Optional[OpenAIWebSearchOptions] = None
        for tool in tools:
            if tool.get("type") == "mcp":
                chat_completion_tools.append(cast(OpenAIMcpServerTool, tool))
            elif tool.get("type") == "web_search_preview" or tool.get("type") == "web_search":
                _search_context_size: Literal["low", "medium", "high"] = cast(Literal["low", "medium", "high"], tool.get("search_context_size"))
                _user_location: Optional[OpenAIWebSearchUserLocation] = cast(Optional[OpenAIWebSearchUserLocation], tool.get("user_location") or None)
                web_search_options = OpenAIWebSearchOptions(
                    search_context_size=_search_context_size,
                    user_location=_user_location,
                )
            else:
                typed_tool = cast(FunctionToolParam, tool)
                chat_completion_tools.append(
                    ChatCompletionToolParam(
                        type="function",
                        function=ChatCompletionToolParamFunctionChunk(
                            name=typed_tool.get("name") or "",
                            description=typed_tool.get("description") or "",
                            parameters=dict(typed_tool.get("parameters", {}) or {}),
                            strict=typed_tool.get("strict", False) or False,
                        ),
                    )
                )
        return chat_completion_tools, web_search_options

    @staticmethod
    def transform_chat_completion_tools_to_responses_tools(
        chat_completion_response: ModelResponse,
    ) -> List[OutputFunctionToolCall]:
        """
        Transform a Chat Completion tools into a Responses API tools
        """
        all_chat_completion_tools: List[ChatCompletionMessageToolCall] = []
        for choice in chat_completion_response.choices:
            if isinstance(choice, Choices):
                if choice.message.tool_calls:
                    all_chat_completion_tools.extend(choice.message.tool_calls)
                    for tool_call in choice.message.tool_calls:
                        TOOL_CALLS_CACHE.set_cache(
                            key=tool_call.id,
                            value=tool_call,
                        )

        responses_tools: List[OutputFunctionToolCall] = []
        for tool in all_chat_completion_tools:
            if tool.type == "function":
                function_definition = tool.function
                responses_tools.append(
                    OutputFunctionToolCall(
                        name=function_definition.name or "",
                        arguments=function_definition.get("arguments") or "",
                        call_id=tool.id or "",
                        id=tool.id or "",
                        type="function_call",  # critical this is "function_call" to work with tools like openai codex
                        status=function_definition.get("status") or "completed",
                    )
                )
        return responses_tools

    @staticmethod
    def transform_chat_completion_response_to_responses_api_response(
        request_input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        chat_completion_response: Union[ModelResponse, dict],
    ) -> ResponsesAPIResponse:
        """
        Transform a Chat Completion response into a Responses API response
        """
        if isinstance(chat_completion_response, dict):
            chat_completion_response = ModelResponse(**chat_completion_response)
        responses_api_response: ResponsesAPIResponse = ResponsesAPIResponse(
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
        return responses_api_response

    @staticmethod
    def _transform_chat_completion_choices_to_responses_output(
        chat_completion_response: ModelResponse,
        choices: List[Choices],
    ) -> List[Union[GenericResponseOutputItem, OutputFunctionToolCall]]:
        responses_output: List[
            Union[GenericResponseOutputItem, OutputFunctionToolCall]
        ] = []
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

        tool_calls = LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(
            chat_completion_response=chat_completion_response
        )
        responses_output.extend(tool_calls)
        return responses_output

    @staticmethod
    def _transform_responses_api_outputs_to_chat_completion_messages(
        responses_api_output: ResponsesAPIResponse,
    ) -> List[
        Union[
            AllMessageValues,
            GenericChatCompletionMessage,
            ChatCompletionMessageToolCall,
        ]
    ]:
        messages: List[
            Union[
                AllMessageValues,
                GenericChatCompletionMessage,
                ChatCompletionMessageToolCall,
            ]
        ] = []
        output_items = responses_api_output.output
        for _output_item in output_items:
            output_item: dict = dict(_output_item)
            if output_item.get("type") == "function_call":
                # handle function call output
                messages.append(
                    LiteLLMCompletionResponsesConfig._transform_responses_output_tool_call_to_chat_completion_output_tool_call(
                        tool_call=output_item
                    )
                )
            else:
                # transform as generic ResponseOutputItem
                messages.append(
                    GenericChatCompletionMessage(
                        role=str(output_item.get("role")) or "user",
                        content=LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(
                            output_item.get("content")
                        ),
                    )
                )
        return messages

    @staticmethod
    def _transform_responses_output_tool_call_to_chat_completion_output_tool_call(
        tool_call: dict,
    ) -> ChatCompletionMessageToolCall:
        return ChatCompletionMessageToolCall(
            id=tool_call.get("id") or "",
            type="function",
            function=Function(
                name=tool_call.get("name") or "",
                arguments=tool_call.get("arguments") or "",
            ),
        )

    @staticmethod
    def _transform_chat_message_to_response_output_text(
        message: Message,
    ) -> OutputText:
        return OutputText(
            type="output_text",
            text=message.content,
            annotations=LiteLLMCompletionResponsesConfig._transform_chat_completion_annotations_to_response_output_annotations(
                annotations=getattr(message, "annotations", None)
            ),
        )

    @staticmethod
    def _transform_chat_completion_annotations_to_response_output_annotations(
        annotations: Optional[List[ChatCompletionAnnotation]],
    ) -> List[GenericResponseOutputItemContentAnnotation]:
        response_output_annotations: List[
            GenericResponseOutputItemContentAnnotation
        ] = []

        if annotations is None:
            return response_output_annotations

        for annotation in annotations:
            annotation_type = annotation.get("type")
            if annotation_type == "url_citation" and "url_citation" in annotation:
                url_citation = annotation["url_citation"]
                response_output_annotations.append(
                    GenericResponseOutputItemContentAnnotation(
                        type=annotation_type,
                        start_index=url_citation.get("start_index"),
                        end_index=url_citation.get("end_index"),
                        url=url_citation.get("url"),
                        title=url_citation.get("title"),
                    )
                )
            # Handle other annotation types here

        return response_output_annotations

    @staticmethod
    def _transform_chat_completion_usage_to_responses_usage(
        chat_completion_response: Union[ModelResponse, Usage],
    ) -> ResponseAPIUsage:
        if isinstance(chat_completion_response, ModelResponse):
            usage: Optional[Usage] = getattr(chat_completion_response, "usage", None)
        else:
            usage = chat_completion_response
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
