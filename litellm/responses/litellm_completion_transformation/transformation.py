"""
Handles transforming from Responses API -> LiteLLM completion  (Chat Completion API)
"""

from collections.abc import Sequence
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union, cast

from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses.tool_param import FunctionToolParam
from typing_extensions import TypedDict

from litellm.caching import InMemoryCache
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.responses.litellm_completion_transformation.session_handler import (
    ResponsesSessionHandler,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionImageObject,
    ChatCompletionImageUrlObject,
    ChatCompletionResponseMessage,
    ChatCompletionSystemMessage,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolMessage,
    ChatCompletionToolParam,
    ChatCompletionUserMessage,
    GenericChatCompletionMessage,
    InputTokensDetails,
    OpenAIMcpServerTool,
    OpenAIWebSearchOptions,
    OpenAIWebSearchUserLocation,
    OutputTokensDetails,
    ResponseAPIUsage,
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
    ResponsesAPIStatus,
    ValidChatCompletionMessageContentTypes,
    ValidChatCompletionMessageContentTypesLiteral,
)
from litellm.types.responses.main import (
    GenericResponseOutputItem,
    GenericResponseOutputItemContentAnnotation,
    OutputFunctionToolCall,
    OutputImageGenerationCall,
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
            "reasoning",
            "stream",
            "temperature",
            "text",
            "tool_choice",
            "tools",
            "top_p",
            "user",
        ]

    @staticmethod
    def _transform_tool_choice(
        tool_choice: Any,
    ) -> Optional[Union[str, Dict[str, Any]]]:
        """
        Transform tool_choice from various formats to OpenAI Chat Completion format.

        Handles:
        - String values: "auto", "none", "required" -> pass through as-is
        - Dict with type only (Cursor IDE format):
            - {"type": "auto"} -> "auto"
            - {"type": "none"} -> "none"
            - {"type": "required"} -> "required"
            - {"type": "tool"} -> "required" (force tool use without specific tool)
        - Dict with function (OpenAI format):
            - {"type": "function", "function": {"name": "..."}} -> pass through as-is

        This normalization is needed because some clients (like Cursor IDE) send
        tool_choice in a dict format like {"type": "tool"} which is not valid for
        providers like Anthropic that require a tool name when forcing tool use.
        """
        if tool_choice is None:
            return None

        if isinstance(tool_choice, str):
            return tool_choice

        if isinstance(tool_choice, dict):
            tool_choice_type = tool_choice.get("type")

            # If it has a function with name, it's standard OpenAI format - pass through
            if tool_choice.get("function") and tool_choice.get("function", {}).get(
                "name"
            ):
                return tool_choice

            # Handle Cursor IDE dict formats without function name
            if tool_choice_type == "auto":
                return "auto"
            elif tool_choice_type == "none":
                return "none"
            elif tool_choice_type in ["required", "tool", "any"]:
                # "tool" without a specific function name means "use any tool"
                # which is equivalent to "required" in OpenAI format
                return "required"
            elif tool_choice_type == "function":
                # function type without name - fall back to required
                return "required"

        # Return as-is for unknown formats
        return tool_choice

    @staticmethod
    def transform_responses_api_request_to_chat_completion_request(
        model: str,
        input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        custom_llm_provider: Optional[str] = None,
        stream: Optional[bool] = None,
        extra_headers: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> dict:
        """
        Transform a Responses API request into a Chat Completion request
        """
        (
            tools,
            web_search_options,
        ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            responses_api_request.get("tools") or []  # type: ignore
        )

        response_format = None
        text_param = responses_api_request.get("text")
        if text_param:
            response_format = LiteLLMCompletionResponsesConfig._transform_text_format_to_response_format(
                text_param
            )

        # Extract reasoning_effort from reasoning parameter
        reasoning_effort = None
        reasoning_param = responses_api_request.get("reasoning")
        if reasoning_param:
            if isinstance(reasoning_param, dict):
                # reasoning can be {"effort": "low|medium|high"}
                reasoning_effort = reasoning_param.get("effort")
            elif isinstance(reasoning_param, str):
                # reasoning could be a string directly
                reasoning_effort = reasoning_param

        litellm_completion_request: dict = {
            "messages": LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
                input=input,
                responses_api_request=responses_api_request,
            ),
            "model": model,
            "tool_choice": LiteLLMCompletionResponsesConfig._transform_tool_choice(
                responses_api_request.get("tool_choice")
            ),
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
            "response_format": response_format,
            "reasoning_effort": reasoning_effort,
            # litellm specific params
            "custom_llm_provider": custom_llm_provider,
            "extra_headers": extra_headers,
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
        chat_completion_session = ChatCompletionSession(
            messages=[], litellm_session_id=None
        )
        if previous_response_id:
            chat_completion_session = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
                previous_response_id=previous_response_id
            )
        _messages = litellm_completion_request.get("messages") or []
        session_messages = chat_completion_session.get("messages") or []
        
        # If session messages are empty (e.g., no database in test environment),
        # we still need to process the new input messages
        # Store original _messages before combining for safety check
        original_new_messages = _messages.copy() if _messages else []
        
        combined_messages = session_messages + _messages
        
        # Fix: Ensure tool_results have corresponding tool_calls in previous assistant message
        # Pass tools parameter to help reconstruct tool_calls if not in cache
        tools = litellm_completion_request.get("tools") or []
        combined_messages = LiteLLMCompletionResponsesConfig._ensure_tool_results_have_corresponding_tool_calls(
            messages=combined_messages,
            tools=tools
        )
        
        # Safety check: Ensure we don't end up with empty messages
        # This can happen when using previous_response_id without a database (e.g., in tests)
        # and session messages are empty but new input messages exist
        if not combined_messages:
            # If we end up with empty messages, try to restore from original inputs
            if original_new_messages:
                # If we had new input messages but they got filtered out,
                # restore them (better to have messages than empty list)
                # This can happen when tool_call_id is empty and can't be recovered
                combined_messages = original_new_messages
            elif session_messages:
                # If we had session messages but they got filtered out,
                # restore them
                combined_messages = session_messages
            else:
                # Both are empty - this likely means function_call_output had empty/invalid call_id
                # Provide a helpful error message
                import litellm
                raise litellm.BadRequestError(
                    message=(
                        f"Unable to create messages for completion request. "
                        f"This can happen when: "
                        f"1) Using previous_response_id without a session database, AND "
                        f"2) Input contains only function_call_output with empty or invalid call_id. "
                        f"Please ensure function_call_output has a valid call_id from a previous response. "
                        f"Original request: previous_response_id={previous_response_id}"
                    ),
                    model=litellm_completion_request.get("model", ""),
                    llm_provider=litellm_completion_request.get("custom_llm_provider", ""),
                )
        
        litellm_completion_request["messages"] = combined_messages
        litellm_completion_request["litellm_trace_id"] = chat_completion_session.get(
            "litellm_session_id"
        )
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

        if isinstance(input, str):
            messages.append(ChatCompletionUserMessage(role="user", content=input))
        elif isinstance(input, list):
            existing_tool_call_ids: Set[str] = set()
            for _input in input:
                chat_completion_messages = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
                    input_item=_input
                )

                if LiteLLMCompletionResponsesConfig._is_input_item_function_call(
                    input_item=_input
                ):
                    call_id_raw = _input.get("call_id") or _input.get("id") or ""
                    if call_id_raw:
                        existing_tool_call_ids.add(str(call_id_raw))

                #########################################################
                # If Input Item is a Tool Call Output, add it to the tool_call_output_messages list
                # preserving the ordering of tool call outputs. Some models require the tool 
                # result to immediately follow the assistant tool call. 
                #########################################################
                if LiteLLMCompletionResponsesConfig._is_input_item_tool_call_output(
                    input_item=_input
                ):
                    if not chat_completion_messages:
                        continue

                    deduped_in_place: List[Any] = []
                    for m in chat_completion_messages:
                        role = ""
                        if isinstance(m, dict):
                            role = str(m.get("role") or "")
                        else:
                            role = str(getattr(m, "role", "") or "")

                        # Drop assistant tool_calls wrappers if we already have this call_id
                        if role == "assistant":
                            tool_calls: Any = (
                                m.get("tool_calls")
                                if isinstance(m, dict)
                                else getattr(m, "tool_calls", None)
                            )
                            call_id = ""
                            if (
                                isinstance(tool_calls, Sequence)
                                and not isinstance(tool_calls, (str, bytes))
                                and len(tool_calls) > 0
                            ):
                                first_call = tool_calls[0]
                                call_id_raw = (
                                    first_call.get("id")
                                    if isinstance(first_call, dict)
                                    else getattr(first_call, "id", None)
                                )
                                if call_id_raw:
                                    call_id = str(call_id_raw)
                            if call_id and call_id in existing_tool_call_ids:
                                continue
                            if call_id:
                                existing_tool_call_ids.add(call_id)

                        deduped_in_place.append(m)

                    messages.extend(deduped_in_place)
                    continue

                messages.extend(chat_completion_messages)
        return messages

    @staticmethod
    def _deduplicate_tool_call_output_messages(
        tool_call_output_messages: List[
            Union[
                AllMessageValues,
                GenericChatCompletionMessage,
                ChatCompletionMessageToolCall,
                ChatCompletionResponseMessage,
            ]
        ],
        existing_tool_call_ids: Set[str],
    ) -> List[
        Union[
            AllMessageValues,
            GenericChatCompletionMessage,
            ChatCompletionMessageToolCall,
            ChatCompletionResponseMessage,
        ]
    ]:
        """Return tool call outputs after dropping assistant entries with duplicate call_ids."""
        if not tool_call_output_messages:
            return []

        filtered_messages: List[
            Union[
                AllMessageValues,
                GenericChatCompletionMessage,
                ChatCompletionMessageToolCall,
                ChatCompletionResponseMessage,
            ]
        ] = []
        seen_tool_call_ids: Set[str] = set(existing_tool_call_ids)

        for tool_call_message in tool_call_output_messages:
            if isinstance(tool_call_message, dict):
                role = tool_call_message.get("role", "")
            else:
                role = getattr(tool_call_message, "role", "")
            call_id = ""

            if role == "assistant":
                tool_calls: Any = None
                if isinstance(tool_call_message, dict):
                    tool_calls = tool_call_message.get("tool_calls")
                else:
                    tool_calls = getattr(tool_call_message, "tool_calls", None)

                if (
                    isinstance(tool_calls, Sequence)
                    and not isinstance(tool_calls, (str, bytes))
                    and len(tool_calls) > 0
                ):
                    first_call = tool_calls[0]
                    call_id_raw = None
                    if isinstance(first_call, dict):
                        call_id_raw = first_call.get("id")
                    else:
                        call_id_raw = getattr(first_call, "id", None)

                    if call_id_raw:
                        call_id = str(call_id_raw)

            if call_id and call_id in seen_tool_call_ids and role == "assistant":
                continue

            if call_id and role == "assistant":
                seen_tool_call_ids.add(call_id)

            filtered_messages.append(tool_call_message)

        return filtered_messages

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
    def _find_previous_assistant_idx(
        messages: List[Any], current_idx: int
    ) -> Optional[int]:
        """Find the index of the previous assistant message."""
        for j in range(current_idx - 1, -1, -1):
            if messages[j].get("role") == "assistant":
                return j
        return None

    @staticmethod
    def _recover_tool_call_id_from_assistant(
        assistant_message: Any, message: Any
    ) -> str:
        """Try to recover empty tool_call_id from assistant message's tool_calls."""
        tool_calls_raw = (
            assistant_message.get("tool_calls")
            if isinstance(assistant_message, dict)
            else getattr(assistant_message, "tool_calls", None)
        )
        if tool_calls_raw and isinstance(tool_calls_raw, list) and len(tool_calls_raw) > 0:
            first_tool_call = tool_calls_raw[0]
            if isinstance(first_tool_call, dict):
                tool_call_id_raw = first_tool_call.get("id", "")
                return str(tool_call_id_raw) if tool_call_id_raw is not None else ""
            elif hasattr(first_tool_call, "id"):
                tool_call_id_raw = getattr(first_tool_call, "id", None)
                return str(tool_call_id_raw) if tool_call_id_raw is not None else ""
        return ""

    @staticmethod
    def _get_tool_calls_list(assistant_message: Any) -> List[Any]:
        """Extract tool_calls as a list from assistant message."""
        tool_calls_raw = (
            assistant_message.get("tool_calls")
            if isinstance(assistant_message, dict)
            else getattr(assistant_message, "tool_calls", None)
        )
        if tool_calls_raw is None:
            return []
        if isinstance(tool_calls_raw, list):
            return tool_calls_raw
        if hasattr(tool_calls_raw, "__iter__") and not isinstance(
            tool_calls_raw, (str, bytes)
        ):
            return list(tool_calls_raw)
        return []

    @staticmethod
    def _check_tool_call_exists(tool_calls: List[Any], tool_call_id: str) -> bool:
        """Check if a tool_call with the given ID exists in the list."""
        for tool_call in tool_calls:
            tool_call_id_to_check: Optional[str] = None
            if isinstance(tool_call, dict):
                tool_call_id_to_check = tool_call.get("id")
            elif hasattr(tool_call, "id"):
                tool_call_id_to_check = getattr(tool_call, "id", None)
            if tool_call_id_to_check == tool_call_id:
                return True
        return False

    @staticmethod
    def _reconstruct_tool_call_from_tools(
        tool_call_id: str, tools: List[Any]
    ) -> Optional[Dict[str, Any]]:
        """Reconstruct a minimal tool_call definition from tools list."""
        for tool in tools:
            if isinstance(tool, dict):
                tool_function = tool.get("function") or {}
                tool_name = tool_function.get("name") or tool.get("name") or ""
                if tool_name:
                    return {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": "{}",  # We don't know the arguments, use empty
                        },
                    }
        return None

    @staticmethod
    def _get_mapping_or_attr_value(obj: Any, key: str, default: Any = None) -> Any:
        """
        Safely read a field from dict-like or attribute-based objects.
        """
        if obj is None:
            return default

        if isinstance(obj, dict):
            return obj.get(key, default)

        getter = getattr(obj, "get", None)
        if callable(getter):
            try:
                return getter(key, default)
            except (TypeError, AttributeError):
                pass

        return getattr(obj, key, default)

    @staticmethod
    def _create_tool_call_chunk(
        tool_use_definition: Dict[str, Any], tool_call_id: str, index: int
    ) -> ChatCompletionToolCallChunk:
        """Create a ChatCompletionToolCallChunk from tool_use_definition."""
        function_raw = LiteLLMCompletionResponsesConfig._get_mapping_or_attr_value(
            tool_use_definition, "function"
        )
        function_name_raw = LiteLLMCompletionResponsesConfig._get_mapping_or_attr_value(
            function_raw, "name"
        )
        function_arguments_raw = LiteLLMCompletionResponsesConfig._get_mapping_or_attr_value(
            function_raw, "arguments"
        )
        function: Dict[str, Any] = {
            "name": function_name_raw or "",
            "arguments": function_arguments_raw or "{}",
        }
        tool_use_id_raw = LiteLLMCompletionResponsesConfig._get_mapping_or_attr_value(
            tool_use_definition, "id"
        )
        tool_use_id: str = (
            str(tool_use_id_raw) if tool_use_id_raw is not None else str(tool_call_id)
        )
        tool_use_type_raw = LiteLLMCompletionResponsesConfig._get_mapping_or_attr_value(
            tool_use_definition, "type"
        )
        tool_use_type: str = (
            str(tool_use_type_raw) if tool_use_type_raw is not None else "function"
        )
        return ChatCompletionToolCallChunk(
            id=tool_use_id,
            type=cast(Literal["function"], tool_use_type),
            function=ChatCompletionToolCallFunctionChunk(
                name=str(function.get("name", "")),
                arguments=str(function.get("arguments", "{}")),
            ),
            index=index,
        )

    @staticmethod
    def _normalize_tool_use_definition(
        tool_use_definition: Any, tool_call_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Normalize cached tool_call definitions to a dict-like shape consumed by _create_tool_call_chunk.
        """
        if not tool_use_definition:
            return None

        if isinstance(tool_use_definition, dict):
            normalized_definition: Dict[str, Any] = dict(tool_use_definition)
        else:
            tool_use_id_raw = LiteLLMCompletionResponsesConfig._get_mapping_or_attr_value(
                tool_use_definition, "id"
            )
            tool_use_type_raw = LiteLLMCompletionResponsesConfig._get_mapping_or_attr_value(
                tool_use_definition, "type"
            )
            function_raw = LiteLLMCompletionResponsesConfig._get_mapping_or_attr_value(
                tool_use_definition, "function"
            )

            # Object does not expose the expected tool_call fields.
            if (
                tool_use_id_raw is None
                and tool_use_type_raw is None
                and function_raw is None
            ):
                return None

            normalized_definition = {
                "id": tool_use_id_raw,
                "type": tool_use_type_raw,
                "function": function_raw,
            }

        function_raw = normalized_definition.get("function")
        if function_raw is not None and not isinstance(function_raw, dict):
            function_name_raw = LiteLLMCompletionResponsesConfig._get_mapping_or_attr_value(
                function_raw, "name"
            )
            function_arguments_raw = LiteLLMCompletionResponsesConfig._get_mapping_or_attr_value(
                function_raw, "arguments"
            )
            if function_name_raw is not None or function_arguments_raw is not None:
                normalized_definition["function"] = {
                    "name": function_name_raw,
                    "arguments": function_arguments_raw,
                }

        normalized_definition["id"] = normalized_definition.get("id") or tool_call_id
        normalized_definition["type"] = (
            normalized_definition.get("type") or "function"
        )
        return normalized_definition

    @staticmethod
    def _add_tool_call_to_assistant(
        assistant_message: Any, tool_call_chunk: ChatCompletionToolCallChunk
    ) -> None:
        """Add a tool_call to an assistant message."""
        if isinstance(assistant_message, dict):
            prev_assistant_dict = cast(Dict[str, Any], assistant_message)
            if "tool_calls" not in prev_assistant_dict:
                prev_assistant_dict["tool_calls"] = []
            tool_calls_list = prev_assistant_dict["tool_calls"]
            if isinstance(tool_calls_list, list):
                tool_calls_list.append(tool_call_chunk)
        elif hasattr(assistant_message, "tool_calls"):
            if assistant_message.tool_calls is None:
                assistant_message.tool_calls = []
            if isinstance(assistant_message.tool_calls, list):
                assistant_message.tool_calls.append(tool_call_chunk)

    @staticmethod
    def _ensure_tool_results_have_corresponding_tool_calls(
        messages: List[Union[AllMessageValues, GenericChatCompletionMessage, ChatCompletionResponseMessage]],
        tools: Optional[List[Any]] = None,
    ) -> List[Union[AllMessageValues, GenericChatCompletionMessage, ChatCompletionResponseMessage]]:
        """
        Ensure that tool_result messages have corresponding tool_calls in the previous assistant message.
        
        This is critical for Anthropic API which requires that each tool_result block has a
        corresponding tool_use block in the previous assistant message.
        
        Args:
            messages: List of messages that may include tool_result messages
            tools: Optional list of tools that can be used to reconstruct tool_calls if not in cache
            
        Returns:
            List of messages with tool_calls added to assistant messages when needed
        """
        if not messages:
            return messages
        
        # Create a deep copy to avoid modifying the original
        import copy
        fixed_messages = copy.deepcopy(messages)
        messages_to_remove = []
        
        # Count non-tool messages to avoid removing all messages
        # This prevents empty messages list when using previous_response_id without a database
        non_tool_messages_count = sum(
            1 for msg in fixed_messages if msg.get("role") != "tool"
        )
        
        for i, message in enumerate(fixed_messages):
            # Only process tool messages - check role first to narrow the type
            if message.get("role") != "tool":
                continue
                
            # At this point, we know it's a tool message, so it should have tool_call_id
            # Use get() with default to safely access tool_call_id
            tool_call_id_raw = message.get("tool_call_id") if isinstance(message, dict) else getattr(message, "tool_call_id", None)
            tool_call_id: str = (
                str(tool_call_id_raw) if tool_call_id_raw is not None else ""
            )
            
            prev_assistant_idx = LiteLLMCompletionResponsesConfig._find_previous_assistant_idx(
                fixed_messages, i
            )
            
            # Try to recover empty tool_call_id from previous assistant message
            if not tool_call_id and prev_assistant_idx is not None:
                prev_assistant = fixed_messages[prev_assistant_idx]
                tool_call_id = LiteLLMCompletionResponsesConfig._recover_tool_call_id_from_assistant(
                    prev_assistant, message
                )
                if tool_call_id:
                    # Type-safe way to set tool_call_id on tool message
                    if isinstance(message, dict):
                        # Cast to dict to allow setting tool_call_id
                        message_dict = cast(Dict[str, Any], message)
                        message_dict["tool_call_id"] = tool_call_id
                    elif hasattr(message, "tool_call_id"):
                        setattr(message, "tool_call_id", tool_call_id)
            
            # Only remove messages with empty tool_call_id if we have other non-tool messages
            # This prevents ending up with an empty messages list when using previous_response_id
            # without a database (e.g., in tests where session messages are empty)
            if not tool_call_id:
                # If we have non-tool messages, we can safely remove this tool message
                # But if removing it would leave us with no messages, keep it to avoid empty list
                if non_tool_messages_count > 0:
                    messages_to_remove.append(i)
                # If no non-tool messages, keep the tool message even with empty call_id
                # The API will return a proper error message about the missing tool_use block
                continue
            
            # Check if the previous assistant message has the corresponding tool_call
            # This needs to run for ALL tool messages with a valid tool_call_id,
            # not just those that had an empty tool_call_id initially
            if prev_assistant_idx is not None and tool_call_id:
                prev_assistant = fixed_messages[prev_assistant_idx]
                tool_calls = LiteLLMCompletionResponsesConfig._get_tool_calls_list(
                    prev_assistant
                )
                
                if not LiteLLMCompletionResponsesConfig._check_tool_call_exists(
                    tool_calls, tool_call_id
                ):
                    _tool_use_definition = TOOL_CALLS_CACHE.get_cache(key=tool_call_id)
                    
                    if not _tool_use_definition and tools:
                        _tool_use_definition = (
                            LiteLLMCompletionResponsesConfig._reconstruct_tool_call_from_tools(
                                tool_call_id, tools
                            )
                        )

                    normalized_tool_use_definition = (
                        LiteLLMCompletionResponsesConfig._normalize_tool_use_definition(
                            _tool_use_definition, tool_call_id
                        )
                    )

                    if normalized_tool_use_definition:
                        tool_call_chunk = (
                            LiteLLMCompletionResponsesConfig._create_tool_call_chunk(
                                normalized_tool_use_definition,
                                tool_call_id,
                                len(tool_calls),
                            )
                        )
                        LiteLLMCompletionResponsesConfig._add_tool_call_to_assistant(
                            prev_assistant, tool_call_chunk
                        )
        
        # Remove messages with empty tool_call_id that couldn't be fixed
        for idx in reversed(messages_to_remove):
            fixed_messages.pop(idx)
        
        return fixed_messages

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
        elif LiteLLMCompletionResponsesConfig._is_input_item_function_call(input_item):
            # handle function call input items
            return LiteLLMCompletionResponsesConfig._transform_responses_api_function_call_to_chat_completion_message(
                function_call=input_item
            )
        else:
            content = input_item.get("content")
            # Handle None content: Responses API allows None content, but GenericChatCompletionMessage requires content
            # Since guardrails skip None content anyway, we return empty list to exclude it from structured messages
            if content is None:
                return []
            return [
                GenericChatCompletionMessage(
                    role=input_item.get("role") or "user",
                    content=LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(
                        content
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
            "tool_result",  # Anthropic/MCP format
        ]

    @staticmethod
    def _is_input_item_function_call(input_item: Any) -> bool:
        """
        Check if the input item is a function call
        """
        return input_item.get("type") == "function_call"

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
        call_id = tool_call_output.get("call_id")
        # If call_id is missing or empty, skip this message
        # Empty call_id means we can't create a valid tool message
        if not call_id:
            return []

        def _normalize_function_call_output_to_tool_content(
            output: Any,
        ) -> Any:
            """
            Normalize Responses API function_call_output.output into a shape that downstream
            chat adapters (esp. Gemini) can reliably consume.

            OpenAI Responses API typically uses:
            - output: string

            Some clients/adapters send:
            - output: [{"type": "input_text", "text": "..."}, {"type": "input_image", ...}]

            For chat tool messages we normalize to either:
            - string (preferred)
            - list of {"type": "text"|"image_url", ...} blocks (for multimodal tool outputs)
            """
            if output is None:
                return ""
            if isinstance(output, str):
                return output

            # Some adapters represent tool output as a list of "input_*" parts
            if isinstance(output, list):
                normalized_blocks: List[Dict[str, Any]] = []
                text_acc: List[str] = []
                for part in output:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")
                    if part_type in ("input_text", "output_text", "text"):
                        txt = part.get("text")
                        if isinstance(txt, str) and txt:
                            text_acc.append(txt)
                            normalized_blocks.append({"type": "text", "text": txt})
                    elif part_type in ("input_image", "image_url"):
                        image_url_val = part.get("image_url") or part.get("url")
                        if isinstance(image_url_val, dict):
                            url = image_url_val.get("url")
                            if isinstance(url, str) and url:
                                normalized_blocks.append(
                                    {"type": "image_url", "image_url": {"url": url}}
                                )
                        elif isinstance(image_url_val, str) and image_url_val:
                            normalized_blocks.append(
                                {"type": "image_url", "image_url": {"url": image_url_val}}
                            )

                # Prefer structured blocks if we have images; otherwise return a string.
                if any(b.get("type") == "image_url" for b in normalized_blocks):
                    # Ensure we include any accumulated text as text blocks too
                    return normalized_blocks
                if text_acc:
                    return "".join(text_acc)
                try:
                    # last resort: keep something meaningful for providers that require a string
                    import json as _json

                    return _json.dumps(output)
                except Exception:
                    return str(output)

            # Fallback for dict/number/etc.
            try:
                import json as _json

                return _json.dumps(output)
            except Exception:
                return str(output)

        tool_output_message = ChatCompletionToolMessage(
            role="tool",
            content=_normalize_function_call_output_to_tool_content(
                tool_call_output.get("output")
            ),
            tool_call_id=str(call_id),
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
                type=cast(Literal["function"], _tool_use_definition.get("type") or "function"),
                function=ChatCompletionToolCallFunctionChunk(
                    name=function.get("name") or "",
                    arguments=str(function.get("arguments") or ""),
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
    def _transform_responses_api_function_call_to_chat_completion_message(
        function_call: Dict[str, Any],
    ) -> List[
        Union[
            AllMessageValues,
            GenericChatCompletionMessage,
            ChatCompletionResponseMessage,
        ]
    ]:
        """
        Transform a Responses API function_call into a Chat Completion message with tool calls

        Handles Input items of this type:
        function_call:
        ```json
        {
            "type": "function_call",
            "arguments":"{\"location\": \"São Paulo, Brazil\"}",
            "call_id": "call_v2wlBzrlTIFl9FxPeY774GHZ",
            "name": "get_weather",
            "id": "fc_685c42deefc0819a822b6936faaa30be0c76bc1491ab6619",
            "status": "completed"
        }
        ```
        """
        # Create a tool call for the function call
        tool_call = ChatCompletionToolCallChunk(
            id=function_call.get("call_id") or function_call.get("id") or "",
            type="function",
            function=ChatCompletionToolCallFunctionChunk(
                name=function_call.get("name") or "",
                arguments=str(function_call.get("arguments") or ""),
            ),
            index=0,
        )

        # Create an assistant message with the tool call
        chat_completion_response_message = ChatCompletionResponseMessage(
            tool_calls=[tool_call],
            role="assistant",
            content=None,  # Function calls don't have content
        )

        return [chat_completion_response_message]

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
    def _transform_input_image_item_to_image_item(
        item: Dict[str, Any],
    ) -> ChatCompletionImageObject:
        """
        Transform a Responses API input_image item to a Chat Completion image item
        """
        image_url_obj = ChatCompletionImageUrlObject(
            url=item.get("image_url") or "", detail=item.get("detail") or "auto"
        )

        return ChatCompletionImageObject(type="image_url", image_url=image_url_obj)

    @staticmethod
    def _transform_responses_api_content_to_chat_completion_content(
        content: Any,
    ) -> Union[str, List[Union[str, Dict[str, Any]]]]:
        """
        Transform a Responses API content into a Chat Completion content

        Note: This function should not be called with None content.
        Callers should check for None before calling this function.
        """
        if content is None:
            # Defensive check: should not happen if callers check first
            # Return empty string as fallback to avoid type errors
            return ""
        elif isinstance(content, str):
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
                    elif item.get("type") == "input_image":
                        content_list.append(
                            dict(
                                LiteLLMCompletionResponsesConfig._transform_input_image_item_to_image_item(
                                    item
                                )
                            )
                        )
                    else:
                        # Skip text blocks with None text to avoid downstream errors
                        text_value = item.get("text")
                        if text_value is None:
                            continue
                        content_list.append(
                            {
                                "type": LiteLLMCompletionResponsesConfig._get_chat_completion_request_content_type(
                                    item.get("type") or "text"
                                ),
                                "text": text_value,
                            }
                        )
            return content_list
        else:
            raise ValueError(f"Invalid content type: {type(content)}")

    @staticmethod
    def _get_chat_completion_request_content_type(
        content_type: str,
    ) -> ValidChatCompletionMessageContentTypesLiteral:
        """
        Transform Responses API content type to valid Chat Completion content type.

        Returns one of ValidChatCompletionMessageContentTypes:
        - User: "text", "image_url", "input_audio", "audio_url", "document",
                "guarded_text", "video_url", "file"
        - Assistant: "text", "thinking", "redacted_thinking"
        """
        # Responses API content has `input_` prefix, if it exists, remove it
        if content_type.startswith("input_"):
            stripped = content_type[len("input_") :]
            # Validate stripped type is valid, otherwise default to "text"
            if stripped in ValidChatCompletionMessageContentTypes:
                return stripped  # type: ignore
            # Handle input_audio -> input_audio (it's already valid)
            if stripped == "audio":
                return "input_audio"
            return "text"

        # Map Responses API specific types to valid Chat Completion types
        if content_type in ["tool_result", "output_text"]:
            return "text"

        # Return as-is if it's a valid type, otherwise default to "text"
        if content_type in ValidChatCompletionMessageContentTypes:
            return content_type  # type: ignore

        return "text"

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
    ) -> Tuple[
        List[Union[ChatCompletionToolParam, OpenAIMcpServerTool]],
        Optional[OpenAIWebSearchOptions],
    ]:
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
            elif (
                tool.get("type") == "web_search_preview"
                or tool.get("type") == "web_search"
            ):
                _search_context_size: Literal["low", "medium", "high"] = cast(
                    Literal["low", "medium", "high"], tool.get("search_context_size")
                )
                _user_location: Optional[OpenAIWebSearchUserLocation] = cast(
                    Optional[OpenAIWebSearchUserLocation],
                    tool.get("user_location") or None,
                )
                web_search_options = OpenAIWebSearchOptions(
                    search_context_size=_search_context_size,
                    user_location=_user_location,
                )
            elif tool.get("type") == "function":
                typed_tool = cast(FunctionToolParam, tool)
                # Ensure parameters has "type": "object" as required by providers like Anthropic
                parameters = dict(typed_tool.get("parameters", {}) or {})
                if not parameters or "type" not in parameters:
                    parameters["type"] = "object"
                chat_completion_tool: Dict[str, Any] = {
                    "type": "function",
                    "function": {
                        "name": typed_tool.get("name") or "",
                        "description": typed_tool.get("description") or "",
                        "parameters": parameters,
                        "strict": typed_tool.get("strict", False) or False,
                    }
                }
                if tool.get("cache_control"):
                    chat_completion_tool["cache_control"] = tool.get("cache_control")  # type: ignore
                if tool.get("defer_loading"):
                    chat_completion_tool["defer_loading"] = tool.get("defer_loading")  # type: ignore
                if tool.get("allowed_callers"):
                    chat_completion_tool["allowed_callers"] = tool.get("allowed_callers")  # type: ignore
                if tool.get("input_examples"):
                    chat_completion_tool["input_examples"] = tool.get("input_examples")  # type: ignore
                chat_completion_tools.append(
                    cast(ChatCompletionToolParam, chat_completion_tool)
                )
            else:
                chat_completion_tools.append(cast(Union[ChatCompletionToolParam, OpenAIMcpServerTool], tool))
        return chat_completion_tools, web_search_options

    @staticmethod
    def transform_chat_completion_tools_to_responses_tools(
        chat_completion_response: ModelResponse,
    ) -> List[ResponseFunctionToolCall]:
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

        responses_tools: List[ResponseFunctionToolCall] = []
        for tool in all_chat_completion_tools:
            if tool.type == "function":
                function_definition = tool.function
                provider_specific_fields: Optional[Dict] = None
                if hasattr(tool, "provider_specific_fields") and getattr(
                    tool, "provider_specific_fields", None
                ):
                    provider_specific_fields = getattr(tool, "provider_specific_fields")
                    if not isinstance(provider_specific_fields, dict):
                        provider_specific_fields = (
                            dict(provider_specific_fields)  # type: ignore
                            if hasattr(provider_specific_fields, "__dict__")
                            else {}
                        )
                elif hasattr(
                    function_definition, "provider_specific_fields"
                ) and getattr(function_definition, "provider_specific_fields", None):
                    provider_specific_fields = getattr(
                        function_definition, "provider_specific_fields"
                    )
                    if not isinstance(provider_specific_fields, dict):
                        provider_specific_fields = (
                            dict(provider_specific_fields)  # type: ignore
                            if hasattr(provider_specific_fields, "__dict__")
                            else {}
                        )

                output_tool_call: ResponseFunctionToolCall = ResponseFunctionToolCall(
                    name=function_definition.name or "",
                    arguments=function_definition.get("arguments") or "",
                    call_id=tool.id or "",
                    id=tool.id or "",
                    type="function_call",  # critical this is "function_call" to work with tools like openai codex
                    status=function_definition.get("status") or "completed",
                )

                # Pass through provider_specific_fields as-is if present
                if provider_specific_fields:
                    setattr(output_tool_call, "provider_specific_fields", provider_specific_fields)  # type: ignore

                responses_tools.append(output_tool_call)
        return responses_tools

    @staticmethod
    def _map_chat_completion_finish_reason_to_responses_status(
        finish_reason: Optional[str],
    ) -> ResponsesAPIStatus:
        """
        Map chat completion finish_reason to responses API status.

        Chat completion finish_reason values include: "stop", "length", "tool_calls", "content_filter", "function_call"
        Responses API status values are: "completed", "failed", "in_progress", "cancelled", "queued", "incomplete"

        Args:
            finish_reason: The finish_reason from a chat completion response

        Returns:
            The corresponding responses API status value (one of ResponsesAPIStatus)
        """
        if finish_reason is None:
            return "completed"

        # Map finish reasons to status
        if finish_reason in ["stop", "tool_calls", "function_call"]:
            return "completed"
        elif finish_reason in ["length", "content_filter"]:
            return "incomplete"
        else:
            # Default to completed for unknown finish reasons
            return "completed"

    @staticmethod
    def convert_response_function_tool_call_to_chat_completion_tool_call(
        tool_call_item: Any,
        index: int = 0,
    ) -> Dict[str, Any]:
        """
        Convert ResponseFunctionToolCall to ChatCompletionToolCallChunk format.

        Args:
            tool_call_item: ResponseFunctionToolCall object or similar with name, arguments, call_id
            index: The index of this tool call

        Returns:
            Dictionary in ChatCompletionToolCallChunk format
        """
        # Extract provider_specific_fields if present
        provider_specific_fields = getattr(
            tool_call_item, "provider_specific_fields", None
        )
        if provider_specific_fields and not isinstance(provider_specific_fields, dict):
            provider_specific_fields = (
                dict(provider_specific_fields)
                if hasattr(provider_specific_fields, "__dict__")
                else {}
            )
        elif hasattr(tool_call_item, "get") and callable(tool_call_item.get):  # type: ignore
            provider_fields = tool_call_item.get("provider_specific_fields")  # type: ignore
            if provider_fields:
                provider_specific_fields = (
                    provider_fields
                    if isinstance(provider_fields, dict)
                    else (
                        dict(provider_fields)  # type: ignore
                        if hasattr(provider_fields, "__dict__")
                        else {}
                    )
                )

        function_dict: Dict[str, Any] = {
            "name": tool_call_item.name,
            "arguments": tool_call_item.arguments,
        }

        if provider_specific_fields:
            function_dict["provider_specific_fields"] = provider_specific_fields

        tool_call_dict: Dict[str, Any] = {
            "id": tool_call_item.call_id,
            "function": function_dict,
            "type": "function",
            "index": 0,
        }

        if provider_specific_fields:
            tool_call_dict["provider_specific_fields"] = provider_specific_fields

        return tool_call_dict

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
        # Get finish_reason from the first choice to determine overall status
        finish_reason: Optional[str] = None
        choices: List[Choices] = getattr(chat_completion_response, "choices", [])
        if choices and len(choices) > 0:
            finish_reason = choices[0].finish_reason

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
            reasoning=None,
            status=LiteLLMCompletionResponsesConfig._map_chat_completion_finish_reason_to_responses_status(
                finish_reason
            ),
            text={},
            truncation=getattr(chat_completion_response, "truncation", None),
            usage=LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
                chat_completion_response=chat_completion_response
            ),
            user=getattr(chat_completion_response, "user", None),
        )
        responses_api_response._hidden_params = getattr(chat_completion_response, "_hidden_params", {})

        # Surface provider-specific fields (generic passthrough from any provider)
        provider_fields = responses_api_response._hidden_params.get("provider_specific_fields")
        if provider_fields:
            setattr(responses_api_response, "provider_specific_fields", provider_fields)

        return responses_api_response

    @staticmethod
    def _transform_chat_completion_choices_to_responses_output(
        chat_completion_response: ModelResponse,
        choices: List[Choices],
    ) -> List[
        Union[
            GenericResponseOutputItem,
            OutputFunctionToolCall,
            OutputImageGenerationCall,
            ResponseFunctionToolCall,
        ]
    ]:
        responses_output: List[
            Union[
                GenericResponseOutputItem,
                OutputFunctionToolCall,
                OutputImageGenerationCall,
                ResponseFunctionToolCall,
            ]
        ] = []

        responses_output.extend(
            LiteLLMCompletionResponsesConfig._extract_reasoning_output_items(
                chat_completion_response, choices
            )
        )
        responses_output.extend(
            LiteLLMCompletionResponsesConfig._extract_message_output_items(
                chat_completion_response, choices
            )
        )
        responses_output.extend(
            LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(
                chat_completion_response=chat_completion_response
            )
        )
        return responses_output

    @staticmethod
    def _extract_reasoning_output_items(
        chat_completion_response: ModelResponse,
        choices: List[Choices],
    ) -> List[GenericResponseOutputItem]:
        for choice in choices:
            if hasattr(choice, "message") and choice.message:
                message = choice.message
                if hasattr(message, "reasoning_content") and message.reasoning_content:
                    # Only check the first choice for reasoning content
                    return [
                        GenericResponseOutputItem(
                            type="reasoning",
                            id=f"rs_{hash(str(message.reasoning_content))}",
                            status=LiteLLMCompletionResponsesConfig._map_chat_completion_finish_reason_to_responses_status(
                                choice.finish_reason
                            ),
                            role="assistant",
                            content=[
                                OutputText(
                                    type="output_text",
                                    text=message.reasoning_content,
                                    annotations=[],
                                )
                            ],
                        )
                    ]
        return []

    @staticmethod
    def _extract_image_generation_output_items(
        chat_completion_response: ModelResponse,
        choice: Choices,
    ) -> List[OutputImageGenerationCall]:
        """
        Extract image generation outputs from a choice that contains images.

        Transforms message.images from chat completion format:
        {
            'image_url': {'url': 'data:image/png;base64,iVBORw0...'},
            'type': 'image_url',
            'index': 0
        }

        To Responses API format:
        {
            'type': 'image_generation_call',
            'id': 'img_...',
            'status': 'completed',
            'result': 'iVBORw0...'  # Pure base64 without data: prefix
        }
        """
        image_generation_items: List[OutputImageGenerationCall] = []

        images = getattr(choice.message, "images", [])
        if not images:
            return image_generation_items

        for idx, image_item in enumerate(images):
            # Extract base64 from data URL
            image_url = image_item.get("image_url", {}).get("url", "")
            base64_data = (
                LiteLLMCompletionResponsesConfig._extract_base64_from_data_url(
                    image_url
                )
            )

            if base64_data:
                image_generation_items.append(
                    OutputImageGenerationCall(
                        type="image_generation_call",
                        id=f"{chat_completion_response.id}_img_{idx}",
                        status=LiteLLMCompletionResponsesConfig._map_finish_reason_to_image_generation_status(
                            choice.finish_reason
                        ),
                        result=base64_data,
                    )
                )

        return image_generation_items

    @staticmethod
    def _map_finish_reason_to_image_generation_status(
        finish_reason: Optional[str],
    ) -> Literal["in_progress", "completed", "incomplete", "failed"]:
        """
        Map finish_reason to image generation status.

        Image generation status only supports: in_progress, completed, incomplete, failed
        (does not support: cancelled, queued like general ResponsesAPIStatus)
        """
        if finish_reason == "stop":
            return "completed"
        elif finish_reason == "length":
            return "incomplete"
        elif finish_reason in ["content_filter", "error"]:
            return "failed"
        else:
            # Default to completed for other cases
            return "completed"

    @staticmethod
    def _extract_base64_from_data_url(data_url: str) -> Optional[str]:
        """
        Extract pure base64 string from a data URL.

        Input: 'data:image/png;base64,iVBORw0KGgoAAAANS...'
        Output: 'iVBORw0KGgoAAAANS...'

        If input is already pure base64 (no prefix), return as-is.
        """
        if not data_url:
            return None

        # Check if it's a data URL with prefix
        if data_url.startswith("data:"):
            # Split by comma to separate prefix from base64 data
            parts = data_url.split(",", 1)
            if len(parts) == 2:
                return parts[1]  # Return the base64 part
            return None
        else:
            # Already pure base64
            return data_url

    @staticmethod
    def _extract_message_output_items(
        chat_completion_response: ModelResponse,
        choices: List[Choices],
    ) -> List[Union[GenericResponseOutputItem, OutputImageGenerationCall]]:
        message_output_items: List[
            Union[GenericResponseOutputItem, OutputImageGenerationCall]
        ] = []
        for choice in choices:
            # Check if message has images (image generation)
            if hasattr(choice.message, "images") and choice.message.images:
                # Extract image generation output
                image_generation_items = LiteLLMCompletionResponsesConfig._extract_image_generation_output_items(
                    chat_completion_response=chat_completion_response,
                    choice=choice,
                )
                message_output_items.extend(image_generation_items)
            else:
                # Regular message output
                message_output_items.append(
                    GenericResponseOutputItem(
                        type="message",
                        id=chat_completion_response.id,
                        status=LiteLLMCompletionResponsesConfig._map_chat_completion_finish_reason_to_responses_status(
                            choice.finish_reason
                        ),
                        role=choice.message.role,
                        content=[
                            LiteLLMCompletionResponsesConfig._transform_chat_message_to_response_output_text(
                                choice.message
                            )
                        ],
                    )
                )
        return message_output_items

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
                content = output_item.get("content")
                # Skip if content is None (GenericChatCompletionMessage requires content)
                if content is not None:
                    messages.append(
                        GenericChatCompletionMessage(
                            role=str(output_item.get("role")) or "user",
                            content=LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(
                                content
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
        annotations = getattr(message, "annotations", None)
        transformed_annotations = LiteLLMCompletionResponsesConfig._transform_chat_completion_annotations_to_response_output_annotations(
            annotations=annotations
        )

        return OutputText(
            type="output_text",
            text=message.content,
            annotations=transformed_annotations,
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

        response_usage = ResponseAPIUsage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

        # Preserve cost field if it exists (for streaming usage with cost calculation)
        if hasattr(usage, "cost") and usage.cost is not None:
            setattr(response_usage, "cost", usage.cost)

        # Translate prompt_tokens_details to input_tokens_details
        if (
            hasattr(usage, "prompt_tokens_details")
            and usage.prompt_tokens_details is not None
        ):
            prompt_details = usage.prompt_tokens_details
            input_details_dict: Dict[str, int] = {}

            if (
                hasattr(prompt_details, "cached_tokens")
                and prompt_details.cached_tokens is not None
            ):
                input_details_dict["cached_tokens"] = prompt_details.cached_tokens
            else:
                input_details_dict["cached_tokens"] = 0

            if (
                hasattr(prompt_details, "text_tokens")
                and prompt_details.text_tokens is not None
            ):
                input_details_dict["text_tokens"] = prompt_details.text_tokens

            if (
                hasattr(prompt_details, "audio_tokens")
                and prompt_details.audio_tokens is not None
            ):
                input_details_dict["audio_tokens"] = prompt_details.audio_tokens

            if input_details_dict:
                response_usage.input_tokens_details = InputTokensDetails(
                    **input_details_dict
                )

        # Translate completion_tokens_details to output_tokens_details
        if (
            hasattr(usage, "completion_tokens_details")
            and usage.completion_tokens_details is not None
        ):
            completion_details = usage.completion_tokens_details
            output_details_dict: Dict[str, int] = {}
            if (
                hasattr(completion_details, "reasoning_tokens")
                and completion_details.reasoning_tokens is not None
            ):
                output_details_dict["reasoning_tokens"] = (
                    completion_details.reasoning_tokens
                )
            else:
                output_details_dict["reasoning_tokens"] = 0

            if (
                hasattr(completion_details, "text_tokens")
                and completion_details.text_tokens is not None
            ):
                output_details_dict["text_tokens"] = completion_details.text_tokens

            if (
                hasattr(completion_details, "image_tokens")
                and completion_details.image_tokens is not None
            ):
                output_details_dict["image_tokens"] = completion_details.image_tokens

            if output_details_dict:
                response_usage.output_tokens_details = OutputTokensDetails(
                    **output_details_dict
                )

        return response_usage

    @staticmethod
    def _transform_text_format_to_response_format(
        text_param: Union[Dict[str, Any], Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Transform Responses API text.format parameter to Chat Completion response_format parameter.

        Responses API text parameter structure:
        {
            "format": {
                "type": "json_schema",
                "name": "schema_name",
                "schema": {...},
                "strict": True
            }
        }

        Chat Completion response_format structure:
        {
            "type": "json_schema",
            "json_schema": {
                "name": "schema_name",
                "schema": {...},
                "strict": True
            }
        }
        """
        if not text_param:
            return None

        if isinstance(text_param, dict):
            format_param = text_param.get("format")
            if format_param and isinstance(format_param, dict):
                format_type = format_param.get("type")

                if format_type == "json_schema":
                    return {
                        "type": "json_schema",
                        "json_schema": {
                            "name": format_param.get("name", "response_schema"),
                            "schema": format_param.get("schema", {}),
                            "strict": format_param.get("strict", False),
                        },
                    }
                elif format_type == "json_object":
                    return {"type": "json_object"}
                elif format_type == "text":
                    return None

        return None
