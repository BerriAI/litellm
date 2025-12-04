import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union, cast

from litellm import verbose_logger

from litellm.litellm_core_utils.json_validation_rule import normalize_tool_schema
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionAssistantToolCall,
    ChatCompletionRequest,
    ChatCompletionSystemMessage,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolChoiceValues,
    ChatCompletionToolMessage,
    ChatCompletionToolParam,
    ChatCompletionUserMessage,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import (
    AdapterCompletionStreamWrapper,
    Choices,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)


class GoogleGenAIStreamWrapper(AdapterCompletionStreamWrapper):
    """
    Wrapper for streaming Google GenAI generate_content responses.
    Transforms OpenAI streaming chunks to Google GenAI format.
    """

    sent_first_chunk: bool = False
    # State tracking for accumulating partial tool calls
    accumulated_tool_calls: Dict[str, Dict[str, Any]]

    def __init__(self, completion_stream: Any):
        self.sent_first_chunk = False
        self.accumulated_tool_calls = {}
        self._returned_response = False
        super().__init__(completion_stream)

    def __next__(self):
        try:
            if not hasattr(self.completion_stream, "__iter__"):
                if self._returned_response:
                    raise StopIteration
                self._returned_response = True
                return GoogleGenAIAdapter().translate_completion_to_generate_content(
                    self.completion_stream
                )

            for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    continue

                transformed_chunk = GoogleGenAIAdapter().translate_streaming_completion_to_generate_content(
                    chunk, self
                )
                if transformed_chunk:
                    return transformed_chunk

            raise StopIteration
        except StopIteration:
            raise
        except Exception:
            raise StopIteration

    async def __anext__(self):
        try:
            if not hasattr(self.completion_stream, "__aiter__"):
                if self._returned_response:
                    raise StopAsyncIteration
                self._returned_response = True
                return GoogleGenAIAdapter().translate_completion_to_generate_content(
                    self.completion_stream
                )

            async for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    continue

                transformed_chunk = GoogleGenAIAdapter().translate_streaming_completion_to_generate_content(
                    chunk, self
                )
                if transformed_chunk:
                    return transformed_chunk

            # After the stream is exhausted, check for any remaining accumulated tool calls
            if self.accumulated_tool_calls:
                try:
                    parts = []
                    for (
                        tool_call_index,
                        tool_call_data,
                    ) in self.accumulated_tool_calls.items():
                        try:
                            # For tool calls with no arguments, accumulated_args will be "", which is not valid JSON.
                            # We default to an empty JSON object in this case.
                            parsed_args = json.loads(
                                tool_call_data["arguments"] or "{}"
                            )
                            function_call_part = {
                                "functionCall": {
                                    "name": tool_call_data["name"]
                                    or "undefined_tool_name",
                                    "args": parsed_args,
                                }
                            }
                            parts.append(function_call_part)
                        except json.JSONDecodeError:
                            # This can happen if the stream is abruptly cut off mid-argument string.
                            verbose_logger.warning(
                                f"Could not parse tool call arguments at end of stream for index {tool_call_index}. "
                                f"Name: {tool_call_data['name']}. "
                                f"Partial args: {tool_call_data['arguments']}"
                            )
                            pass
                    if parts:
                        final_chunk = {
                            "candidates": [
                                {
                                    "content": {"parts": parts, "role": "model"},
                                    "finishReason": "STOP",
                                    "index": 0,
                                    "safetyRatings": [],
                                }
                            ]
                        }
                        return final_chunk
                finally:
                    # Ensure the accumulator is always cleared to prevent memory leaks
                    self.accumulated_tool_calls.clear()
            raise StopAsyncIteration
        except StopAsyncIteration:
            raise
        except Exception:
            raise StopAsyncIteration

    def google_genai_sse_wrapper(self) -> Iterator[bytes]:
        """
        Convert Google GenAI streaming chunks to Server-Sent Events format.
        """
        for chunk in self.completion_stream:
            if isinstance(chunk, dict):
                payload = f"data: {json.dumps(chunk)}\n\n"
                yield payload.encode()
            else:
                yield chunk

    async def async_google_genai_sse_wrapper(self) -> AsyncIterator[bytes]:
        """
        Async version of google_genai_sse_wrapper.
        """
        from litellm.types.utils import ModelResponseStream

        async for chunk in self.completion_stream:
            if isinstance(chunk, dict):
                payload = f"data: {json.dumps(chunk)}\n\n"
                yield payload.encode()
            elif isinstance(chunk, ModelResponseStream):
                # Transform OpenAI streaming chunk to Google GenAI format
                transformed_chunk = GoogleGenAIAdapter().translate_streaming_completion_to_generate_content(
                    chunk, self
                )

                if isinstance(transformed_chunk, dict):  # Only return non-empty chunks
                    payload = f"data: {json.dumps(transformed_chunk)}\n\n"
                    yield payload.encode()
                else:
                    # For empty chunks, continue to next iteration
                    continue
            else:
                # For other chunk types, yield them directly
                if hasattr(chunk, "encode"):
                    yield chunk.encode()
                else:
                    yield str(chunk).encode()


class GoogleGenAIAdapter:
    """Adapter for transforming Google GenAI generate_content requests to/from litellm.completion format"""

    def __init__(self) -> None:
        pass

    def translate_generate_content_to_completion(
        self,
        model: str,
        contents: Union[List[Dict[str, Any]], Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
        litellm_params: Optional[GenericLiteLLMParams] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Transform generate_content request to litellm completion format

        Args:
            model: The model name
            contents: Generate content contents (can be list or single dict)
            config: Optional config parameters
            **kwargs: Additional parameters from the original request

        Returns:
            Dict in OpenAI format
        """

        # Extract top-level fields from kwargs
        system_instruction = kwargs.get("systemInstruction") or kwargs.get(
            "system_instruction"
        )
        tools = kwargs.get("tools")
        tool_config = kwargs.get("toolConfig") or kwargs.get("tool_config")

        # Normalize contents to list format
        if isinstance(contents, dict):
            contents_list = [contents]
        else:
            contents_list = contents

        # Transform contents to OpenAI messages format
        messages = self._transform_contents_to_messages(
            contents_list, system_instruction=system_instruction
        )

        # Create base request as dict (which is compatible with ChatCompletionRequest)
        completion_request: ChatCompletionRequest = {
            "model": model,
            "messages": messages,
        }

        #########################################################
        # Supported OpenAI chat completion params
        # - temperature
        # - max_tokens
        # - top_p
        # - frequency_penalty
        # - presence_penalty
        # - stop
        # - tools
        # - tool_choice
        #########################################################

        # Add config parameters if provided
        if config:
            # Map common Google GenAI config parameters to OpenAI equivalents
            if "temperature" in config:
                completion_request["temperature"] = config["temperature"]
            if "maxOutputTokens" in config:
                completion_request["max_tokens"] = config["maxOutputTokens"]
            if "topP" in config:
                completion_request["top_p"] = config["topP"]
            if "topK" in config:
                # OpenAI doesn't have direct topK, but we can pass it as extra
                pass
            if "stopSequences" in config:
                completion_request["stop"] = config["stopSequences"]

        # Handle tools transformation
        if tools:
            # Check if tools are already in OpenAI format or Google GenAI format
            if isinstance(tools, list) and len(tools) > 0:
                # Tools are in Google GenAI format, transform them
                openai_tools = self._transform_google_genai_tools_to_openai(tools)

                if openai_tools:
                    completion_request["tools"] = openai_tools

        # Handle tool_config (tool choice)
        if tool_config:
            tool_choice = self._transform_google_genai_tool_config_to_openai(
                tool_config
            )
            if tool_choice:
                completion_request["tool_choice"] = tool_choice

        #########################################################
        # forward any litellm specific params
        #########################################################
        completion_request_dict = dict(completion_request)
        if litellm_params:
            completion_request_dict = self._add_generic_litellm_params_to_request(
                completion_request_dict=completion_request_dict,
                litellm_params=litellm_params,
            )

        return completion_request_dict

    def _add_generic_litellm_params_to_request(
        self,
        completion_request_dict: Dict[str, Any],
        litellm_params: Optional[GenericLiteLLMParams] = None,
    ) -> dict:
        """Add generic litellm params to request. e.g add api_base, api_key, api_version, etc.

        Args:
            completion_request_dict: Dict[str, Any]
            litellm_params: GenericLiteLLMParams

        Returns:
            Dict[str, Any]
        """
        allowed_fields = GenericLiteLLMParams.model_fields.keys()
        if litellm_params:
            litellm_dict = litellm_params.model_dump(exclude_none=True)
            for key, value in litellm_dict.items():
                if key in allowed_fields:
                    completion_request_dict[key] = value
        return completion_request_dict

    def translate_completion_output_params_streaming(
        self,
        completion_stream: Any,
    ) -> Union[AsyncIterator[bytes], None]:
        """Transform streaming completion output to Google GenAI format"""
        google_genai_wrapper = GoogleGenAIStreamWrapper(
            completion_stream=completion_stream
        )
        # Return the SSE-wrapped version for proper event formatting
        return google_genai_wrapper.async_google_genai_sse_wrapper()

    def _transform_google_genai_tools_to_openai(
        self,
        tools: List[Dict[str, Any]],
    ) -> List[ChatCompletionToolParam]:
        """Transform Google GenAI tools to OpenAI tools format"""
        openai_tools: List[Dict[str, Any]] = []

        for tool in tools:
            if "functionDeclarations" in tool:
                for func_decl in tool["functionDeclarations"]:
                    function_chunk: Dict[str, Any] = {
                        "name": func_decl.get("name", ""),
                    }

                    if "description" in func_decl:
                        function_chunk["description"] = func_decl["description"]
                    if "parametersJsonSchema" in func_decl:
                        function_chunk["parameters"] = func_decl["parametersJsonSchema"]

                    openai_tool = {"type": "function", "function": function_chunk}
                    openai_tools.append(openai_tool)

        # normalize the tool schemas
        normalized_tools = [normalize_tool_schema(tool) for tool in openai_tools]

        return cast(List[ChatCompletionToolParam], normalized_tools)

    def _transform_google_genai_tool_config_to_openai(
        self,
        tool_config: Dict[str, Any],
    ) -> Optional[ChatCompletionToolChoiceValues]:
        """Transform Google GenAI tool_config to OpenAI tool_choice"""
        function_calling_config = tool_config.get("functionCallingConfig", {})
        mode = function_calling_config.get("mode", "AUTO")

        mode_mapping = {"AUTO": "auto", "ANY": "required", "NONE": "none"}

        tool_choice = mode_mapping.get(mode, "auto")
        return cast(ChatCompletionToolChoiceValues, tool_choice)

    def _transform_contents_to_messages(
        self,
        contents: List[Dict[str, Any]],
        system_instruction: Optional[Dict[str, Any]] = None,
    ) -> List[AllMessageValues]:
        """Transform Google GenAI contents to OpenAI messages format"""
        messages: List[AllMessageValues] = []

        # Handle system instruction
        if system_instruction:
            system_parts = system_instruction.get("parts", [])
            if system_parts and "text" in system_parts[0]:
                messages.append(
                    ChatCompletionSystemMessage(
                        role="system", content=system_parts[0]["text"]
                    )
                )

        for content in contents:
            role = content.get("role", "user")
            parts = content.get("parts", [])

            if role == "user":
                # Handle user messages with potential function responses
                combined_text = ""
                tool_messages: List[ChatCompletionToolMessage] = []

                for part in parts:
                    if isinstance(part, dict):
                        if "text" in part:
                            combined_text += part["text"]
                        elif "functionResponse" in part:
                            # Transform function response to tool message
                            func_response = part["functionResponse"]
                            tool_message = ChatCompletionToolMessage(
                                role="tool",
                                tool_call_id=f"call_{func_response.get('name', 'unknown')}",
                                content=json.dumps(func_response.get("response", {})),
                            )
                            tool_messages.append(tool_message)
                    elif isinstance(part, str):
                        combined_text += part

                # Add user message if there's text content
                if combined_text:
                    messages.append(
                        ChatCompletionUserMessage(role="user", content=combined_text)
                    )

                # Add tool messages
                messages.extend(tool_messages)

            elif role == "model":
                # Handle assistant messages with potential function calls
                combined_text = ""
                tool_calls: List[ChatCompletionAssistantToolCall] = []

                for part in parts:
                    if isinstance(part, dict):
                        if "text" in part:
                            combined_text += part["text"]
                        elif "functionCall" in part:
                            # Transform function call to tool call
                            func_call = part["functionCall"]
                            tool_call = ChatCompletionAssistantToolCall(
                                id=f"call_{func_call.get('name', 'unknown')}",
                                type="function",
                                function=ChatCompletionToolCallFunctionChunk(
                                    name=func_call.get("name", ""),
                                    arguments=json.dumps(func_call.get("args", {})),
                                ),
                            )
                            tool_calls.append(tool_call)
                    elif isinstance(part, str):
                        combined_text += part

                # Create assistant message
                if tool_calls:
                    assistant_message = ChatCompletionAssistantMessage(
                        role="assistant",
                        content=combined_text if combined_text else None,
                        tool_calls=tool_calls,
                    )
                else:
                    assistant_message = ChatCompletionAssistantMessage(
                        role="assistant",
                        content=combined_text if combined_text else None,
                    )

                messages.append(assistant_message)

        return messages

    def translate_completion_to_generate_content(
        self,
        response: ModelResponse,
    ) -> Dict[str, Any]:
        """
        Transform litellm completion response to Google GenAI generate_content format

        Args:
            response: ModelResponse from litellm.completion

        Returns:
            Dict in Google GenAI generate_content response format
        """


        # Extract the main response content
        choice = response.choices[0] if response.choices else None
        if not choice:
            raise ValueError("Invalid completion response: no choices found")

        # Handle different choice types (Choices vs StreamingChoices)
        if isinstance(choice, Choices):
            if not choice.message:
                raise ValueError(
                    "Invalid completion response: no message found in choice"
                )
            parts = self._transform_openai_message_to_google_genai_parts(choice.message)
        else:
            # Fallback for generic choice objects
            message_content = getattr(choice, "message", {}).get(
                "content", ""
            ) or getattr(choice, "delta", {}).get("content", "")
            parts = [{"text": message_content}] if message_content else []

        # Create Google GenAI format response
        generate_content_response: Dict[str, Any] = {
            "candidates": [
                {
                    "content": {"parts": parts, "role": "model"},
                    "finishReason": self._map_finish_reason(
                        getattr(choice, "finish_reason", None)
                    ),
                    "index": 0,
                    "safetyRatings": [],
                }
            ],
            "usageMetadata": (
                self._map_usage(getattr(response, "usage", None))
                if hasattr(response, "usage") and getattr(response, "usage", None)
                else {
                    "promptTokenCount": 0,
                    "candidatesTokenCount": 0,
                    "totalTokenCount": 0,
                }
            ),
        }

        # Add text field for convenience (common in Google GenAI responses)
        text_content = ""
        for part in parts:
            if isinstance(part, dict) and "text" in part:
                text_content += part["text"]
        if text_content:
            generate_content_response["text"] = text_content

        return generate_content_response

    def translate_streaming_completion_to_generate_content(
        self,
        response: Union[ModelResponse, ModelResponseStream],
        wrapper: GoogleGenAIStreamWrapper,
    ) -> Optional[Dict[str, Any]]:
        """
        Transform streaming litellm completion chunk to Google GenAI generate_content format

        Args:
            response: Streaming ModelResponse chunk from litellm.completion
            wrapper: GoogleGenAIStreamWrapper instance

        Returns:
            Dict in Google GenAI streaming generate_content response format
        """

        # Extract the main response content from streaming chunk
        choice = response.choices[0] if response.choices else None
        if not choice:
            # Return empty chunk if no choices
            return None

        # Handle streaming choice
        if isinstance(choice, StreamingChoices):
            if choice.delta:
                parts = self._transform_openai_delta_to_google_genai_parts_with_accumulation(
                    choice.delta, wrapper
                )
            else:
                parts = []
            finish_reason = getattr(choice, "finish_reason", None)
        else:
            # Fallback for generic choice objects
            message_content = getattr(choice, "delta", {}).get("content", "")
            parts = [{"text": message_content}] if message_content else []
            finish_reason = getattr(choice, "finish_reason", None)

        # Only create response chunk if we have parts or it's the final chunk
        if not parts and not finish_reason:
            return None

        # Create Google GenAI streaming format response
        streaming_chunk: Dict[str, Any] = {
            "candidates": [
                {
                    "content": {"parts": parts, "role": "model"},
                    "finishReason": (
                        self._map_finish_reason(finish_reason)
                        if finish_reason
                        else None
                    ),
                    "index": 0,
                    "safetyRatings": [],
                }
            ]
        }

        # Add usage metadata only in the final chunk (when finish_reason is present)
        if finish_reason:
            usage_metadata = (
                self._map_usage(getattr(response, "usage", None))
                if hasattr(response, "usage") and getattr(response, "usage", None)
                else {
                    "promptTokenCount": 0,
                    "candidatesTokenCount": 0,
                    "totalTokenCount": 0,
                }
            )
            streaming_chunk["usageMetadata"] = usage_metadata

        # Add text field for convenience (common in Google GenAI responses)
        text_content = ""
        for part in parts:
            if isinstance(part, dict) and "text" in part:
                text_content += part["text"]
        if text_content:
            streaming_chunk["text"] = text_content

        return streaming_chunk

    def _transform_openai_message_to_google_genai_parts(
        self,
        message: Any,
    ) -> List[Dict[str, Any]]:
        """Transform OpenAI message to Google GenAI parts format"""
        parts: List[Dict[str, Any]] = []

        # Add text content if present
        if hasattr(message, "content") and message.content:
            parts.append({"text": message.content})

        # Add tool calls if present
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                if hasattr(tool_call, "function") and tool_call.function:
                    try:
                        args = (
                            json.loads(tool_call.function.arguments)
                            if tool_call.function.arguments
                            else {}
                        )
                    except json.JSONDecodeError:
                        args = {}

                    function_call_part = {
                        "functionCall": {
                            "name": tool_call.function.name or "undefined_tool_name",
                            "args": args,
                        }
                    }
                    parts.append(function_call_part)

        return parts if parts else [{"text": ""}]

    def _transform_openai_delta_to_google_genai_parts_with_accumulation(
        self, delta: Any, wrapper: GoogleGenAIStreamWrapper
    ) -> List[Dict[str, Any]]:
        """Transforms OpenAI delta to Google GenAI parts, accumulating streaming tool calls."""

        # 1. Initialize wrapper state if it doesn't exist
        if not hasattr(wrapper, "accumulated_tool_calls"):
            wrapper.accumulated_tool_calls = {}

        parts: List[Dict[str, Any]] = []

        if hasattr(delta, "content") and delta.content:
            parts.append({"text": delta.content})

        # 2. Ensure tool_calls is iterable
        tool_calls = delta.tool_calls or []

        for tool_call in tool_calls:
            if not hasattr(tool_call, "function"):
                continue

            # 3. Use `index` as the primary key for accumulation
            tool_call_index = getattr(tool_call, "index", None)
            if tool_call_index is None:
                continue  # Index is essential for tracking streaming tool calls

            # Initialize accumulator for this index if it's new
            if tool_call_index not in wrapper.accumulated_tool_calls:
                wrapper.accumulated_tool_calls[tool_call_index] = {
                    "name": "",
                    "arguments": "",
                }

            # Accumulate name and arguments
            function_name = getattr(tool_call.function, "name", None)
            args_chunk = getattr(tool_call.function, "arguments", None)

            # Optimization: Skip chunks that have no new data
            if not function_name and not args_chunk:
                verbose_logger.debug(
                    f"Skipping empty tool call chunk for index: {tool_call_index}"
                )
                continue

            if function_name:
                wrapper.accumulated_tool_calls[tool_call_index]["name"] = function_name

            if args_chunk:
                wrapper.accumulated_tool_calls[tool_call_index][
                    "arguments"
                ] += args_chunk

            # Attempt to parse and emit a complete tool call
            accumulated_data = wrapper.accumulated_tool_calls[tool_call_index]
            accumulated_name = accumulated_data["name"]
            accumulated_args = accumulated_data["arguments"]

            # 5. Attempt to parse arguments even if name hasn't arrived.
            try:
                # Attempt to parse the accumulated arguments string
                parsed_args = json.loads(accumulated_args)

                # If parsing succeeds, but we don't have a name yet, wait.
                # The part will be created by a later chunk that brings the name.
                if accumulated_name:
                    # If successful, create the part and clean up
                    function_call_part = {
                        "functionCall": {"name": accumulated_name, "args": parsed_args}
                    }
                    parts.append(function_call_part)

                    # Remove the completed tool call from the accumulator
                    del wrapper.accumulated_tool_calls[tool_call_index]

            except json.JSONDecodeError:
                # The JSON for arguments is still incomplete.
                # We will continue to accumulate and wait for more chunks.
                pass

        return parts

    def _map_finish_reason(self, finish_reason: Optional[str]) -> str:
        """Map OpenAI finish reasons to Google GenAI finish reasons"""
        if not finish_reason:
            return "STOP"

        mapping = {
            "stop": "STOP",
            "length": "MAX_TOKENS",
            "content_filter": "SAFETY",
            "tool_calls": "STOP",
            "function_call": "STOP",
        }

        return mapping.get(finish_reason, "STOP")

    def _map_usage(self, usage: Any) -> Dict[str, int]:
        """Map OpenAI usage to Google GenAI usage format"""
        return {
            "promptTokenCount": getattr(usage, "prompt_tokens", 0) or 0,
            "candidatesTokenCount": getattr(usage, "completion_tokens", 0) or 0,
            "totalTokenCount": getattr(usage, "total_tokens", 0) or 0,
        }
