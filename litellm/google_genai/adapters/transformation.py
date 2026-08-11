import json
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from litellm import verbose_logger
from litellm.litellm_core_utils.json_validation_rule import normalize_json_schema_types
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionAssistantToolCall,
    ChatCompletionImageObject,
    ChatCompletionImageUrlObject,
    ChatCompletionRequest,
    ChatCompletionSystemMessage,
    ChatCompletionTextObject,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolChoiceStringValues,
    ChatCompletionToolChoiceValues,
    ChatCompletionToolMessage,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    ChatCompletionUserMessage,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import (
    AdapterCompletionStreamWrapper,
    ChatCompletionDeltaToolCall,
    ChatCompletionMessageToolCall,
    Delta,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)


class _InlineData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mime_type: str = "image/jpeg"
    data: str = ""


class _FunctionCall(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = "unknown"
    args: JsonValue = Field(default_factory=dict)


class _FunctionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = "unknown"
    response: JsonValue = Field(default_factory=dict)


class _Part(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    text: str | None = None
    inline_data: _InlineData | None = Field(default=None, alias="inlineData")
    function_call: _FunctionCall | None = Field(default=None, alias="functionCall")
    function_response: _FunctionResponse | None = Field(default=None, alias="functionResponse")


class _Content(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str = "user"
    parts: tuple[_Part | str, ...] = ()


class _FunctionDeclaration(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    name: str = ""
    description: str | None = None
    parameters_json_schema: JsonValue = Field(default=None, alias="parametersJsonSchema")


class _Tool(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    function_declarations: tuple[_FunctionDeclaration, ...] = Field(default=(), alias="functionDeclarations")


class _FunctionCallingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: str = "AUTO"


class _ToolConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    function_calling_config: _FunctionCallingConfig = Field(
        default_factory=_FunctionCallingConfig, alias="functionCallingConfig"
    )


class _TokenCounts(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class _UsageCarrier(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    usage: _TokenCounts | None = None


class _GenerateContentConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    temperature: float | None = None
    max_output_tokens: int | None = Field(default=None, alias="maxOutputTokens")
    top_p: float | None = Field(default=None, alias="topP")
    stop_sequences: list[str] | None = Field(default=None, alias="stopSequences")


_CONTENT_ADAPTER: Final = TypeAdapter(_Content)
_TOOLS_ADAPTER: Final = TypeAdapter(tuple[_Tool, ...])
_TOOL_CONFIG_ADAPTER: Final = TypeAdapter(_ToolConfig)
_CONFIG_ADAPTER: Final = TypeAdapter(_GenerateContentConfig)
_JSON_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_USAGE_ADAPTER: Final = TypeAdapter(_UsageCarrier)
_CompletionStream = Iterable[object] | AsyncIterable[object] | ModelResponse
_TOOL_CHOICE_BY_MODE: Final[Mapping[str, ChatCompletionToolChoiceStringValues]] = MappingProxyType(
    {"AUTO": "auto", "ANY": "required", "NONE": "none"}
)
_EMPTY_USAGE: Final[Mapping[str, int]] = MappingProxyType(
    {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0}
)


def _litellm_param_values(litellm_params: GenericLiteLLMParams) -> Mapping[str, object]:
    return litellm_params.model_dump(exclude_none=True)


def _text_of(parts: Sequence[Mapping[str, object]]) -> str:
    texts: Final = (part.get("text") for part in parts)
    return "".join(text for text in texts if isinstance(text, str))


def _usage_metadata(response: ModelResponse | ModelResponseStream) -> dict[str, int]:
    usage: Final = _USAGE_ADAPTER.validate_python(response).usage
    if usage is None:
        return dict(_EMPTY_USAGE)
    return {
        "promptTokenCount": usage.prompt_tokens,
        "candidatesTokenCount": usage.completion_tokens,
        "totalTokenCount": usage.total_tokens,
    }


class GoogleGenAIStreamWrapper(AdapterCompletionStreamWrapper):
    """
    Wrapper for streaming Google GenAI generate_content responses.
    Transforms OpenAI streaming chunks to Google GenAI format.
    """

    completion_stream: _CompletionStream
    sent_first_chunk: bool = False
    accumulated_tool_calls: dict[int, dict[str, str]]

    def __init__(self, completion_stream: _CompletionStream) -> None:
        self.sent_first_chunk = False
        self.accumulated_tool_calls = {}
        self._returned_response = False
        super().__init__(completion_stream)

    def __next__(self) -> dict[str, object]:
        try:
            stream: Final = self.completion_stream
            if not isinstance(stream, Iterable):
                if self._returned_response or not isinstance(stream, ModelResponse):
                    raise StopIteration
                self._returned_response = True
                return GoogleGenAIAdapter().translate_completion_to_generate_content(stream)

            for chunk in stream:
                if chunk == "None" or chunk is None:
                    continue
                if not isinstance(chunk, (ModelResponse, ModelResponseStream)):
                    continue

                transformed_chunk = GoogleGenAIAdapter().translate_streaming_completion_to_generate_content(chunk, self)
                if transformed_chunk:
                    return transformed_chunk

            raise StopIteration
        except StopIteration:
            raise
        except Exception:
            raise StopIteration

    async def __anext__(self) -> dict[str, object]:
        try:
            stream: Final = self.completion_stream
            if not isinstance(stream, AsyncIterable):
                if self._returned_response or not isinstance(stream, ModelResponse):
                    raise StopAsyncIteration
                self._returned_response = True
                return GoogleGenAIAdapter().translate_completion_to_generate_content(stream)

            async for chunk in stream:
                if chunk == "None" or chunk is None:
                    continue
                if not isinstance(chunk, (ModelResponse, ModelResponseStream)):
                    continue

                transformed_chunk = GoogleGenAIAdapter().translate_streaming_completion_to_generate_content(chunk, self)
                if transformed_chunk:
                    return transformed_chunk

            if self.accumulated_tool_calls:
                try:
                    parts: Final[list[dict[str, object]]] = []
                    for (
                        tool_call_index,
                        tool_call_data,
                    ) in self.accumulated_tool_calls.items():
                        try:
                            parsed_args = _JSON_ADAPTER.validate_json(tool_call_data["arguments"] or "{}")
                            function_call_part: dict[str, object] = {
                                "functionCall": {
                                    "name": tool_call_data["name"] or "undefined_tool_name",
                                    "args": parsed_args,
                                }
                            }
                            parts.append(function_call_part)
                        except ValidationError:
                            verbose_logger.warning(
                                "Could not parse tool call arguments at end of stream for index %s. Name: %s. Partial args: %s",
                                tool_call_index,
                                tool_call_data["name"],
                                tool_call_data["arguments"],
                            )
                    if parts:
                        final_chunk: Final[dict[str, object]] = {
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
        stream: Final = self.completion_stream
        if not isinstance(stream, Iterable):
            return

        for chunk in stream:
            if isinstance(chunk, bytes):
                yield chunk
            elif isinstance(chunk, str):
                yield chunk.encode()
            else:
                yield f"data: {json.dumps(chunk)}\n\n".encode()

    async def async_google_genai_sse_wrapper(self) -> AsyncIterator[bytes]:
        """
        Async version of google_genai_sse_wrapper.
        """
        stream: Final = self.completion_stream
        if not isinstance(stream, AsyncIterable):
            return

        async for chunk in stream:
            if isinstance(chunk, Mapping):
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            elif isinstance(chunk, ModelResponseStream):
                transformed_chunk = GoogleGenAIAdapter().translate_streaming_completion_to_generate_content(chunk, self)

                if transformed_chunk is not None:
                    yield f"data: {json.dumps(transformed_chunk)}\n\n".encode()
            elif isinstance(chunk, str):
                yield chunk.encode()
            elif isinstance(chunk, bytes):
                yield chunk
            else:
                yield str(chunk).encode()


class GoogleGenAIAdapter:
    """Adapter for transforming Google GenAI generate_content requests to/from litellm.completion format"""

    def __init__(self) -> None:
        pass

    def translate_generate_content_to_completion(
        self,
        model: str,
        contents: Sequence[Mapping[str, object]] | Mapping[str, object],
        config: Mapping[str, object] | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
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

        system_instruction: Final = kwargs.get("systemInstruction") or kwargs.get("system_instruction")
        tools: Final = kwargs.get("tools")
        tool_config: Final = kwargs.get("toolConfig") or kwargs.get("tool_config")

        contents_list: Final = [contents] if isinstance(contents, Mapping) else contents

        messages: Final = self._transform_contents_to_messages(contents_list, system_instruction=system_instruction)

        completion_request: Final[ChatCompletionRequest] = {
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

        if config:
            generate_content_config: Final = _CONFIG_ADAPTER.validate_python(config)
            if generate_content_config.temperature is not None:
                completion_request["temperature"] = generate_content_config.temperature
            if generate_content_config.max_output_tokens is not None:
                completion_request["max_tokens"] = generate_content_config.max_output_tokens
            if generate_content_config.top_p is not None:
                completion_request["top_p"] = generate_content_config.top_p
            if generate_content_config.stop_sequences is not None:
                completion_request["stop"] = generate_content_config.stop_sequences

        if tools:
            openai_tools: Final = self._transform_google_genai_tools_to_openai(tools)

            if openai_tools:
                completion_request["tools"] = openai_tools

        if tool_config:
            tool_choice: Final = self._transform_google_genai_tool_config_to_openai(tool_config)
            if tool_choice:
                completion_request["tool_choice"] = tool_choice

        #########################################################
        # forward any litellm specific params
        #########################################################
        completion_request_dict: Final[dict[str, object]] = dict(completion_request)
        if litellm_params is None:
            return completion_request_dict

        return self._add_generic_litellm_params_to_request(
            completion_request_dict=completion_request_dict,
            litellm_params=litellm_params,
        )

    def _add_generic_litellm_params_to_request(
        self,
        completion_request_dict: dict[str, object],
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict[str, object]:
        """Add generic litellm params to request. e.g add api_base, api_key, api_version, etc.

        Args:
            completion_request_dict: dict[str, object]
            litellm_params: GenericLiteLLMParams

        Returns:
            dict[str, object]
        """
        allowed_fields: Final = GenericLiteLLMParams.model_fields.keys()
        if litellm_params:
            for key, value in _litellm_param_values(litellm_params).items():
                if key in allowed_fields:
                    completion_request_dict[key] = value
        return completion_request_dict

    def translate_completion_output_params_streaming(
        self,
        completion_stream: _CompletionStream,
    ) -> AsyncIterator[bytes] | None:
        """Transform streaming completion output to Google GenAI format"""
        google_genai_wrapper: Final = GoogleGenAIStreamWrapper(completion_stream=completion_stream)
        # Return the SSE-wrapped version for proper event formatting
        return google_genai_wrapper.async_google_genai_sse_wrapper()

    def _transform_google_genai_tools_to_openai(
        self,
        tools: object,
    ) -> list[ChatCompletionToolParam]:
        """Transform Google GenAI tools to OpenAI tools format"""
        openai_tools: Final[list[ChatCompletionToolParam]] = []

        for tool in _TOOLS_ADAPTER.validate_python(tools):
            for func_decl in tool.function_declarations:
                function_chunk: ChatCompletionToolParamFunctionChunk = {"name": func_decl.name}

                if func_decl.description is not None:
                    function_chunk["description"] = func_decl.description

                normalized_schema = normalize_json_schema_types(func_decl.parameters_json_schema)
                if isinstance(normalized_schema, dict):
                    function_chunk["parameters"] = normalized_schema

                openai_tools.append(ChatCompletionToolParam(type="function", function=function_chunk))

        return openai_tools

    def _transform_google_genai_tool_config_to_openai(
        self,
        tool_config: object,
    ) -> ChatCompletionToolChoiceValues | None:
        """Transform Google GenAI tool_config to OpenAI tool_choice"""
        mode: Final = _TOOL_CONFIG_ADAPTER.validate_python(tool_config).function_calling_config.mode
        return _TOOL_CHOICE_BY_MODE.get(mode, "auto")

    def _transform_contents_to_messages(
        self,
        contents: Sequence[Mapping[str, object]],
        system_instruction: object = None,
    ) -> list[AllMessageValues]:
        """Transform Google GenAI contents to OpenAI messages format"""
        messages: Final[list[AllMessageValues]] = []

        if system_instruction is not None:
            system_parts: Final = _CONTENT_ADAPTER.validate_python(system_instruction).parts
            first_system_part: Final = system_parts[0] if system_parts else None
            if isinstance(first_system_part, _Part) and first_system_part.text is not None:
                messages.append(ChatCompletionSystemMessage(role="system", content=first_system_part.text))

        for raw_content in contents:
            content = _CONTENT_ADAPTER.validate_python(raw_content)

            if content.role == "user":
                content_parts: list[ChatCompletionTextObject | ChatCompletionImageObject] = []
                tool_messages: list[ChatCompletionToolMessage] = []

                for part in content.parts:
                    if isinstance(part, str):
                        content_parts.append(ChatCompletionTextObject(type="text", text=part))
                    elif part.text is not None:
                        content_parts.append(ChatCompletionTextObject(type="text", text=part.text))
                    elif part.inline_data is not None:
                        content_parts.append(
                            ChatCompletionImageObject(
                                type="image_url",
                                image_url=ChatCompletionImageUrlObject(
                                    url=f"data:{part.inline_data.mime_type};base64,{part.inline_data.data}"
                                ),
                            )
                        )
                    elif part.function_response is not None:
                        tool_messages.append(
                            ChatCompletionToolMessage(
                                role="tool",
                                tool_call_id=f"call_{part.function_response.name}",
                                content=json.dumps(part.function_response.response),
                            )
                        )

                if content_parts:
                    first_content_part = content_parts[0]
                    if len(content_parts) == 1 and first_content_part["type"] == "text":
                        messages.append(ChatCompletionUserMessage(role="user", content=first_content_part["text"]))
                    else:
                        messages.append(ChatCompletionUserMessage(role="user", content=content_parts))

                messages.extend(tool_messages)

            elif content.role == "model":
                combined_text = ""
                tool_calls: list[ChatCompletionAssistantToolCall] = []

                for part in content.parts:
                    if isinstance(part, str):
                        combined_text += part
                    elif part.text is not None:
                        combined_text += part.text
                    elif part.function_call is not None:
                        tool_calls.append(
                            ChatCompletionAssistantToolCall(
                                id=f"call_{part.function_call.name}",
                                type="function",
                                function=ChatCompletionToolCallFunctionChunk(
                                    name=part.function_call.name,
                                    arguments=json.dumps(part.function_call.args),
                                ),
                            )
                        )

                if tool_calls:
                    messages.append(
                        ChatCompletionAssistantMessage(
                            role="assistant",
                            content=combined_text if combined_text else None,
                            tool_calls=tool_calls,
                        )
                    )
                else:
                    messages.append(
                        ChatCompletionAssistantMessage(
                            role="assistant",
                            content=combined_text if combined_text else None,
                        )
                    )

        return messages

    def translate_completion_to_generate_content(
        self,
        response: ModelResponse,
    ) -> dict[str, object]:
        """
        Transform litellm completion response to Google GenAI generate_content format

        Args:
            response: ModelResponse from litellm.completion

        Returns:
            Dict in Google GenAI generate_content response format
        """

        choice: Final = response.choices[0] if response.choices else None
        if not choice:
            raise ValueError("Invalid completion response: no choices found")

        if not choice.message:
            raise ValueError("Invalid completion response: no message found in choice")

        parts: Final = self._transform_openai_message_to_google_genai_parts(choice.message)

        generate_content_response: Final[dict[str, object]] = {
            "candidates": [
                {
                    "content": {"parts": parts, "role": "model"},
                    "finishReason": self._map_finish_reason(choice.finish_reason),
                    "index": 0,
                    "safetyRatings": [],
                }
            ],
            "usageMetadata": _usage_metadata(response),
        }

        text_content: Final = _text_of(parts)
        if text_content:
            generate_content_response["text"] = text_content

        return generate_content_response

    def translate_streaming_completion_to_generate_content(
        self,
        response: ModelResponse | ModelResponseStream,
        wrapper: GoogleGenAIStreamWrapper,
    ) -> dict[str, object] | None:
        """
        Transform streaming litellm completion chunk to Google GenAI generate_content format

        Args:
            response: Streaming ModelResponse chunk from litellm.completion
            wrapper: GoogleGenAIStreamWrapper instance

        Returns:
            Dict in Google GenAI streaming generate_content response format
        """

        choice: Final = response.choices[0] if response.choices else None
        if not choice:
            return None

        parts: Final[list[dict[str, object]]] = (
            self._transform_openai_delta_to_google_genai_parts_with_accumulation(choice.delta, wrapper)
            if isinstance(choice, StreamingChoices) and choice.delta
            else []
        )

        finish_reason: Final = choice.finish_reason

        if not parts and not finish_reason:
            return None

        streaming_chunk: Final[dict[str, object]] = {
            "candidates": [
                {
                    "content": {"parts": parts, "role": "model"},
                    "finishReason": (self._map_finish_reason(finish_reason) if finish_reason else None),
                    "index": 0,
                    "safetyRatings": [],
                }
            ]
        }

        if finish_reason:
            streaming_chunk["usageMetadata"] = _usage_metadata(response)

        text_content: Final = _text_of(parts)
        if text_content:
            streaming_chunk["text"] = text_content

        return streaming_chunk

    def _transform_openai_message_to_google_genai_parts(
        self,
        message: Message,
    ) -> list[dict[str, object]]:
        """Transform OpenAI message to Google GenAI parts format"""
        parts: Final[list[dict[str, object]]] = []

        if message.content:
            parts.append({"text": message.content})

        for tool_call in message.tool_calls or []:
            if not isinstance(tool_call, ChatCompletionMessageToolCall):
                continue

            try:
                args = _JSON_ADAPTER.validate_json(tool_call.function.arguments or "{}")
            except ValidationError:
                args = {}

            parts.append(
                {
                    "functionCall": {
                        "name": tool_call.function.name or "undefined_tool_name",
                        "args": args,
                    }
                }
            )

        return parts if parts else [{"text": ""}]

    def _transform_openai_delta_to_google_genai_parts_with_accumulation(
        self, delta: Delta, wrapper: GoogleGenAIStreamWrapper
    ) -> list[dict[str, object]]:
        """Transforms OpenAI delta to Google GenAI parts, accumulating streaming tool calls."""

        parts: Final[list[dict[str, object]]] = []

        if delta.content:
            parts.append({"text": delta.content})

        for tool_call in delta.tool_calls or []:
            if not isinstance(tool_call, ChatCompletionDeltaToolCall):
                continue

            tool_call_index = tool_call.index
            accumulated = wrapper.accumulated_tool_calls.setdefault(tool_call_index, {"name": "", "arguments": ""})

            function_name = tool_call.function.name
            args_chunk = tool_call.function.arguments

            if not function_name and not args_chunk:
                verbose_logger.debug("Skipping empty tool call chunk for index: %s", tool_call_index)
                continue

            if function_name:
                accumulated["name"] = function_name

            if args_chunk:
                accumulated["arguments"] += args_chunk

            accumulated_name = accumulated["name"]

            try:
                parsed_args = _JSON_ADAPTER.validate_json(accumulated["arguments"])
            except ValidationError:
                continue

            if accumulated_name:
                parts.append({"functionCall": {"name": accumulated_name, "args": parsed_args}})
                del wrapper.accumulated_tool_calls[tool_call_index]

        return parts

    def _map_finish_reason(self, finish_reason: str | None) -> str:
        """Map OpenAI finish reasons to Google GenAI finish reasons"""
        if not finish_reason:
            return "STOP"

        mapping: Final = {
            "stop": "STOP",
            "length": "MAX_TOKENS",
            "content_filter": "SAFETY",
            "tool_calls": "STOP",
            "function_call": "STOP",
        }

        return mapping.get(finish_reason, "STOP")
