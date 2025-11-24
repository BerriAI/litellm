import json
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

from openai.types.chat.chat_completion_chunk import Choice as OpenAIStreamingChoice

from litellm.types.llms.anthropic import (
    AllAnthropicToolsValues,
    AnthopicMessagesAssistantMessageParam,
    AnthropicFinishReason,
    AnthropicMessagesRequest,
    AnthropicMessagesToolChoice,
    AnthropicMessagesUserMessageParam,
    AnthropicResponseContentBlockRedactedThinking,
    AnthropicResponseContentBlockText,
    AnthropicResponseContentBlockThinking,
    AnthropicResponseContentBlockToolUse,
    ContentBlockDelta,
    ContentJsonBlockDelta,
    ContentTextBlockDelta,
    ContentThinkingBlockDelta,
    ContentThinkingSignatureBlockDelta,
    MessageBlockDelta,
    MessageDelta,
    UsageDelta,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
    AnthropicUsage,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionAssistantToolCall,
    ChatCompletionImageObject,
    ChatCompletionImageUrlObject,
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionRequest,
    ChatCompletionSystemMessage,
    ChatCompletionTextObject,
    ChatCompletionThinkingBlock,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolChoiceFunctionParam,
    ChatCompletionToolChoiceObjectParam,
    ChatCompletionToolChoiceValues,
    ChatCompletionToolMessage,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    ChatCompletionUserMessage,
)
from litellm.types.utils import Choices, ModelResponse, StreamingChoices, Usage

from .streaming_iterator import AnthropicStreamWrapper

if TYPE_CHECKING:
    from litellm.types.llms.anthropic import ContentBlockContentBlockDict


class AnthropicAdapter:
    def __init__(self) -> None:
        pass

    def translate_completion_input_params(
        self, kwargs
    ) -> Optional[ChatCompletionRequest]:
        """
        - translate params, where needed
        - pass rest, as is
        """

        #########################################################
        # Validate required params
        #########################################################
        model = kwargs.pop("model")
        messages = kwargs.pop("messages")
        if not model:
            raise ValueError(
                "Bad Request: model is required for Anthropic Messages Request"
            )
        if not messages:
            raise ValueError(
                "Bad Request: messages is required for Anthropic Messages Request"
            )

        #########################################################
        # Created Typed Request Body
        #########################################################
        request_body = AnthropicMessagesRequest(
            model=model, messages=messages, **kwargs
        )

        translated_body = (
            LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
                anthropic_message_request=request_body
            )
        )

        return translated_body

    def translate_completion_output_params(
        self, response: ModelResponse
    ) -> Optional[AnthropicMessagesResponse]:
        return LiteLLMAnthropicMessagesAdapter().translate_openai_response_to_anthropic(
            response=response
        )

    def translate_completion_output_params_streaming(
        self, completion_stream: Any, model: str
    ) -> Union[AsyncIterator[bytes], None]:
        anthropic_wrapper = AnthropicStreamWrapper(
            completion_stream=completion_stream, model=model
        )
        # Return the SSE-wrapped version for proper event formatting
        return anthropic_wrapper.async_anthropic_sse_wrapper()


class LiteLLMAnthropicMessagesAdapter:
    def __init__(self):
        pass

    ### FOR [BETA] `/v1/messages` endpoint support

    def _extract_signature_from_tool_call(
        self, tool_call: Any
    ) -> Optional[str]:
        """
        Extract signature from a tool call's provider_specific_fields.
        Only checks provider_specific_fields, not thinking blocks.
        """
        signature = None
        
        if hasattr(tool_call, "provider_specific_fields") and tool_call.provider_specific_fields:
            if "thought_signature" in tool_call.provider_specific_fields:
                signature = tool_call.provider_specific_fields["thought_signature"]
        elif (
            hasattr(tool_call.function, "provider_specific_fields")
            and tool_call.function.provider_specific_fields
        ):
            if "thought_signature" in tool_call.function.provider_specific_fields:
                signature = tool_call.function.provider_specific_fields["thought_signature"]
        
        return signature

    def _extract_signature_from_tool_use_content(
        self, content: Dict[str, Any]
    ) -> Optional[str]:
        """
        Extract signature from a tool_use content block's provider_specific_fields.
        """
        provider_specific_fields = content.get("provider_specific_fields", {})
        if provider_specific_fields:
            return provider_specific_fields.get("signature")
        return None


    def translatable_anthropic_params(self) -> List:
        """
        Which anthropic params, we need to translate to the openai format.
        """
        return ["messages", "metadata", "system", "tool_choice", "tools"]

    def translate_anthropic_messages_to_openai(  # noqa: PLR0915
        self,
        messages: List[
            Union[
                AnthropicMessagesUserMessageParam,
                AnthopicMessagesAssistantMessageParam,
            ]
        ],
    ) -> List:
        new_messages: List[AllMessageValues] = []
        for m in messages:
            user_message: Optional[ChatCompletionUserMessage] = None
            tool_message_list: List[ChatCompletionToolMessage] = []
            new_user_content_list: List[
                Union[ChatCompletionTextObject, ChatCompletionImageObject]
            ] = []
            ## USER MESSAGE ##
            if m["role"] == "user":
                ## translate user message
                message_content = m.get("content")
                if message_content and isinstance(message_content, str):
                    user_message = ChatCompletionUserMessage(
                        role="user", content=message_content
                    )
                elif message_content and isinstance(message_content, list):
                    for content in message_content:
                        if content.get("type") == "text":
                            text_obj = ChatCompletionTextObject(
                                type="text", text=content.get("text", "")
                            )
                            new_user_content_list.append(text_obj)
                        elif content.get("type") == "image":
                            # Convert Anthropic image format to OpenAI format
                            source = content.get("source", {})
                            openai_image_url = (
                                self._translate_anthropic_image_to_openai(source)
                            )

                            if openai_image_url:
                                image_url_obj = ChatCompletionImageUrlObject(
                                    url=openai_image_url
                                )
                                image_obj = ChatCompletionImageObject(
                                    type="image_url", image_url=image_url_obj
                                )
                                new_user_content_list.append(image_obj)
                        elif content.get("type") == "tool_result":
                            if "content" not in content:
                                tool_result = ChatCompletionToolMessage(
                                    role="tool",
                                    tool_call_id=content.get("tool_use_id", ""),
                                    content="",
                                )
                                tool_message_list.append(tool_result)
                            elif isinstance(content.get("content"), str):
                                tool_result = ChatCompletionToolMessage(
                                    role="tool",
                                    tool_call_id=content.get("tool_use_id", ""),
                                    content=str(content.get("content", "")),
                                )
                                tool_message_list.append(tool_result)
                            elif isinstance(content.get("content"), list):
                                for c in content.get("content", []):
                                    if isinstance(c, str):
                                        tool_result = ChatCompletionToolMessage(
                                            role="tool",
                                            tool_call_id=content.get("tool_use_id", ""),
                                            content=c,
                                        )
                                        tool_message_list.append(tool_result)
                                    elif isinstance(c, dict):
                                        if c.get("type") == "text":
                                            tool_result = ChatCompletionToolMessage(
                                                role="tool",
                                                tool_call_id=content.get(
                                                    "tool_use_id", ""
                                                ),
                                                content=c.get("text", ""),
                                            )
                                            tool_message_list.append(tool_result)
                                        elif c.get("type") == "image":
                                            # Convert Anthropic image format to OpenAI format for tool results
                                            source = c.get("source", {})
                                            openai_image_url = (
                                                self._translate_anthropic_image_to_openai(
                                                    source
                                                )
                                                or ""
                                            )

                                            tool_result = ChatCompletionToolMessage(
                                                role="tool",
                                                tool_call_id=content.get(
                                                    "tool_use_id", ""
                                                ),
                                                content=openai_image_url,
                                            )
                                            tool_message_list.append(tool_result)

            if len(tool_message_list) > 0:
                new_messages.extend(tool_message_list)

            if user_message is not None:
                new_messages.append(user_message)

            if len(new_user_content_list) > 0:
                new_messages.append({"role": "user", "content": new_user_content_list})  # type: ignore

            ## ASSISTANT MESSAGE ##
            assistant_message_str: Optional[str] = None
            tool_calls: List[ChatCompletionAssistantToolCall] = []
            thinking_blocks: List[
                Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]
            ] = []
            if m["role"] == "assistant":
                if isinstance(m.get("content"), str):
                    assistant_message_str = str(m.get("content", ""))
                elif isinstance(m.get("content"), list):
                    for content in m.get("content", []):
                        if isinstance(content, str):
                            assistant_message_str = str(content)
                        elif isinstance(content, dict):
                            if content.get("type") == "text":
                                if assistant_message_str is None:
                                    assistant_message_str = content.get("text", "")
                                else:
                                    assistant_message_str += content.get("text", "")
                            elif content.get("type") == "tool_use":
                                function_chunk: ChatCompletionToolCallFunctionChunk = {
                                    "name": content.get("name", ""),
                                    "arguments": json.dumps(content.get("input", {})),
                                }
                                signature = self._extract_signature_from_tool_use_content(content)
                                
                                if signature:
                                    provider_specific_fields: Dict[str, Any] = (
                                        function_chunk.get("provider_specific_fields") or {}
                                    )
                                    provider_specific_fields["thought_signature"] = signature
                                    function_chunk["provider_specific_fields"] = provider_specific_fields

                                tool_calls.append(
                                    ChatCompletionAssistantToolCall(
                                        id=content.get("id", ""),
                                        type="function",
                                        function=function_chunk,
                                    )
                                )
                            elif content.get("type") == "thinking":
                                thinking_block = ChatCompletionThinkingBlock(
                                    type="thinking",
                                    thinking=content.get("thinking") or "",
                                    signature=content.get("signature") or "",
                                    cache_control=content.get("cache_control", {}),
                                )
                                thinking_blocks.append(thinking_block)
                            elif content.get("type") == "redacted_thinking":
                                redacted_thinking_block = (
                                    ChatCompletionRedactedThinkingBlock(
                                        type="redacted_thinking",
                                        data=content.get("data") or "",
                                        cache_control=content.get("cache_control", {}),
                                    )
                                )
                                thinking_blocks.append(redacted_thinking_block)

            if (
                assistant_message_str is not None
                or len(tool_calls) > 0
                or len(thinking_blocks) > 0
            ):
                assistant_message = ChatCompletionAssistantMessage(
                    role="assistant",
                    content=assistant_message_str,
                    thinking_blocks=(
                        thinking_blocks if len(thinking_blocks) > 0 else None
                    ),
                )
                if len(tool_calls) > 0:
                    assistant_message["tool_calls"] = tool_calls
                if len(thinking_blocks) > 0:
                    assistant_message["thinking_blocks"] = thinking_blocks  # type: ignore
                new_messages.append(assistant_message)

        return new_messages

    def translate_anthropic_tool_choice_to_openai(
        self, tool_choice: AnthropicMessagesToolChoice
    ) -> ChatCompletionToolChoiceValues:
        if tool_choice["type"] == "any":
            return "required"
        elif tool_choice["type"] == "auto":
            return "auto"
        elif tool_choice["type"] == "tool":
            tc_function_param = ChatCompletionToolChoiceFunctionParam(
                name=tool_choice.get("name", "")
            )
            return ChatCompletionToolChoiceObjectParam(
                type="function", function=tc_function_param
            )
        else:
            raise ValueError(
                "Incompatible tool choice param submitted - {}".format(tool_choice)
            )

    def translate_anthropic_tools_to_openai(
        self, tools: List[AllAnthropicToolsValues]
    ) -> List[ChatCompletionToolParam]:
        new_tools: List[ChatCompletionToolParam] = []
        mapped_tool_params = ["name", "input_schema", "description"]
        for tool in tools:
            function_chunk = ChatCompletionToolParamFunctionChunk(
                name=tool["name"],
            )
            if "input_schema" in tool:
                function_chunk["parameters"] = tool["input_schema"]  # type: ignore
            if "description" in tool:
                function_chunk["description"] = tool["description"]  # type: ignore

            for k, v in tool.items():
                if k not in mapped_tool_params:  # pass additional computer kwargs
                    function_chunk.setdefault("parameters", {}).update({k: v})
            new_tools.append(
                ChatCompletionToolParam(type="function", function=function_chunk)
            )

        return new_tools

    def translate_anthropic_to_openai(
        self, anthropic_message_request: AnthropicMessagesRequest
    ) -> ChatCompletionRequest:
        """
        This is used by the beta Anthropic Adapter, for translating anthropic `/v1/messages` requests to the openai format.
        """
        # Debug: Processing Anthropic message request
        new_messages: List[AllMessageValues] = []

        ## CONVERT ANTHROPIC MESSAGES TO OPENAI
        messages_list: List[
            Union[
                AnthropicMessagesUserMessageParam, AnthopicMessagesAssistantMessageParam
            ]
        ] = cast(
            List[
                Union[
                    AnthropicMessagesUserMessageParam,
                    AnthopicMessagesAssistantMessageParam,
                ]
            ],
            anthropic_message_request["messages"],
        )
        new_messages = self.translate_anthropic_messages_to_openai(
            messages=messages_list
        )
        ## ADD SYSTEM MESSAGE TO MESSAGES
        if "system" in anthropic_message_request:
            system_content = anthropic_message_request["system"]
            if system_content:
                new_messages.insert(
                    0,
                    ChatCompletionSystemMessage(role="system", content=system_content),
                )

        new_kwargs: ChatCompletionRequest = {
            "model": anthropic_message_request["model"],
            "messages": new_messages,
        }
        ## CONVERT METADATA (user_id)
        if "metadata" in anthropic_message_request:
            metadata = anthropic_message_request["metadata"]
            if metadata and "user_id" in metadata:
                new_kwargs["user"] = metadata["user_id"]

        # Pass litellm proxy specific metadata
        if "litellm_metadata" in anthropic_message_request:
            # metadata will be passed to litellm.acompletion(), it's a litellm_param
            new_kwargs["metadata"] = anthropic_message_request.pop("litellm_metadata")

        ## CONVERT TOOL CHOICE
        if "tool_choice" in anthropic_message_request:
            tool_choice = anthropic_message_request["tool_choice"]
            if tool_choice:
                new_kwargs["tool_choice"] = (
                    self.translate_anthropic_tool_choice_to_openai(
                        tool_choice=cast(AnthropicMessagesToolChoice, tool_choice)
                    )
                )
        ## CONVERT TOOLS
        if "tools" in anthropic_message_request:
            tools = anthropic_message_request["tools"]
            if tools:
                new_kwargs["tools"] = self.translate_anthropic_tools_to_openai(
                    tools=cast(List[AllAnthropicToolsValues], tools)
                )

        translatable_params = self.translatable_anthropic_params()
        for k, v in anthropic_message_request.items():
            if k not in translatable_params:  # pass remaining params as is
                new_kwargs[k] = v  # type: ignore

        return new_kwargs

    def _translate_anthropic_image_to_openai(self, image_source: dict) -> Optional[str]:
        """
        Translate Anthropic image source format to OpenAI-compatible image URL.

        Anthropic supports two image source formats:
        1. Base64: {"type": "base64", "media_type": "image/jpeg", "data": "..."}
        2. URL: {"type": "url", "url": "https://..."}

        Returns the properly formatted image URL string, or None if invalid format.
        """
        if not isinstance(image_source, dict):
            return None

        source_type = image_source.get("type")

        if source_type == "base64":
            # Base64 image format
            media_type = image_source.get("media_type", "image/jpeg")
            image_data = image_source.get("data", "")
            if image_data:
                return f"data:{media_type};base64,{image_data}"
        elif source_type == "url":
            # URL-referenced image format
            return image_source.get("url", "")

        return None

    def _translate_openai_content_to_anthropic(self, choices: List[Choices]) -> List[
        Union[
            AnthropicResponseContentBlockText,
            AnthropicResponseContentBlockToolUse,
            AnthropicResponseContentBlockThinking,
            AnthropicResponseContentBlockRedactedThinking,
        ]
    ]:
        new_content: List[
            Union[
                AnthropicResponseContentBlockText,
                AnthropicResponseContentBlockToolUse,
                AnthropicResponseContentBlockThinking,
                AnthropicResponseContentBlockRedactedThinking,
            ]
        ] = []
        for choice in choices:
            # Handle thinking blocks first
            if (
                hasattr(choice.message, "thinking_blocks")
                and choice.message.thinking_blocks
            ):
                for thinking_block in choice.message.thinking_blocks:
                    if thinking_block.get("type") == "thinking":
                        thinking_value = thinking_block.get("thinking", "")
                        signature_value = thinking_block.get("signature", "")
                        new_content.append(
                            AnthropicResponseContentBlockThinking(
                                type="thinking",
                                thinking=(
                                    str(thinking_value)
                                    if thinking_value is not None
                                    else ""
                                ),
                                signature=(
                                    str(signature_value)
                                    if signature_value is not None
                                    else None
                                ),
                            )
                        )
                    elif thinking_block.get("type") == "redacted_thinking":
                        data_value = thinking_block.get("data", "")
                        new_content.append(
                            AnthropicResponseContentBlockRedactedThinking(
                                type="redacted_thinking",
                                data=str(data_value) if data_value is not None else "",
                            )
                        )

            # Handle tool calls
            if (
                choice.message.tool_calls is not None
                and len(choice.message.tool_calls) > 0
            ):
                for tool_call in choice.message.tool_calls:
                    # Extract signature from provider_specific_fields only
                    signature = self._extract_signature_from_tool_call(tool_call)
                    
                    provider_specific_fields = {}
                    if signature:
                        provider_specific_fields["signature"] = signature
                    
                    tool_use_block = AnthropicResponseContentBlockToolUse(
                        type="tool_use",
                        id=tool_call.id,
                        name=tool_call.function.name or "",
                        input=(
                            json.loads(tool_call.function.arguments)
                            if tool_call.function.arguments
                            else {}
                        ),
                    )
                    # Add provider_specific_fields if signature is present
                    if provider_specific_fields:
                        tool_use_block.provider_specific_fields = provider_specific_fields
                    new_content.append(tool_use_block)
            # Handle text content
            elif choice.message.content is not None:
                new_content.append(
                    AnthropicResponseContentBlockText(
                        type="text", text=choice.message.content
                    )
                )

        return new_content

    def _translate_openai_finish_reason_to_anthropic(
        self, openai_finish_reason: str
    ) -> AnthropicFinishReason:
        if openai_finish_reason == "stop":
            return "end_turn"
        elif openai_finish_reason == "length":
            return "max_tokens"
        elif openai_finish_reason == "tool_calls":
            return "tool_use"
        return "end_turn"

    def translate_openai_response_to_anthropic(
        self, response: ModelResponse
    ) -> AnthropicMessagesResponse:
        ## translate content block
        anthropic_content = self._translate_openai_content_to_anthropic(choices=response.choices)  # type: ignore
        ## extract finish reason
        anthropic_finish_reason = self._translate_openai_finish_reason_to_anthropic(
            openai_finish_reason=response.choices[0].finish_reason  # type: ignore
        )
        # extract usage
        usage: Usage = getattr(response, "usage")
        anthropic_usage = AnthropicUsage(
            input_tokens=usage.prompt_tokens or 0,
            output_tokens=usage.completion_tokens or 0,
        )
        translated_obj = AnthropicMessagesResponse(
            id=response.id,
            type="message",
            role="assistant",
            model=response.model or "unknown-model",
            stop_sequence=None,
            usage=anthropic_usage,
            content=anthropic_content,  # type: ignore
            stop_reason=anthropic_finish_reason,
        )

        return translated_obj

    def _translate_streaming_openai_chunk_to_anthropic_content_block(
        self, choices: List[Union[OpenAIStreamingChoice, StreamingChoices]]
    ) -> Tuple[
        Literal["text", "tool_use", "thinking"],
        "ContentBlockContentBlockDict",
    ]:
        from litellm._uuid import uuid
        from litellm.types.llms.anthropic import TextBlock, ToolUseBlock

        for choice in choices:
            if choice.delta.content is not None and len(choice.delta.content) > 0:
                return "text", TextBlock(type="text", text="")
            elif (
                choice.delta.tool_calls is not None
                and len(choice.delta.tool_calls) > 0
                and choice.delta.tool_calls[0].function is not None
            ):
                return "tool_use", ToolUseBlock(
                    type="tool_use",
                    id=choice.delta.tool_calls[0].id or str(uuid.uuid4()),
                    name=choice.delta.tool_calls[0].function.name or "",
                    input={},
                )
            elif isinstance(choice, StreamingChoices) and hasattr(
                choice.delta, "thinking_blocks"
            ):
                thinking_blocks = choice.delta.thinking_blocks or []
                if len(thinking_blocks) > 0:
                    thinking_block = thinking_blocks[0]
                    if thinking_block["type"] == "thinking":
                        thinking = thinking_block.get("thinking") or ""
                        signature = thinking_block.get("signature") or ""

                        assert isinstance(thinking, str)
                        assert isinstance(signature, str)

                        if thinking and signature:
                            raise ValueError(
                                "Both `thinking` and `signature` in a single streaming chunk isn't supported."
                            )

                        return "thinking", ChatCompletionThinkingBlock(
                            type="thinking", thinking=thinking, signature=signature
                        )

        return "text", TextBlock(type="text", text="")

    def _translate_streaming_openai_chunk_to_anthropic(
        self, choices: List[Union[OpenAIStreamingChoice, StreamingChoices]]
    ) -> Tuple[
        Literal["text_delta", "input_json_delta", "thinking_delta", "signature_delta"],
        Union[
            ContentTextBlockDelta,
            ContentJsonBlockDelta,
            ContentThinkingBlockDelta,
            ContentThinkingSignatureBlockDelta,
        ],
    ]:

        text: str = ""
        reasoning_content: str = ""
        reasoning_signature: str = ""
        partial_json: Optional[str] = None
        for choice in choices:
            if choice.delta.content is not None and len(choice.delta.content) > 0:
                text += choice.delta.content
            elif choice.delta.tool_calls is not None:
                partial_json = ""
                for tool in choice.delta.tool_calls:
                    if (
                        tool.function is not None
                        and tool.function.arguments is not None
                    ):
                        partial_json = (partial_json or "") + tool.function.arguments
            elif isinstance(choice, StreamingChoices) and hasattr(
                choice.delta, "thinking_blocks"
            ):
                thinking_blocks = choice.delta.thinking_blocks or []
                if len(thinking_blocks) > 0:
                    for thinking_block in thinking_blocks:
                        if thinking_block["type"] == "thinking":
                            thinking = thinking_block.get("thinking") or ""
                            signature = thinking_block.get("signature") or ""

                            assert isinstance(thinking, str)
                            assert isinstance(signature, str)

                            reasoning_content += thinking
                            reasoning_signature += signature

        if reasoning_content and reasoning_signature:
            raise ValueError(
                "Both `reasoning` and `signature` in a single streaming chunk isn't supported."
            )

        if partial_json is not None:
            return "input_json_delta", ContentJsonBlockDelta(
                type="input_json_delta", partial_json=partial_json
            )
        elif reasoning_content:
            return "thinking_delta", ContentThinkingBlockDelta(
                type="thinking_delta", thinking=reasoning_content
            )
        elif reasoning_signature:
            return "signature_delta", ContentThinkingSignatureBlockDelta(
                type="signature_delta", signature=reasoning_signature
            )
        else:
            return "text_delta", ContentTextBlockDelta(type="text_delta", text=text)

    def translate_streaming_openai_response_to_anthropic(
        self, response: ModelResponse, current_content_block_index: int
    ) -> Union[ContentBlockDelta, MessageBlockDelta]:
        ## base case - final chunk w/ finish reason
        if response.choices[0].finish_reason is not None:
            delta = MessageDelta(
                stop_reason=self._translate_openai_finish_reason_to_anthropic(
                    response.choices[0].finish_reason
                ),
            )
            if getattr(response, "usage", None) is not None:
                litellm_usage_chunk: Optional[Usage] = response.usage  # type: ignore
            elif (
                hasattr(response, "_hidden_params")
                and "usage" in response._hidden_params
            ):
                litellm_usage_chunk = response._hidden_params["usage"]
            else:
                litellm_usage_chunk = None
            if litellm_usage_chunk is not None:
                usage_delta = UsageDelta(
                    input_tokens=litellm_usage_chunk.prompt_tokens or 0,
                    output_tokens=litellm_usage_chunk.completion_tokens or 0,
                )
            else:
                usage_delta = UsageDelta(input_tokens=0, output_tokens=0)
            return MessageBlockDelta(
                type="message_delta", delta=delta, usage=usage_delta
            )
        (
            type_of_content,
            content_block_delta,
        ) = self._translate_streaming_openai_chunk_to_anthropic(
            choices=response.choices  # type: ignore
        )
        return ContentBlockDelta(
            type="content_block_delta",
            index=current_content_block_index,
            delta=content_block_delta,
        )
