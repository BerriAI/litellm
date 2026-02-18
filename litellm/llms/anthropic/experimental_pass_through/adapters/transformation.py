import hashlib
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

# OpenAI has a 64-character limit for function/tool names
# Anthropic does not have this limit, so we need to truncate long names
OPENAI_MAX_TOOL_NAME_LENGTH = 64
TOOL_NAME_HASH_LENGTH = 8
TOOL_NAME_PREFIX_LENGTH = OPENAI_MAX_TOOL_NAME_LENGTH - TOOL_NAME_HASH_LENGTH - 1  # 55


def truncate_tool_name(name: str) -> str:
    """
    Truncate tool names that exceed OpenAI's 64-character limit.

    Uses format: {55-char-prefix}_{8-char-hash} to avoid collisions
    when multiple tools have similar long names.

    Args:
        name: The original tool name

    Returns:
        The original name if <= 64 chars, otherwise truncated with hash
    """
    if len(name) <= OPENAI_MAX_TOOL_NAME_LENGTH:
        return name

    # Create deterministic hash from full name to avoid collisions
    name_hash = hashlib.sha256(name.encode()).hexdigest()[:TOOL_NAME_HASH_LENGTH]
    return f"{name[:TOOL_NAME_PREFIX_LENGTH]}_{name_hash}"


def create_tool_name_mapping(
    tools: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Create a mapping of truncated tool names to original names.

    Args:
        tools: List of tool definitions with 'name' field

    Returns:
        Dict mapping truncated names to original names (only for truncated tools)
    """
    mapping: Dict[str, str] = {}
    for tool in tools:
        original_name = tool.get("name", "")
        truncated_name = truncate_tool_name(original_name)
        if truncated_name != original_name:
            mapping[truncated_name] = original_name
    return mapping

from openai.types.chat.chat_completion_chunk import Choice as OpenAIStreamingChoice

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    parse_tool_call_arguments,
)
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
        Translate Anthropic request params to OpenAI format.

        - translate params, where needed
        - pass rest, as is

        Note: Use translate_completion_input_params_with_tool_mapping() if you need
        the tool name mapping for restoring original names in responses.
        """
        result, _ = self.translate_completion_input_params_with_tool_mapping(kwargs)
        return result

    def translate_completion_input_params_with_tool_mapping(
        self, kwargs
    ) -> Tuple[Optional[ChatCompletionRequest], Dict[str, str]]:
        """
        Translate Anthropic request params to OpenAI format, returning tool name mapping.

        This method handles truncation of tool names that exceed OpenAI's 64-character
        limit. The mapping allows restoring original names when translating responses.

        Returns:
            Tuple of (openai_request, tool_name_mapping)
            - tool_name_mapping maps truncated tool names back to original names
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

        translated_body, tool_name_mapping = (
            LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
                anthropic_message_request=request_body
            )
        )

        return translated_body, tool_name_mapping

    def translate_completion_output_params(
        self,
        response: ModelResponse,
        tool_name_mapping: Optional[Dict[str, str]] = None,
    ) -> Optional[AnthropicMessagesResponse]:
        """
        Translate OpenAI response to Anthropic format.

        Args:
            response: The OpenAI ModelResponse
            tool_name_mapping: Optional mapping of truncated tool names to original names.
                              Used to restore original names for tools that exceeded
                              OpenAI's 64-char limit.
        """
        return LiteLLMAnthropicMessagesAdapter().translate_openai_response_to_anthropic(
            response=response,
            tool_name_mapping=tool_name_mapping,
        )

    def translate_completion_output_params_streaming(
        self,
        completion_stream: Any,
        model: str,
        tool_name_mapping: Optional[Dict[str, str]] = None,
    ) -> Union[AsyncIterator[bytes], None]:
        """
        Translate OpenAI streaming response to Anthropic format.

        Args:
            completion_stream: The OpenAI streaming response
            model: The model name
            tool_name_mapping: Optional mapping of truncated tool names to original names.
        """
        anthropic_wrapper = AnthropicStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            tool_name_mapping=tool_name_mapping,
        )
        # Return the SSE-wrapped version for proper event formatting
        return anthropic_wrapper.async_anthropic_sse_wrapper()


class LiteLLMAnthropicMessagesAdapter:
    def __init__(self):
        pass

    ### FOR [BETA] `/v1/messages` endpoint support

    def _extract_signature_from_tool_call(self, tool_call: Any) -> Optional[str]:
        """
        Extract signature from a tool call's provider_specific_fields.
        Only checks provider_specific_fields, not thinking blocks.
        """
        signature = None

        if (
            hasattr(tool_call, "provider_specific_fields")
            and tool_call.provider_specific_fields
        ):
            if "thought_signature" in tool_call.provider_specific_fields:
                signature = tool_call.provider_specific_fields["thought_signature"]
        elif (
            hasattr(tool_call.function, "provider_specific_fields")
            and tool_call.function.provider_specific_fields
        ):
            if "thought_signature" in tool_call.function.provider_specific_fields:
                signature = tool_call.function.provider_specific_fields[
                    "thought_signature"
                ]

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

    def _add_cache_control_if_applicable(
        self,
        source: Any,
        target: Any,
        model: Optional[str],
    ) -> None:
        """
        Extract cache_control from source and add to target if it should be preserved.

        This method accepts Any type to support both regular dicts and TypedDict objects.
        TypedDict objects (like ChatCompletionTextObject, ChatCompletionImageObject, etc.)
        are dicts at runtime but have specific types at type-check time. Using Any allows
        this method to work with both while maintaining runtime correctness.

        Args:
            source: Dict or TypedDict containing potential cache_control field
            target: Dict or TypedDict to add cache_control to
            model: Model name to check if cache_control should be preserved
        """
        # TypedDict objects are dicts at runtime, so .get() works
        cache_control = source.get("cache_control") if isinstance(source, dict) else getattr(source, "cache_control", None)
        if cache_control and model and self.is_anthropic_claude_model(model):
            # TypedDict objects support dict operations at runtime
            # Use type ignore consistent with codebase pattern (see anthropic/chat/transformation.py:432)
            if isinstance(target, dict):
                target["cache_control"] = cache_control  # type: ignore[typeddict-item]
            else:
                # Fallback for non-dict objects (shouldn't happen in practice)
                cast(Dict[str, Any], target)["cache_control"] = cache_control

    def translatable_anthropic_params(self) -> List:
        """
        Which anthropic params, we need to translate to the openai format.
        """
        return ["messages", "metadata", "system", "tool_choice", "tools", "thinking", "output_format"]

    def translate_anthropic_messages_to_openai(  # noqa: PLR0915
        self,
        messages: List[
            Union[
                AnthropicMessagesUserMessageParam,
                AnthopicMessagesAssistantMessageParam,
            ]
        ],
        model: Optional[str] = None,
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
                            self._add_cache_control_if_applicable(content, text_obj, model)
                            new_user_content_list.append(text_obj)  # type: ignore
                        elif content.get("type") == "image":
                            # Convert Anthropic image format to OpenAI format
                            source = content.get("source", {})
                            openai_image_url = (
                                self._translate_anthropic_image_to_openai(cast(dict, source))
                            )

                            if openai_image_url:
                                image_url_obj = ChatCompletionImageUrlObject(
                                    url=openai_image_url
                                )
                                image_obj = ChatCompletionImageObject(
                                    type="image_url", image_url=image_url_obj
                                )
                                self._add_cache_control_if_applicable(content, image_obj, model)
                                new_user_content_list.append(image_obj)  # type: ignore
                        elif content.get("type") == "document":
                            # Convert Anthropic document format (PDF, etc.) to OpenAI format
                            source = content.get("source", {})
                            openai_image_url = (
                                self._translate_anthropic_image_to_openai(cast(dict, source))
                            )

                            if openai_image_url:
                                image_url_obj = ChatCompletionImageUrlObject(
                                    url=openai_image_url
                                )
                                doc_obj = ChatCompletionImageObject(
                                    type="image_url", image_url=image_url_obj
                                )
                                self._add_cache_control_if_applicable(content, doc_obj, model)
                                new_user_content_list.append(doc_obj)  # type: ignore
                        elif content.get("type") == "tool_result":
                            if "content" not in content:
                                tool_result = ChatCompletionToolMessage(
                                    role="tool",
                                    tool_call_id=content.get("tool_use_id", ""),
                                    content="",
                                )
                                self._add_cache_control_if_applicable(content, tool_result, model)
                                tool_message_list.append(tool_result)  # type: ignore[arg-type]
                            elif isinstance(content.get("content"), str):
                                tool_result = ChatCompletionToolMessage(
                                    role="tool",
                                    tool_call_id=content.get("tool_use_id", ""),
                                    content=str(content.get("content", "")),
                                )
                                self._add_cache_control_if_applicable(content, tool_result, model)
                                tool_message_list.append(tool_result)  # type: ignore[arg-type]
                            elif isinstance(content.get("content"), list):
                                # Combine all content items into a single tool message
                                # to avoid creating multiple tool_result blocks with the same ID
                                # (each tool_use must have exactly one tool_result)
                                content_items = list(content.get("content", []))

                                # For single-item content, maintain backward compatibility with string/url format
                                if len(content_items) == 1:
                                    c = content_items[0]
                                    if isinstance(c, str):
                                        tool_result = ChatCompletionToolMessage(
                                            role="tool",
                                            tool_call_id=content.get("tool_use_id", ""),
                                            content=c,
                                        )
                                        self._add_cache_control_if_applicable(content, tool_result, model)
                                        tool_message_list.append(tool_result)  # type: ignore[arg-type]
                                    elif isinstance(c, dict):
                                        if c.get("type") == "text":
                                            tool_result = ChatCompletionToolMessage(
                                                role="tool",
                                                tool_call_id=content.get(
                                                    "tool_use_id", ""
                                                ),
                                                content=c.get("text", ""),
                                            )
                                            self._add_cache_control_if_applicable(content, tool_result, model)
                                            tool_message_list.append(tool_result)  # type: ignore[arg-type]
                                        elif c.get("type") == "image":
                                            source = c.get("source", {})
                                            openai_image_url = (
                                                self._translate_anthropic_image_to_openai(
                                                    cast(dict, source)
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
                                            self._add_cache_control_if_applicable(content, tool_result, model)
                                            tool_message_list.append(tool_result)  # type: ignore[arg-type]
                                else:
                                    # For multiple content items, combine into a single tool message
                                    # with list content to preserve all items while having one tool_use_id
                                    combined_content_parts: List[
                                        Union[
                                            ChatCompletionTextObject,
                                            ChatCompletionImageObject,
                                        ]
                                    ] = []
                                    for c in content_items:
                                        if isinstance(c, str):
                                            combined_content_parts.append(
                                                ChatCompletionTextObject(
                                                    type="text", text=c
                                                )
                                            )
                                        elif isinstance(c, dict):
                                            if c.get("type") == "text":
                                                combined_content_parts.append(
                                                    ChatCompletionTextObject(
                                                        type="text",
                                                        text=c.get("text", ""),
                                                    )
                                                )
                                            elif c.get("type") == "image":
                                                source = c.get("source", {})
                                                openai_image_url = (
                                                    self._translate_anthropic_image_to_openai(
                                                        cast(dict, source)
                                                    )
                                                    or ""
                                                )
                                                if openai_image_url:
                                                    combined_content_parts.append(
                                                        ChatCompletionImageObject(
                                                            type="image_url",
                                                            image_url=ChatCompletionImageUrlObject(
                                                                url=openai_image_url
                                                            ),
                                                        )
                                                    )
                                    # Create a single tool message with combined content
                                    if combined_content_parts:
                                        tool_result = ChatCompletionToolMessage(
                                            role="tool",
                                            tool_call_id=content.get("tool_use_id", ""),
                                            content=combined_content_parts,  # type: ignore
                                        )
                                        self._add_cache_control_if_applicable(content, tool_result, model)
                                        tool_message_list.append(tool_result)  # type: ignore[arg-type]

            if len(tool_message_list) > 0:
                new_messages.extend(tool_message_list)

            if user_message is not None:
                new_messages.append(user_message)

            if len(new_user_content_list) > 0:
                new_messages.append({"role": "user", "content": new_user_content_list})  # type: ignore

            ## ASSISTANT MESSAGE ##
            assistant_message_str: Optional[str] = None
            assistant_content_list: List[Dict[str, Any]] = []  # For content blocks with cache_control
            has_cache_control_in_text = False
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
                                text_block: Dict[str, Any] = {
                                    "type": "text",
                                    "text": content.get("text", ""),
                                }
                                self._add_cache_control_if_applicable(content, text_block, model)
                                if "cache_control" in text_block:
                                    has_cache_control_in_text = True
                                assistant_content_list.append(text_block)
                            elif content.get("type") == "tool_use":
                                # Truncate tool name for OpenAI's 64-char limit
                                tool_name = truncate_tool_name(content.get("name", ""))
                                function_chunk: ChatCompletionToolCallFunctionChunk = {
                                    "name": tool_name,
                                    "arguments": json.dumps(content.get("input", {})),
                                }
                                signature = (
                                    self._extract_signature_from_tool_use_content(
                                        cast(Dict[str, Any], content)
                                    )
                                )

                                if signature:
                                    provider_specific_fields: Dict[str, Any] = (
                                        function_chunk.get("provider_specific_fields")
                                        or {}
                                    )
                                    provider_specific_fields["thought_signature"] = (
                                        signature
                                    )
                                    function_chunk["provider_specific_fields"] = (
                                        provider_specific_fields
                                    )

                                tool_call = ChatCompletionAssistantToolCall(
                                    id=content.get("id", ""),
                                    type="function",
                                    function=function_chunk,
                                )
                                self._add_cache_control_if_applicable(content, tool_call, model)
                                tool_calls.append(tool_call)
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
                or len(assistant_content_list) > 0
                or len(tool_calls) > 0
                or len(thinking_blocks) > 0
            ):
                # Use list format if any text block has cache_control, otherwise use string
                if has_cache_control_in_text and len(assistant_content_list) > 0:
                    assistant_content: Any = assistant_content_list
                elif len(assistant_content_list) > 0 and not has_cache_control_in_text:
                    # Concatenate text blocks into string when no cache_control
                    assistant_content = "".join(
                        block.get("text", "") for block in assistant_content_list
                    )
                else:
                    assistant_content = assistant_message_str

                assistant_message = ChatCompletionAssistantMessage(
                    role="assistant",
                    content=assistant_content,
                    thinking_blocks=(
                        thinking_blocks if len(thinking_blocks) > 0 else None
                    ),
                )
                if len(tool_calls) > 0:
                    assistant_message["tool_calls"] = tool_calls  # type: ignore
                if len(thinking_blocks) > 0:
                    assistant_message["thinking_blocks"] = thinking_blocks  # type: ignore
                new_messages.append(assistant_message)

        return new_messages

    @staticmethod
    def translate_anthropic_thinking_to_reasoning_effort(
        thinking: Dict[str, Any]
    ) -> Optional[str]:
        """
        Translate Anthropic's thinking parameter to OpenAI's reasoning_effort.

        Anthropic thinking format: {'type': 'enabled'|'disabled', 'budget_tokens': int}
        OpenAI reasoning_effort: 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'default'

        Mapping:
        - budget_tokens >= 10000 -> 'high'
        - budget_tokens >= 5000  -> 'medium'
        - budget_tokens >= 2000  -> 'low'
        - budget_tokens < 2000   -> 'minimal'
        """
        if not isinstance(thinking, dict):
            return None

        thinking_type = thinking.get("type", "disabled")

        if thinking_type == "disabled":
            return None
        elif thinking_type == "enabled":
            budget_tokens = thinking.get("budget_tokens", 0)
            if budget_tokens >= 10000:
                return "high"
            elif budget_tokens >= 5000:
                return "medium"
            elif budget_tokens >= 2000:
                return "low"
            else:
                return "minimal"

        return None

    @staticmethod
    def is_anthropic_claude_model(model: str) -> bool:
        """
        Check if the model is an Anthropic Claude model that supports the thinking parameter.

        Returns True for:
        - anthropic/* models
        - bedrock/*anthropic* models (including converse)
        - vertex_ai/*claude* models
        """
        model_lower = model.lower()
        return (
            "anthropic" in model_lower
            or "claude" in model_lower
        )

    @staticmethod
    def translate_thinking_for_model(
        thinking: Dict[str, Any],
        model: str,
    ) -> Dict[str, Any]:
        """
        Translate Anthropic thinking parameter based on the target model.

        For Claude/Anthropic models: returns {'thinking': <original_thinking>}
            - Preserves exact budget_tokens value

        For non-Claude models: returns {'reasoning_effort': <mapped_value>}
            - Converts thinking to reasoning_effort to avoid UnsupportedParamsError

        Args:
            thinking: Anthropic thinking dict with 'type' and 'budget_tokens'
            model: The target model name

        Returns:
            Dict with either 'thinking' or 'reasoning_effort' key
        """
        if LiteLLMAnthropicMessagesAdapter.is_anthropic_claude_model(model):
            return {"thinking": thinking}
        else:
            reasoning_effort = LiteLLMAnthropicMessagesAdapter.translate_anthropic_thinking_to_reasoning_effort(
                thinking
            )
            if reasoning_effort:
                return {"reasoning_effort": reasoning_effort}
            return {}

    def translate_anthropic_tool_choice_to_openai(
        self, tool_choice: AnthropicMessagesToolChoice
    ) -> ChatCompletionToolChoiceValues:
        if tool_choice["type"] == "any":
            return "required"
        elif tool_choice["type"] == "auto":
            return "auto"
        elif tool_choice["type"] == "tool":
            # Truncate tool name if it exceeds OpenAI's 64-char limit
            original_name = tool_choice.get("name", "")
            truncated_name = truncate_tool_name(original_name)
            tc_function_param = ChatCompletionToolChoiceFunctionParam(
                name=truncated_name
            )
            return ChatCompletionToolChoiceObjectParam(
                type="function", function=tc_function_param
            )
        else:
            raise ValueError(
                "Incompatible tool choice param submitted - {}".format(tool_choice)
            )

    def translate_anthropic_tools_to_openai(
        self, tools: List[AllAnthropicToolsValues], model: Optional[str] = None
    ) -> Tuple[List[ChatCompletionToolParam], Dict[str, str]]:
        """
        Translate Anthropic tools to OpenAI format.

        Returns:
            Tuple of (translated_tools, tool_name_mapping)
            - tool_name_mapping maps truncated names back to original names
              for tools that exceeded OpenAI's 64-char limit
        """
        new_tools: List[ChatCompletionToolParam] = []
        tool_name_mapping: Dict[str, str] = {}
        mapped_tool_params = ["name", "input_schema", "description", "cache_control"]
        for tool in tools:
            original_name = tool["name"]
            truncated_name = truncate_tool_name(original_name)

            # Store mapping if name was truncated
            if truncated_name != original_name:
                tool_name_mapping[truncated_name] = original_name

            function_chunk = ChatCompletionToolParamFunctionChunk(
                name=truncated_name,
            )
            if "input_schema" in tool:
                function_chunk["parameters"] = tool["input_schema"]  # type: ignore
            if "description" in tool:
                function_chunk["description"] = tool["description"]  # type: ignore

            for k, v in tool.items():
                if k not in mapped_tool_params:  # pass additional computer kwargs
                    function_chunk.setdefault("parameters", {}).update({k: v})
            tool_param = ChatCompletionToolParam(type="function", function=function_chunk)
            self._add_cache_control_if_applicable(tool, tool_param, model)
            new_tools.append(tool_param)  # type: ignore[arg-type]

        return new_tools, tool_name_mapping  # type: ignore[return-value]

    def translate_anthropic_output_format_to_openai(
        self, output_format: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Translate Anthropic's output_format to OpenAI's response_format.

        Anthropic output_format: {"type": "json_schema", "schema": {...}}
        OpenAI response_format: {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}

        Args:
            output_format: Anthropic output_format dict with 'type' and 'schema'

        Returns:
            OpenAI-compatible response_format dict, or None if invalid
        """
        if not isinstance(output_format, dict):
            return None

        output_type = output_format.get("type")
        if output_type != "json_schema":
            return None

        schema = output_format.get("schema")
        if not schema:
            return None

        # Convert to OpenAI response_format structure
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "schema": schema,
                "strict": True,
            },
        }

    def _add_system_message_to_messages(
        self,
        new_messages: List[AllMessageValues],
        anthropic_message_request: AnthropicMessagesRequest,
    ) -> None:
        """Add system message to messages list if present in request."""
        if "system" not in anthropic_message_request:
            return
        system_content = anthropic_message_request["system"]
        if not system_content:
            return
        # Handle system as string or array of content blocks
        if isinstance(system_content, str):
            new_messages.insert(
                0,
                ChatCompletionSystemMessage(role="system", content=system_content),
            )
        elif isinstance(system_content, list):
            # Convert Anthropic system content blocks to OpenAI format
            openai_system_content: List[Dict[str, Any]] = []
            model_name = anthropic_message_request.get("model", "")
            for block in system_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_block: Dict[str, Any] = {
                        "type": "text",
                        "text": block.get("text", ""),
                    }
                    self._add_cache_control_if_applicable(block, text_block, model_name)
                    openai_system_content.append(text_block)
            if openai_system_content:
                new_messages.insert(
                    0,
                    ChatCompletionSystemMessage(role="system", content=openai_system_content),  # type: ignore
                )

    def translate_anthropic_to_openai(
        self, anthropic_message_request: AnthropicMessagesRequest
    ) -> Tuple[ChatCompletionRequest, Dict[str, str]]:
        """
        This is used by the beta Anthropic Adapter, for translating anthropic `/v1/messages` requests to the openai format.

        Returns:
            Tuple of (openai_request, tool_name_mapping)
            - tool_name_mapping maps truncated tool names back to original names
              for tools that exceeded OpenAI's 64-char limit
        """
        # Debug: Processing Anthropic message request
        new_messages: List[AllMessageValues] = []
        tool_name_mapping: Dict[str, str] = {}

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
            messages=messages_list,
            model=anthropic_message_request.get("model"),
        )
        ## ADD SYSTEM MESSAGE TO MESSAGES
        self._add_system_message_to_messages(new_messages, anthropic_message_request)

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
                new_kwargs["tools"], tool_name_mapping = self.translate_anthropic_tools_to_openai(
                    tools=cast(List[AllAnthropicToolsValues], tools),
                    model=new_kwargs.get("model"),
                )

        ## CONVERT THINKING
        if "thinking" in anthropic_message_request:
            thinking = anthropic_message_request["thinking"]
            if thinking:
                model = new_kwargs.get("model", "")
                if self.is_anthropic_claude_model(model):
                    new_kwargs["thinking"] = thinking  # type: ignore
                else:
                    reasoning_effort = self.translate_anthropic_thinking_to_reasoning_effort(
                        cast(Dict[str, Any], thinking)
                    )
                    if reasoning_effort:
                        new_kwargs["reasoning_effort"] = reasoning_effort

        ## CONVERT OUTPUT_FORMAT to RESPONSE_FORMAT
        if "output_format" in anthropic_message_request:
            output_format = anthropic_message_request["output_format"]
            if output_format:
                response_format = self.translate_anthropic_output_format_to_openai(
                    output_format=output_format
                )
                if response_format:
                    new_kwargs["response_format"] = response_format

        translatable_params = self.translatable_anthropic_params()
        for k, v in anthropic_message_request.items():
            if k not in translatable_params:  # pass remaining params as is
                new_kwargs[k] = v  # type: ignore

        return new_kwargs, tool_name_mapping

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

    def _translate_openai_content_to_anthropic(
        self,
        choices: List[Choices],
        tool_name_mapping: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        new_content: List[Dict[str, Any]] = []
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
                            ).model_dump()
                        )
                    elif thinking_block.get("type") == "redacted_thinking":
                        data_value = thinking_block.get("data", "")
                        new_content.append(
                            AnthropicResponseContentBlockRedactedThinking(
                                type="redacted_thinking",
                                data=str(data_value) if data_value is not None else "",
                            ).model_dump()
                        )
            # Handle reasoning_content when thinking_blocks is not present
            elif (
                hasattr(choice.message, "reasoning_content")
                and choice.message.reasoning_content
            ):
                new_content.append(
                    AnthropicResponseContentBlockThinking(
                        type="thinking",
                        thinking=str(choice.message.reasoning_content),
                        signature=None,
                    ).model_dump()
                )

            # Handle text content
            if choice.message.content is not None:
                new_content.append(
                    AnthropicResponseContentBlockText(
                        type="text", text=choice.message.content
                    ).model_dump()
                )
            # Handle tool calls (in parallel to text content)
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

                    # Restore original tool name if it was truncated
                    truncated_name = tool_call.function.name or ""
                    original_name = (
                        tool_name_mapping.get(truncated_name, truncated_name)
                        if tool_name_mapping
                        else truncated_name
                    )

                    tool_use_block = AnthropicResponseContentBlockToolUse(
                        type="tool_use",
                        id=tool_call.id,
                        name=original_name,
                        input=parse_tool_call_arguments(
                            tool_call.function.arguments,
                            tool_name=original_name,
                            context="Anthropic pass-through adapter",
                        ),
                    )
                    # Add provider_specific_fields if signature is present
                    if provider_specific_fields:
                        tool_use_block.provider_specific_fields = (
                            provider_specific_fields
                        )
                    new_content.append(tool_use_block.model_dump())

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
        self,
        response: ModelResponse,
        tool_name_mapping: Optional[Dict[str, str]] = None,
    ) -> AnthropicMessagesResponse:
        """
        Translate OpenAI response to Anthropic format.

        Args:
            response: The OpenAI ModelResponse
            tool_name_mapping: Optional mapping of truncated tool names to original names.
                              Used to restore original names for tools that exceeded
                              OpenAI's 64-char limit.
        """
        ## translate content block
        anthropic_content = self._translate_openai_content_to_anthropic(
            choices=response.choices,  # type: ignore
            tool_name_mapping=tool_name_mapping,
        )
        ## extract finish reason
        anthropic_finish_reason = self._translate_openai_finish_reason_to_anthropic(
            openai_finish_reason=response.choices[0].finish_reason  # type: ignore
        )
        # extract usage
        usage: Usage = getattr(response, "usage")
        uncached_input_tokens = usage.prompt_tokens or 0
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
            uncached_input_tokens -= cached_tokens
        
        anthropic_usage = AnthropicUsage(
            input_tokens=uncached_input_tokens,
            output_tokens=usage.completion_tokens or 0,
        )
        # Add cache tokens if available (for prompt caching support)
        if hasattr(usage, "_cache_creation_input_tokens") and usage._cache_creation_input_tokens > 0:
            anthropic_usage["cache_creation_input_tokens"] = usage._cache_creation_input_tokens
        if hasattr(usage, "_cache_read_input_tokens") and usage._cache_read_input_tokens > 0:
            anthropic_usage["cache_read_input_tokens"] = usage._cache_read_input_tokens

        translated_obj = AnthropicMessagesResponse(
            id=response.id,
            type="message",
            role="assistant",
            model=response.model or "unknown-model",
            stop_sequence=None,
            usage=anthropic_usage,  # type: ignore
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
            if (
                choice.delta.tool_calls is not None
                and len(choice.delta.tool_calls) > 0
                and choice.delta.tool_calls[0].function is not None
            ):
                return "tool_use", ToolUseBlock(
                    type="tool_use",
                    id=choice.delta.tool_calls[0].id or str(uuid.uuid4()),
                    name=choice.delta.tool_calls[0].function.name or "",
                    input={},  # type: ignore[typeddict-item]
                )
            elif choice.delta.content is not None and len(choice.delta.content) > 0:
                return "text", TextBlock(type="text", text="")
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
            if choice.delta.tool_calls is not None:
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
            # Handle reasoning_content when thinking_blocks is not present
            # This handles providers like OpenRouter that return reasoning_content
            elif isinstance(choice, StreamingChoices) and hasattr(
                choice.delta, "reasoning_content"
            ):
                if choice.delta.reasoning_content is not None:
                    reasoning_content += choice.delta.reasoning_content

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
                uncached_input_tokens = litellm_usage_chunk.prompt_tokens or 0
                if hasattr(litellm_usage_chunk, "prompt_tokens_details") and litellm_usage_chunk.prompt_tokens_details:
                    cached_tokens = getattr(litellm_usage_chunk.prompt_tokens_details, "cached_tokens", 0) or 0
                    uncached_input_tokens -= cached_tokens
                
                usage_delta = UsageDelta(
                    input_tokens=uncached_input_tokens,
                    output_tokens=litellm_usage_chunk.completion_tokens or 0,
                )
                # Add cache tokens if available (for prompt caching support)
                if hasattr(litellm_usage_chunk, "_cache_creation_input_tokens") and litellm_usage_chunk._cache_creation_input_tokens > 0:
                    usage_delta["cache_creation_input_tokens"] = litellm_usage_chunk._cache_creation_input_tokens
                if hasattr(litellm_usage_chunk, "_cache_read_input_tokens") and litellm_usage_chunk._cache_read_input_tokens > 0:
                    usage_delta["cache_read_input_tokens"] = litellm_usage_chunk._cache_read_input_tokens
            else:
                usage_delta = UsageDelta(input_tokens=0, output_tokens=0)
            return MessageBlockDelta(
                type="message_delta", delta=delta, usage=usage_delta  # type: ignore
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
