import copy
import hashlib
import json
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Final, Literal, TypeVar, cast

import litellm
from litellm.llms.anthropic.experimental_pass_through.utils import (
    is_reasoning_auto_summary_enabled,
    prompt_cache_key_from_user_id,
)

# OpenAI has a 64-character limit for function/tool names
# Anthropic does not have this limit, so we need to truncate long names
OPENAI_MAX_TOOL_NAME_LENGTH: Final = 64
TOOL_NAME_HASH_LENGTH: Final = 8
TOOL_NAME_PREFIX_LENGTH: Final = OPENAI_MAX_TOOL_NAME_LENGTH - TOOL_NAME_HASH_LENGTH - 1  # 55
PROVIDERS_PROXYING_AN_UNKNOWN_BACKEND: Final = frozenset({"litellm_proxy"})


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
    name_hash: Final = hashlib.sha256(name.encode()).hexdigest()[:TOOL_NAME_HASH_LENGTH]
    return f"{name[:TOOL_NAME_PREFIX_LENGTH]}_{name_hash}"


def create_tool_name_mapping(
    tools: list[dict[str, Any]],
) -> dict[str, str]:
    """
    Create a mapping of truncated tool names to original names.

    Args:
        tools: List of tool definitions with 'name' field

    Returns:
        Dict mapping truncated names to original names (only for truncated tools)
    """
    mapping: Final[dict[str, str]] = {}
    for tool in tools:
        original_name = tool.get("name", "")
        truncated_name = truncate_tool_name(original_name)
        if truncated_name != original_name:
            mapping[truncated_name] = original_name
    return mapping


from openai.types.chat.chat_completion_chunk import Choice as OpenAIStreamingChoice

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    parse_tool_call_arguments,
    reasoning_content_from_thinking_blocks,
    with_prompt_cache_breakpoint,
)
from litellm.litellm_core_utils.prompt_templates.factory import (
    THOUGHT_SIGNATURE_SEPARATOR,
)
from litellm.litellm_core_utils.reasoning_effort_utils import (
    reasoning_effort_from_thinking_budget,
)
from litellm.llms.anthropic.common_utils import normalize_anthropic_tool_use_id
from litellm.llms.anthropic.experimental_pass_through.context_management import (
    PolyfillResult,
)
from litellm.types.llms.anthropic import (
    ANTHROPIC_HOSTED_TOOLS,
    AllAnthropicPassThroughMessageValues,
    AllAnthropicToolsValues,
    AnthropicFinishReason,
    AnthropicMessagesRequest,
    AnthropicMessagesSystemMessageParam,
    AnthropicMessagesToolChoice,
    AnthropicResponseContentBlockRedactedThinking,
    AnthropicResponseContentBlockText,
    AnthropicResponseContentBlockThinking,
    AnthropicResponseContentBlockToolUse,
    AnthropicThinkingParam,
    AppliedEdit,
    ContentBlockDelta,
    ContentJsonBlockDelta,
    ContentTextBlockDelta,
    ContentThinkingBlockDelta,
    ContentThinkingSignatureBlockDelta,
    ContextManagementResponse,
    MessageBlockDelta,
    MessageDelta,
    ServerToolUsage,
    StreamingContentBlockDeltaType,
    UsageDelta,
    UsageIteration,
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

    def translate_completion_input_params(self, kwargs) -> ChatCompletionRequest | None:
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
        self, kwargs, *, custom_llm_provider: str | None = None
    ) -> tuple[ChatCompletionRequest | None, dict[str, str]]:
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
        model: Final = kwargs.pop("model")
        messages: Final = kwargs.pop("messages")
        if not model:
            raise ValueError("Bad Request: model is required for Anthropic Messages Request")
        if not messages:
            raise ValueError("Bad Request: messages is required for Anthropic Messages Request")

        #########################################################
        # Created Typed Request Body
        #########################################################
        request_body: Final = AnthropicMessagesRequest(model=model, messages=messages, **kwargs)

        (
            translated_body,
            tool_name_mapping,
        ) = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
            anthropic_message_request=request_body,
            custom_llm_provider=custom_llm_provider,
        )

        return translated_body, tool_name_mapping

    def translate_completion_output_params(
        self,
        response: ModelResponse,
        tool_name_mapping: dict[str, str] | None = None,
        polyfill_result: PolyfillResult | None = None,
    ) -> AnthropicMessagesResponse | None:
        """
        Translate OpenAI response to Anthropic format.

        Args:
            response: The OpenAI ModelResponse
            tool_name_mapping: Optional mapping of truncated tool names to original names.
                              Used to restore original names for tools that exceeded
                              OpenAI's 64-char limit.
            polyfill_result: PolyfillResult from context_management polyfill.
        """
        return LiteLLMAnthropicMessagesAdapter().translate_openai_response_to_anthropic(
            response=response,
            tool_name_mapping=tool_name_mapping,
            polyfill_result=polyfill_result,
        )

    def translate_completion_output_params_streaming(
        self,
        completion_stream: Any,
        model: str,
        tool_name_mapping: dict[str, str] | None = None,
        polyfill_result: PolyfillResult | None = None,
        is_async: bool = True,
    ) -> AsyncIterator[bytes] | Iterator[bytes] | None:
        """
        Translate OpenAI streaming response to Anthropic format.

        Args:
            completion_stream: The OpenAI streaming response
            model: The model name
            tool_name_mapping: Optional mapping of truncated tool names to original names.
            polyfill_result: PolyfillResult from context_management polyfill.
            is_async: When ``True`` (default, for back-compat with existing
                async callers) returns an ``AsyncIterator[bytes]``. When
                ``False`` returns a sync ``Iterator[bytes]`` so sync callers
                (e.g. ``litellm.anthropic.messages.create(stream=True)`` via
                the sync handler) don't get back an async iterator they
                can't iterate without an event loop.
        """
        applied_edits: Final = polyfill_result.applied_edits_for_response() if polyfill_result else None
        compaction_block: Final = polyfill_result.compaction_block if polyfill_result is not None else None
        iterations_usage: Final = polyfill_result.iterations_usage if polyfill_result is not None else None
        anthropic_wrapper: Final = AnthropicStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            tool_name_mapping=tool_name_mapping,
            applied_edits=applied_edits,
            compaction_block=compaction_block,
            iterations_usage=iterations_usage,
        )
        # Return the SSE-wrapped version for proper event formatting.
        if is_async:
            return anthropic_wrapper.async_anthropic_sse_wrapper()
        return anthropic_wrapper.anthropic_sse_wrapper()


_BlockT: Final = TypeVar("_BlockT", bound=Mapping[str, object])


class LiteLLMAnthropicMessagesAdapter:
    def __init__(self):
        pass

    ### FOR [BETA] `/v1/messages` endpoint support

    def _extract_signature_from_tool_call(self, tool_call: Any) -> str | None:
        """
        Extract signature from a tool call's provider_specific_fields.
        Only checks provider_specific_fields, not thinking blocks.
        """
        signature = None

        if hasattr(tool_call, "provider_specific_fields") and tool_call.provider_specific_fields:
            if "thought_signature" in tool_call.provider_specific_fields:
                signature = tool_call.provider_specific_fields["thought_signature"]
        elif hasattr(tool_call.function, "provider_specific_fields") and tool_call.function.provider_specific_fields:
            if "thought_signature" in tool_call.function.provider_specific_fields:
                signature = tool_call.function.provider_specific_fields["thought_signature"]

        return signature

    def _extract_signature_from_tool_use_content(self, content: dict[str, Any]) -> str | None:
        """
        Extract signature from a tool_use content block's provider_specific_fields.
        """
        provider_specific_fields: Final = content.get("provider_specific_fields", {})
        if provider_specific_fields:
            return provider_specific_fields.get("signature")
        return None

    def _add_cache_control_if_applicable(
        self,
        source: Any,
        target: Any,
        model: str | None,
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
        cache_control: Final = (
            source.get("cache_control") if isinstance(source, dict) else getattr(source, "cache_control", None)
        )
        if cache_control and model and (self.is_anthropic_claude_model(model) or self.is_bedrock_arn_model(model)):
            # TypedDict objects support dict operations at runtime
            # Use type ignore consistent with codebase pattern (see anthropic/chat/transformation.py:432)
            if isinstance(target, dict):
                target["cache_control"] = cache_control
            else:
                # Fallback for non-dict objects (shouldn't happen in practice)
                cast(dict[str, object], target)["cache_control"] = cache_control

    @staticmethod
    def _add_prompt_cache_breakpoint_if_present(source: object, target: _BlockT) -> _BlockT:
        if isinstance(source, dict) and "prompt_cache_breakpoint" in source:
            return with_prompt_cache_breakpoint(target, source["prompt_cache_breakpoint"])
        return target

    def translatable_anthropic_params(self) -> list[str]:
        """
        Which anthropic params, we need to translate to the openai format.
        """
        return [
            "messages",
            "metadata",
            "system",
            "tool_choice",
            "tools",
            "thinking",
            "output_format",
            "output_config",
            "stop_sequences",
        ]

    def _is_web_search_tool(self, tool: Mapping[str, object]) -> bool:
        """
        Check if a tool is an Anthropic web search tool.

        Anthropic web search tools have:
        - type starting with "web_search" (e.g., "web_search_20260209")
        - name = "web_search"

        Args:
            tool: Tool definition dict

        Returns:
            True if this is a web search tool
        """
        tool_type: Final = tool.get("type", "")
        tool_name: Final = tool.get("name", "")
        return (isinstance(tool_type, str) and tool_type.startswith("web_search")) or tool_name == "web_search"

    def translate_anthropic_messages_to_openai(
        self,
        messages: list[AllAnthropicPassThroughMessageValues],
        model: str | None = None,
    ) -> list:
        new_messages: Final[list[AllMessageValues]] = []
        for m in messages:
            user_message: ChatCompletionUserMessage | None = None
            tool_message_list: list[ChatCompletionToolMessage] = []
            new_user_content_list: list[ChatCompletionTextObject | ChatCompletionImageObject] = []
            if m["role"] == "system":
                system_message = self._translate_midturn_system_message_to_openai(m, model)
                if system_message is not None:
                    new_messages.append(system_message)
                continue
            ## USER MESSAGE ##
            if m["role"] == "user":
                ## translate user message
                message_content = m.get("content")
                if message_content and isinstance(message_content, str):
                    user_message = ChatCompletionUserMessage(role="user", content=message_content)
                elif message_content and isinstance(message_content, list):
                    for content in message_content:
                        if content.get("type") == "text":
                            text_obj = ChatCompletionTextObject(type="text", text=content.get("text", ""))
                            self._add_cache_control_if_applicable(content, text_obj, model)
                            new_user_content_list.append(
                                self._add_prompt_cache_breakpoint_if_present(content, text_obj)
                            )
                        elif content.get("type") == "image":
                            # Convert Anthropic image format to OpenAI format
                            source = content.get("source", {})
                            openai_image_url = self._translate_anthropic_image_to_openai(cast(dict, source))

                            if openai_image_url:
                                image_url_obj = ChatCompletionImageUrlObject(url=openai_image_url)
                                image_obj = ChatCompletionImageObject(type="image_url", image_url=image_url_obj)
                                self._add_cache_control_if_applicable(content, image_obj, model)
                                new_user_content_list.append(
                                    self._add_prompt_cache_breakpoint_if_present(content, image_obj)
                                )
                        elif content.get("type") == "document":
                            # Convert Anthropic document format (PDF, etc.) to OpenAI format
                            source = content.get("source", {})
                            openai_image_url = self._translate_anthropic_image_to_openai(cast(dict, source))

                            if openai_image_url:
                                image_url_obj = ChatCompletionImageUrlObject(url=openai_image_url)
                                doc_obj = ChatCompletionImageObject(type="image_url", image_url=image_url_obj)
                                self._add_cache_control_if_applicable(content, doc_obj, model)
                                new_user_content_list.append(doc_obj)
                        elif content.get("type") == "tool_result":
                            if "content" not in content:
                                tool_result = ChatCompletionToolMessage(
                                    role="tool",
                                    tool_call_id=content.get("tool_use_id", ""),
                                    content="",
                                )
                                self._add_cache_control_if_applicable(content, tool_result, model)
                                tool_message_list.append(tool_result)
                            elif isinstance(content.get("content"), str):
                                tool_result = ChatCompletionToolMessage(
                                    role="tool",
                                    tool_call_id=content.get("tool_use_id", ""),
                                    content=str(content.get("content", "")),
                                )
                                self._add_cache_control_if_applicable(content, tool_result, model)
                                tool_message_list.append(tool_result)
                            elif isinstance(content.get("content"), list):
                                # Combine all content items into a single tool message
                                # to avoid creating multiple tool_result blocks with the same ID
                                # (each tool_use must have exactly one tool_result)
                                content_items = list(content.get("content", []))

                                # Single-item text keeps the backward-compatible string format; a single
                                # image or document becomes a structured image_url part
                                if len(content_items) == 1:
                                    c = content_items[0]
                                    if isinstance(c, str):
                                        tool_result = ChatCompletionToolMessage(
                                            role="tool",
                                            tool_call_id=content.get("tool_use_id", ""),
                                            content=c,
                                        )
                                        self._add_cache_control_if_applicable(content, tool_result, model)
                                        tool_message_list.append(tool_result)
                                    elif isinstance(c, dict):
                                        if c.get("type") == "text":
                                            tool_result = ChatCompletionToolMessage(
                                                role="tool",
                                                tool_call_id=content.get("tool_use_id", ""),
                                                content=c.get("text", ""),
                                            )
                                            self._add_cache_control_if_applicable(content, tool_result, model)
                                            tool_message_list.append(tool_result)
                                        elif c.get("type") in ("image", "document"):
                                            image_part = self._tool_result_image_part(c.get("source"))
                                            tool_result = ChatCompletionToolMessage(
                                                role="tool",
                                                tool_call_id=content.get("tool_use_id", ""),
                                                content=[image_part]  # mutable-ok: content must be a json list
                                                if image_part
                                                else "",
                                            )
                                            self._add_cache_control_if_applicable(content, tool_result, model)
                                            tool_message_list.append(tool_result)
                                else:
                                    # For multiple content items, combine into a single tool message
                                    # with list content to preserve all items while having one tool_use_id
                                    combined_content_parts: list[
                                        ChatCompletionTextObject | ChatCompletionImageObject
                                    ] = []
                                    for c in content_items:
                                        if isinstance(c, str):
                                            combined_content_parts.append(ChatCompletionTextObject(type="text", text=c))
                                        elif isinstance(c, dict):
                                            if c.get("type") == "text":
                                                combined_content_parts.append(
                                                    ChatCompletionTextObject(
                                                        type="text",
                                                        text=c.get("text", ""),
                                                    )
                                                )
                                            elif c.get("type") in ("image", "document"):
                                                image_part = self._tool_result_image_part(c.get("source"))
                                                if image_part:
                                                    combined_content_parts.append(image_part)
                                    # Create a single tool message with combined content
                                    if combined_content_parts:
                                        tool_result = ChatCompletionToolMessage(
                                            role="tool",
                                            tool_call_id=content.get("tool_use_id", ""),
                                            content=combined_content_parts,
                                        )
                                        self._add_cache_control_if_applicable(content, tool_result, model)
                                        tool_message_list.append(tool_result)

            if len(tool_message_list) > 0:
                new_messages.extend(tool_message_list)

            if user_message is not None:
                new_messages.append(user_message)

            if len(new_user_content_list) > 0:
                new_messages.append({"role": "user", "content": new_user_content_list})

            ## ASSISTANT MESSAGE ##
            assistant_message_str: str | None = None
            assistant_content_list: list[dict[str, Any]] = []  # For content blocks with cache_control
            has_cache_control_in_text = False
            tool_calls: list[ChatCompletionAssistantToolCall] = []
            thinking_blocks: list[ChatCompletionThinkingBlock | ChatCompletionRedactedThinkingBlock] = []
            if m["role"] == "assistant":
                if isinstance(m.get("content"), str):
                    assistant_message_str = str(m.get("content", ""))
                elif isinstance(m.get("content"), list):
                    for content in m.get("content", []):
                        if isinstance(content, str):
                            assistant_message_str = str(content)
                        elif isinstance(content, dict):
                            if content.get("type") == "text":
                                text_block: dict[str, object] = {
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
                                signature = self._extract_signature_from_tool_use_content(
                                    cast(dict[str, object], content)
                                )

                                if signature:
                                    provider_specific_fields: dict[str, object] = (
                                        function_chunk.get("provider_specific_fields") or {}
                                    )
                                    provider_specific_fields["thought_signature"] = signature
                                    function_chunk["provider_specific_fields"] = provider_specific_fields

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
                                redacted_thinking_block = ChatCompletionRedactedThinkingBlock(
                                    type="redacted_thinking",
                                    data=content.get("data") or "",
                                    cache_control=content.get("cache_control", {}),
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
                    assistant_content = "".join(block.get("text", "") for block in assistant_content_list)
                else:
                    assistant_content = assistant_message_str

                assistant_message = ChatCompletionAssistantMessage(
                    role="assistant",
                    content=assistant_content,
                    thinking_blocks=(thinking_blocks if len(thinking_blocks) > 0 else None),
                )
                if len(tool_calls) > 0:
                    assistant_message["tool_calls"] = tool_calls
                if len(thinking_blocks) > 0:
                    assistant_message["thinking_blocks"] = thinking_blocks
                reasoning_content = reasoning_content_from_thinking_blocks(thinking_blocks)
                if reasoning_content:
                    assistant_message["reasoning_content"] = reasoning_content
                new_messages.append(assistant_message)

        return new_messages

    @staticmethod
    def translate_anthropic_thinking_to_reasoning_effort(
        thinking: AnthropicThinkingParam,
    ) -> str | None:
        """
        Translate Anthropic's thinking parameter to OpenAI's reasoning_effort.

        Anthropic thinking format: {'type': 'enabled'|'disabled', 'budget_tokens': int}
        OpenAI reasoning_effort: 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'default'

        ``budget_tokens`` is bucketed via the shared
        ``reasoning_effort_from_thinking_budget`` thresholds.
        """
        if not isinstance(thinking, dict):
            return None

        thinking_type: Final = thinking.get("type", "disabled")

        if thinking_type == "disabled":
            return "none"
        elif thinking_type == "enabled":
            return reasoning_effort_from_thinking_budget(thinking.get("budget_tokens", 0))
        elif thinking_type == "adaptive":
            # Adaptive thinking: effort is controlled by output_config.effort,
            # not budget_tokens. Return a default; caller should override with
            # output_config.effort when available.
            return "medium"

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
        model_lower: Final = model.lower()
        return "anthropic" in model_lower or "claude" in model_lower

    @staticmethod
    def is_bedrock_arn_model(model: str) -> bool:
        """
        Check if the model string is a Bedrock ARN, such as an Application
        Inference Profile (e.g. arn:aws:bedrock:us-east-1:123:application-inference-profile/id).

        These ARNs contain neither "anthropic" nor "claude", so is_anthropic_claude_model
        cannot identify them even though, on the /v1/messages endpoint, they point at Claude.
        Match ":bedrock:" in the ARN service field so another service's ARN that merely names
        bedrock in a resource (arn:aws:sagemaker:.../my-bedrock-endpoint) is not matched.
        """
        model_lower: Final = model.lower()
        return "arn:" in model_lower and ":bedrock:" in model_lower

    @staticmethod
    def translate_thinking_for_model(
        thinking: AnthropicThinkingParam,
        model: str,
    ) -> dict[str, object]:
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
        if LiteLLMAnthropicMessagesAdapter.is_anthropic_claude_model(
            model
        ) or LiteLLMAnthropicMessagesAdapter.is_bedrock_arn_model(model):
            return {"thinking": thinking}
        else:
            reasoning_effort: Final = LiteLLMAnthropicMessagesAdapter.translate_anthropic_thinking_to_reasoning_effort(
                thinking
            )
            if reasoning_effort:
                return {
                    "reasoning_effort": LiteLLMAnthropicMessagesAdapter._apply_reasoning_summary_wrapping(
                        reasoning_effort, thinking
                    )
                }
            return {}

    @staticmethod
    def _apply_reasoning_summary_wrapping(
        reasoning_effort: str,
        thinking: Mapping[str, object],
    ) -> Any:
        """
        Apply the reasoning_effort/summary wrapping rules shared by every
        thinking->reasoning_effort translation path.

        Disabled thinking always stays a plain string - there's no reasoning
        trace to summarize, and non-Claude providers (e.g. Fireworks) expect
        reasoning_effort as a plain string, not a summary dict.
        """
        thinking_type: Final = thinking.get("type") if isinstance(thinking, dict) else None
        if thinking_type == "disabled":
            return reasoning_effort

        summary: Final = thinking.get("summary") if isinstance(thinking, dict) else None
        if summary:
            return {"effort": reasoning_effort, "summary": summary}
        if is_reasoning_auto_summary_enabled():
            return {"effort": reasoning_effort, "summary": "detailed"}
        return reasoning_effort

    def translate_anthropic_tool_choice_to_openai(
        self, tool_choice: AnthropicMessagesToolChoice
    ) -> ChatCompletionToolChoiceValues:
        if tool_choice["type"] == "any":
            return "required"
        elif tool_choice["type"] == "auto":
            return "auto"
        elif tool_choice["type"] == "tool":
            # Truncate tool name if it exceeds OpenAI's 64-char limit
            original_name: Final = tool_choice.get("name", "")
            truncated_name: Final = truncate_tool_name(original_name)
            tc_function_param: Final = ChatCompletionToolChoiceFunctionParam(name=truncated_name)
            return ChatCompletionToolChoiceObjectParam(type="function", function=tc_function_param)
        elif tool_choice["type"] == "none":
            return "none"
        else:
            raise ValueError(f"Incompatible tool choice param submitted - {tool_choice}")

    def translate_anthropic_tools_to_openai(
        self, tools: list[AllAnthropicToolsValues], model: str | None = None
    ) -> tuple[list[ChatCompletionToolParam], dict[str, str]]:
        """
        Translate Anthropic tools to OpenAI format.

        Returns:
            Tuple of (translated_tools, tool_name_mapping)
            - tool_name_mapping maps truncated names back to original names
              for tools that exceeded OpenAI's 64-char limit
        """
        new_tools: Final[list[ChatCompletionToolParam]] = []
        tool_name_mapping: Final[dict[str, str]] = {}
        # "type" is the Anthropic tool type (e.g. "custom"); it must not be
        # merged into the OpenAI function `parameters` schema below, or it
        # overwrites the real parameters.type ("object") and the provider
        # rejects the request. See #30557.
        mapped_tool_params: Final = [
            "name",
            "input_schema",
            "description",
            "cache_control",
            "strict",
            "type",
        ]

        for idx, tool in enumerate(tools):
            # Check if this is an Anthropic-native tool that should be kept as-is
            tool_type = tool.get("type", "")
            if any(tool_type.startswith(t.value) for t in ANTHROPIC_HOSTED_TOOLS):
                # Keep Anthropic-native tools in their original format
                new_tools.append(tool)
                continue

            raw_name = tool.get("name")
            if raw_name is None or (isinstance(raw_name, str) and not str(raw_name).strip()):
                original_name = f"litellm_unnamed_tool_{idx}"
            else:
                original_name = str(raw_name)
            truncated_name = truncate_tool_name(original_name)

            # Store mapping if name was truncated
            if truncated_name != original_name:
                tool_name_mapping[truncated_name] = original_name

            function_chunk = ChatCompletionToolParamFunctionChunk(
                name=truncated_name,
            )
            if "input_schema" in tool:
                function_chunk["parameters"] = tool["input_schema"]
            if "description" in tool:
                function_chunk["description"] = tool["description"]
            if "strict" in tool:
                function_chunk["strict"] = bool(tool["strict"])

            for k, v in tool.items():
                if k not in mapped_tool_params:  # pass additional computer kwargs
                    function_chunk.setdefault("parameters", {}).update({k: v})
            tool_param = ChatCompletionToolParam(type="function", function=function_chunk)
            self._add_cache_control_if_applicable(tool, tool_param, model)
            new_tools.append(tool_param)

        return new_tools, tool_name_mapping

    def translate_anthropic_output_format_to_openai(self, output_format: Any) -> dict[str, object] | None:
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

        output_type: Final = output_format.get("type")
        if output_type != "json_schema":
            return None

        schema = output_format.get("schema")
        if not schema:
            return None

        # Deep copy to avoid mutating the original schema
        schema = copy.deepcopy(schema)
        # OpenAI strict mode requires additionalProperties: false on every object
        self._add_additional_properties_false(schema)

        # Convert to OpenAI response_format structure
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "schema": schema,
                "strict": True,
            },
        }

    @staticmethod
    def _add_additional_properties_false(schema: dict) -> None:
        """
        Recursively ensure object schemas comply with OpenAI strict mode.

        OpenAI's strict mode requires:
        1. 'additionalProperties': false at every object nesting level
        2. All property keys listed in 'required'
        """
        if not isinstance(schema, dict):
            return

        if schema.get("type") == "object" and "properties" in schema:
            schema["additionalProperties"] = False
            schema["required"] = list(schema["properties"].keys())
            for prop in schema["properties"].values():
                LiteLLMAnthropicMessagesAdapter._add_additional_properties_false(prop)

        # Handle array items
        if "items" in schema:
            LiteLLMAnthropicMessagesAdapter._add_additional_properties_false(schema["items"])

        # Handle anyOf/oneOf/allOf
        for key in ("anyOf", "oneOf", "allOf"):
            if key in schema:
                for sub_schema in schema[key]:
                    LiteLLMAnthropicMessagesAdapter._add_additional_properties_false(sub_schema)

        # Handle $defs / definitions
        for key in ("$defs", "definitions"):
            if key in schema:
                for def_schema in schema[key].values():
                    LiteLLMAnthropicMessagesAdapter._add_additional_properties_false(def_schema)

    def _translate_midturn_system_message_to_openai(
        self,
        message: AnthropicMessagesSystemMessageParam,
        model: str | None,
    ) -> ChatCompletionSystemMessage | None:
        """Translate an in-sequence system entry without changing its role or position."""
        content: Final = message.get("content")
        if isinstance(content, str):
            return ChatCompletionSystemMessage(role="system", content=content) if content else None
        if not isinstance(content, list):
            return None
        text_parts: Final[list[ChatCompletionTextObject]] = []  # mutable-ok: API message payload
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":  # pyright: ignore[reportUnnecessaryIsInstance]  # untrusted client payload
                continue
            text = block.get("text")
            if not text:
                continue
            text_obj = ChatCompletionTextObject(type="text", text=text)
            self._add_cache_control_if_applicable(block, text_obj, model)
            text_parts.append(self._add_prompt_cache_breakpoint_if_present(block, text_obj))
        return ChatCompletionSystemMessage(role="system", content=text_parts) if text_parts else None

    def _add_system_message_to_messages(
        self,
        new_messages: list[AllMessageValues],
        anthropic_message_request: AnthropicMessagesRequest,
    ) -> None:
        """Add system message to messages list if present in request."""
        if "system" not in anthropic_message_request:
            return
        system_content: Final = anthropic_message_request["system"]
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
            openai_system_content: Final[list[dict[str, Any]]] = []
            model_name: Final = anthropic_message_request.get("model", "")
            for block in system_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_block: dict[str, object] = {
                        "type": "text",
                        "text": block.get("text", ""),
                    }
                    self._add_cache_control_if_applicable(block, text_block, model_name)
                    openai_system_content.append(self._add_prompt_cache_breakpoint_if_present(block, text_block))
            if openai_system_content:
                new_messages.insert(
                    0,
                    ChatCompletionSystemMessage(role="system", content=openai_system_content),
                )

    @staticmethod
    def _supports_prompt_cache_key(model: str | None, custom_llm_provider: str | None) -> bool:
        if not model or not custom_llm_provider:
            return False
        if custom_llm_provider in PROVIDERS_PROXYING_AN_UNKNOWN_BACKEND:
            return False
        supported_params: Final = litellm.get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        return "prompt_cache_key" in (supported_params or ())

    def _translate_metadata_to_openai(
        self,
        anthropic_message_request: AnthropicMessagesRequest,
        new_kwargs: ChatCompletionRequest,
        *,
        custom_llm_provider: str | None = None,
    ) -> None:
        """Translate metadata fields from Anthropic request to OpenAI request."""
        if "metadata" in anthropic_message_request:
            metadata: Final = anthropic_message_request["metadata"]
            if metadata and "user_id" in metadata:
                new_kwargs["user"] = metadata["user_id"]
                prompt_cache_key: Final = prompt_cache_key_from_user_id(metadata["user_id"])
                if prompt_cache_key is not None and self._supports_prompt_cache_key(
                    anthropic_message_request.get("model"), custom_llm_provider
                ):
                    new_kwargs["prompt_cache_key"] = prompt_cache_key

        if "litellm_metadata" in anthropic_message_request:
            # metadata will be passed to litellm.acompletion(), it's a litellm_param
            new_kwargs["metadata"] = anthropic_message_request.pop("litellm_metadata")

    def _translate_tool_choice_to_openai(
        self,
        anthropic_message_request: AnthropicMessagesRequest,
        new_kwargs: ChatCompletionRequest,
    ) -> None:
        """Translate Anthropic tool_choice to OpenAI format."""
        if "tool_choice" not in anthropic_message_request:
            return
        tool_choice: Final = anthropic_message_request["tool_choice"]
        if not tool_choice:
            return
        new_kwargs["tool_choice"] = self.translate_anthropic_tool_choice_to_openai(
            tool_choice=cast(AnthropicMessagesToolChoice, tool_choice)
        )

    def _translate_stop_sequences_to_openai(
        self,
        anthropic_message_request: AnthropicMessagesRequest,
        new_kwargs: ChatCompletionRequest,
    ) -> None:
        if "stop_sequences" not in anthropic_message_request:
            return
        stop_sequences: Final = anthropic_message_request["stop_sequences"]
        if not stop_sequences:
            return
        new_kwargs["stop"] = stop_sequences

    def _translate_tools_to_openai(
        self,
        anthropic_message_request: AnthropicMessagesRequest,
        new_kwargs: ChatCompletionRequest,
    ) -> dict[str, str]:
        """Translate tools and extract web_search_options when needed."""
        if "tools" not in anthropic_message_request:
            return {}

        tools: Final = anthropic_message_request["tools"]
        if not tools:
            return {}

        web_search_tools: Final[list[AllAnthropicToolsValues]] = []
        regular_tools: Final[list[AllAnthropicToolsValues]] = []
        for tool in tools:
            cast_tool = cast(dict[str, object], tool)
            if self._is_web_search_tool(cast_tool):
                web_search_tools.append(cast(AllAnthropicToolsValues, tool))
            else:
                regular_tools.append(cast(AllAnthropicToolsValues, tool))

        if web_search_tools:
            new_kwargs["web_search_options"] = {}

        if not regular_tools:
            return {}

        translated_tools, tool_name_mapping = self.translate_anthropic_tools_to_openai(
            tools=regular_tools,
            model=new_kwargs.get("model"),
        )
        new_kwargs["tools"] = translated_tools
        return tool_name_mapping

    def _translate_thinking_to_openai(
        self,
        anthropic_message_request: AnthropicMessagesRequest,
        new_kwargs: ChatCompletionRequest,
    ) -> None:
        """Translate Anthropic thinking to either thinking or reasoning_effort."""
        if "thinking" not in anthropic_message_request:
            return

        thinking: Final = anthropic_message_request["thinking"]
        if not thinking:
            return

        model: Final = new_kwargs.get("model", "")
        if self.is_anthropic_claude_model(model) or self.is_bedrock_arn_model(model):
            new_kwargs["thinking"] = thinking
            # Adaptive thinking without its effort tier makes Bedrock Converse
            # return zero reasoning blocks, so forward output_config (minus
            # `format`, already translated to response_format) for Bedrock
            # targets only: other bridged providers reject the raw param, and
            # get_llm_provider strips the `bedrock/` prefix before this runs.
            if model.startswith(("bedrock/", "converse/", "invoke/")) or self.is_bedrock_arn_model(model):
                claude_output_config: Final = anthropic_message_request.get("output_config")
                if isinstance(claude_output_config, dict):
                    effort_config: Final = {k: v for k, v in claude_output_config.items() if k != "format"}
                    if effort_config:
                        new_kwargs["output_config"] = effort_config  # rebind-ok: out-param store like thinking above
            return

        reasoning_effort = self.translate_anthropic_thinking_to_reasoning_effort(cast(AnthropicThinkingParam, thinking))
        if not reasoning_effort:
            return

        thinking_type: Final = thinking.get("type") if isinstance(thinking, dict) else None

        # For adaptive thinking, override with output_config.effort if available
        if thinking_type == "adaptive":
            output_config: Final = anthropic_message_request.get("output_config")
            if isinstance(output_config, dict) and output_config.get("effort"):
                reasoning_effort = output_config["effort"]

        new_kwargs["reasoning_effort"] = self._apply_reasoning_summary_wrapping(
            reasoning_effort, cast(dict[str, object], thinking)
        )

    def _translate_output_format_to_openai(
        self,
        anthropic_message_request: AnthropicMessagesRequest,
        new_kwargs: ChatCompletionRequest,
    ) -> None:
        """Translate Anthropic structured-output config to OpenAI ``response_format``.

        Accepts either the legacy top-level ``output_format`` field OR the
        newer ``output_config.format`` (sub-key on ``output_config``) so that
        both shapes flow through to non-Anthropic backends as
        ``response_format``. Without the ``output_config.format`` branch,
        callers using the new Anthropic Structured Outputs API would have
        their schema silently dropped on the adapter path — only the legacy
        top-level ``output_format`` was being mapped.

        ``output_format`` takes precedence when both are provided.
        """
        output_format: object = anthropic_message_request.get("output_format")
        if not output_format:
            output_config: Final = anthropic_message_request.get("output_config")
            if isinstance(output_config, dict):
                output_format = output_config.get("format")
        if not output_format:
            return
        response_format: Final = self.translate_anthropic_output_format_to_openai(output_format=output_format)
        if response_format:
            new_kwargs["response_format"] = response_format

    def _copy_untranslated_anthropic_params(
        self,
        anthropic_message_request: AnthropicMessagesRequest,
        new_kwargs: ChatCompletionRequest,
    ) -> None:
        """Copy through anthropic params that do not require translation."""
        translatable_params: Final = self.translatable_anthropic_params()
        for k, v in anthropic_message_request.items():
            if k not in translatable_params:  # pass remaining params as is
                new_kwargs[k] = v

    def translate_anthropic_to_openai(
        self,
        anthropic_message_request: AnthropicMessagesRequest,
        *,
        custom_llm_provider: str | None = None,
    ) -> tuple[ChatCompletionRequest, dict[str, str]]:
        """
        This is used by the beta Anthropic Adapter, for translating anthropic `/v1/messages` requests to the openai format.

        Returns:
            Tuple of (openai_request, tool_name_mapping)
            - tool_name_mapping maps truncated tool names back to original names
              for tools that exceeded OpenAI's 64-char limit
        """
        # Debug: Processing Anthropic message request
        new_messages: list[AllMessageValues] = []
        tool_name_mapping: dict[str, str] = {}

        ## CONVERT ANTHROPIC MESSAGES TO OPENAI
        messages_list: Final[list[AllAnthropicPassThroughMessageValues]] = cast(
            list[AllAnthropicPassThroughMessageValues],
            anthropic_message_request["messages"],
        )
        new_messages = self.translate_anthropic_messages_to_openai(
            messages=messages_list,
            model=anthropic_message_request.get("model"),
        )
        ## ADD SYSTEM MESSAGE TO MESSAGES
        self._add_system_message_to_messages(new_messages, anthropic_message_request)

        new_kwargs: Final[ChatCompletionRequest] = {
            "model": anthropic_message_request["model"],
            "messages": new_messages,
        }
        ## CONVERT METADATA (user_id + litellm metadata)
        self._translate_metadata_to_openai(
            anthropic_message_request=anthropic_message_request,
            new_kwargs=new_kwargs,
            custom_llm_provider=custom_llm_provider,
        )
        ## CONVERT TOOL CHOICE
        self._translate_tool_choice_to_openai(
            anthropic_message_request=anthropic_message_request,
            new_kwargs=new_kwargs,
        )
        ## CONVERT TOOLS
        tool_name_mapping = self._translate_tools_to_openai(
            anthropic_message_request=anthropic_message_request,
            new_kwargs=new_kwargs,
        )
        ## CONVERT THINKING
        self._translate_thinking_to_openai(
            anthropic_message_request=anthropic_message_request,
            new_kwargs=new_kwargs,
        )
        ## CONVERT STOP_SEQUENCES
        self._translate_stop_sequences_to_openai(
            anthropic_message_request=anthropic_message_request,
            new_kwargs=new_kwargs,
        )
        ## CONVERT OUTPUT_FORMAT to RESPONSE_FORMAT
        self._translate_output_format_to_openai(
            anthropic_message_request=anthropic_message_request,
            new_kwargs=new_kwargs,
        )
        self._copy_untranslated_anthropic_params(
            anthropic_message_request=anthropic_message_request,
            new_kwargs=new_kwargs,
        )

        return new_kwargs, tool_name_mapping

    def _translate_anthropic_image_to_openai(self, image_source: Mapping[str, str]) -> str | None:
        """
        Translate Anthropic image source format to OpenAI-compatible image URL.

        Anthropic supports two image source formats:
        1. Base64: {"type": "base64", "media_type": "image/jpeg", "data": "..."}
        2. URL: {"type": "url", "url": "https://..."}

        Returns the properly formatted image URL string, or None if invalid format.
        """
        if not isinstance(image_source, dict):
            return None

        source_type: Final = image_source.get("type")

        if source_type == "base64":
            # Base64 image format
            media_type: Final = image_source.get("media_type", "image/jpeg")
            image_data: Final = image_source.get("data", "")
            if image_data:
                return f"data:{media_type};base64,{image_data}"
        elif source_type == "url":
            # URL-referenced image format
            return image_source.get("url", "")

        return None

    def _tool_result_image_part(self, image_source: object) -> ChatCompletionImageObject | None:
        if not isinstance(image_source, dict):
            return None
        openai_image_url = self._translate_anthropic_image_to_openai(image_source)
        if not openai_image_url:
            return None
        return ChatCompletionImageObject(type="image_url", image_url=ChatCompletionImageUrlObject(url=openai_image_url))

    def _translate_openai_content_to_anthropic(
        self,
        choices: list[Choices],
        tool_name_mapping: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        new_content: Final[list[dict[str, Any]]] = []
        for choice in choices:
            # Handle thinking blocks first
            if hasattr(choice.message, "thinking_blocks") and choice.message.thinking_blocks:
                for thinking_block in choice.message.thinking_blocks:
                    if thinking_block.get("type") == "thinking":
                        thinking_value = thinking_block.get("thinking", "")
                        signature_value = thinking_block.get("signature", "")
                        new_content.append(
                            AnthropicResponseContentBlockThinking(
                                type="thinking",
                                thinking=(str(thinking_value) if thinking_value is not None else ""),
                                signature=(str(signature_value) if signature_value is not None else None),
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
            elif hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
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
                    AnthropicResponseContentBlockText(type="text", text=choice.message.content).model_dump()
                )
            # Handle tool calls (in parallel to text content)
            if choice.message.tool_calls is not None and len(choice.message.tool_calls) > 0:
                for tool_call in choice.message.tool_calls:
                    # Extract signature from provider_specific_fields only
                    signature = self._extract_signature_from_tool_call(tool_call)

                    provider_specific_fields = {}
                    if signature:
                        provider_specific_fields["signature"] = signature

                    # Restore original tool name if it was truncated
                    truncated_name = tool_call.function.name or ""
                    original_name = (
                        tool_name_mapping.get(truncated_name, truncated_name) if tool_name_mapping else truncated_name
                    )

                    # Strip Gemini thought-signature suffix and normalize id chars
                    # (e.g. ``functions.Bash:0`` from cross-provider clients).
                    raw_id = tool_call.id or ""
                    tool_use_block = AnthropicResponseContentBlockToolUse(
                        type="tool_use",
                        id=normalize_anthropic_tool_use_id(raw_id),
                        name=original_name,
                        input=parse_tool_call_arguments(
                            tool_call.function.arguments,
                            tool_name=original_name,
                            context="Anthropic pass-through adapter",
                        ),
                    )
                    # Add provider_specific_fields if signature is present
                    if provider_specific_fields:
                        tool_use_block.provider_specific_fields = provider_specific_fields
                    new_content.append(tool_use_block.model_dump())

        return new_content

    def _translate_openai_finish_reason_to_anthropic(self, openai_finish_reason: str) -> AnthropicFinishReason:
        if openai_finish_reason == "stop":
            return "end_turn"
        elif openai_finish_reason == "length":
            return "max_tokens"
        elif openai_finish_reason == "tool_calls":
            return "tool_use"
        return "end_turn"

    @staticmethod
    def _positive_int(value: object) -> int:
        if isinstance(value, bool):
            return 0
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, float) and value.is_integer() and value > 0:
            return int(value)
        return 0

    @classmethod
    def _first_positive_usage_value(cls, usage: Usage, field_names: tuple[str, ...]) -> int:
        for field_name in field_names:
            value = cls._positive_int(getattr(usage, field_name, None))
            if value > 0:
                return value
        return 0

    @classmethod
    def _first_positive_prompt_tokens_detail_value(cls, usage: Usage, field_names: tuple[str, ...]) -> int:
        prompt_tokens_details: Final = getattr(usage, "prompt_tokens_details", None)
        if prompt_tokens_details is None:
            return 0

        for field_name in field_names:
            if isinstance(prompt_tokens_details, dict):
                value = cls._positive_int(prompt_tokens_details.get(field_name))
            else:
                value = cls._positive_int(getattr(prompt_tokens_details, field_name, None))
            if value > 0:
                return value
        return 0

    @classmethod
    def _get_cache_read_input_tokens(cls, usage: Usage) -> int:
        explicit_value = cls._first_positive_usage_value(usage, ("cache_read_input_tokens", "_cache_read_input_tokens"))
        if explicit_value > 0:
            return explicit_value
        return cls._first_positive_prompt_tokens_detail_value(usage, ("cached_tokens",))

    @classmethod
    def _get_cache_creation_input_tokens(cls, usage: Usage) -> int:
        explicit_value: Final = cls._first_positive_usage_value(
            usage, ("cache_creation_input_tokens", "_cache_creation_input_tokens")
        )
        if explicit_value > 0:
            return explicit_value
        return cls._first_positive_prompt_tokens_detail_value(usage, ("cache_creation_tokens", "cache_write_tokens"))

    @classmethod
    def _get_web_search_request_count(cls, usage: Usage) -> int:
        from litellm.litellm_core_utils.llm_cost_calc.utils import (
            get_web_search_requests_from_usage,
        )

        from_server_tool_use: Final = cls._positive_int(get_web_search_requests_from_usage(usage))
        if from_server_tool_use > 0:
            return from_server_tool_use
        return cls._first_positive_prompt_tokens_detail_value(usage, ("web_search_requests",))

    @classmethod
    def _translate_openai_usage_to_anthropic_usage_delta(cls, usage: Usage) -> UsageDelta:
        cache_read_input_tokens: Final = cls._get_cache_read_input_tokens(usage)
        cache_creation_input_tokens: Final = cls._get_cache_creation_input_tokens(usage)
        web_search_requests: Final = cls._get_web_search_request_count(usage)
        input_tokens: Final = max(
            (usage.prompt_tokens or 0) - cache_read_input_tokens - cache_creation_input_tokens,
            0,
        )

        usage_delta: Final = UsageDelta(
            input_tokens=input_tokens,
            output_tokens=usage.completion_tokens or 0,
        )
        if cache_creation_input_tokens > 0:
            usage_delta["cache_creation_input_tokens"] = cache_creation_input_tokens
        if cache_read_input_tokens > 0:
            usage_delta["cache_read_input_tokens"] = cache_read_input_tokens
        if web_search_requests > 0:
            return UsageDelta(
                **usage_delta,
                server_tool_use=ServerToolUsage(web_search_requests=web_search_requests),
            )
        return usage_delta

    @classmethod
    def _translate_openai_usage_to_anthropic_usage(cls, usage: Usage) -> AnthropicUsage:
        return cast(
            AnthropicUsage,
            cls._translate_openai_usage_to_anthropic_usage_delta(usage),
        )

    def translate_openai_response_to_anthropic(
        self,
        response: ModelResponse,
        tool_name_mapping: dict[str, str] | None = None,
        polyfill_result: PolyfillResult | None = None,
    ) -> AnthropicMessagesResponse:
        """
        Translate OpenAI response to Anthropic format.

        Args:
            response: The OpenAI ModelResponse
            tool_name_mapping: Optional mapping of truncated tool names to original names.
                              Used to restore original names for tools that exceeded
                              OpenAI's 64-char limit.
            polyfill_result: PolyfillResult from context_management polyfill.
        """
        ## translate content block
        anthropic_content: Final = self._translate_openai_content_to_anthropic(
            choices=response.choices,
            tool_name_mapping=tool_name_mapping,
        )

        if polyfill_result is not None and polyfill_result.compaction_block is not None:
            anthropic_content.insert(0, polyfill_result.compaction_block)

        ## extract finish reason
        anthropic_finish_reason: Final = self._translate_openai_finish_reason_to_anthropic(
            openai_finish_reason=response.choices[0].finish_reason
        )
        # extract usage
        usage: Final[Usage] = getattr(response, "usage")
        anthropic_usage: Final = self._translate_openai_usage_to_anthropic_usage(usage)

        if polyfill_result is not None and polyfill_result.iterations_usage is not None:
            message_iteration: Final[UsageIteration] = {
                "type": "message",
                "input_tokens": anthropic_usage["input_tokens"],
                "output_tokens": usage.completion_tokens or 0,
            }
            anthropic_usage["iterations"] = list(polyfill_result.iterations_usage) + [message_iteration]

        translated_obj: Final = AnthropicMessagesResponse(
            id=response.id,
            type="message",
            role="assistant",
            model=response.model or "unknown-model",
            stop_sequence=None,
            usage=anthropic_usage,
            content=anthropic_content,
            stop_reason=anthropic_finish_reason,
        )

        applied_edits: Final = polyfill_result.applied_edits_for_response() if polyfill_result else None
        if applied_edits:
            translated_obj["context_management"] = ContextManagementResponse(applied_edits=list(applied_edits))

        return translated_obj

    def _translate_streaming_openai_chunk_to_anthropic_content_block(
        self, choices: list[OpenAIStreamingChoice | StreamingChoices]
    ) -> tuple[
        Literal["text", "tool_use", "thinking"],
        "ContentBlockContentBlockDict",
    ]:
        from litellm._uuid import uuid
        from litellm.types.llms.anthropic import TextBlock

        for choice in choices:
            if (
                choice.delta.tool_calls is not None
                and len(choice.delta.tool_calls) > 0
                and choice.delta.tool_calls[0].function is not None
            ):
                raw_id = choice.delta.tool_calls[0].id or str(uuid.uuid4())
                tool_name = choice.delta.tool_calls[0].function.name or ""
                thought_sig: str | None = None
                if THOUGHT_SIGNATURE_SEPARATOR in raw_id:
                    parts = raw_id.split(THOUGHT_SIGNATURE_SEPARATOR, 1)
                    thought_sig = parts[1] if len(parts) > 1 else None
                tool_block: dict[str, object] = {
                    "type": "tool_use",
                    "id": normalize_anthropic_tool_use_id(raw_id),
                    "name": tool_name,
                    "input": {},
                }
                if thought_sig:
                    tool_block["provider_specific_fields"] = {
                        "signature": thought_sig,
                    }
                return "tool_use", cast("ContentBlockContentBlockDict", tool_block)
            elif choice.delta.content is not None and len(choice.delta.content) > 0:
                return "text", TextBlock(type="text", text="")
            elif isinstance(choice, StreamingChoices) and hasattr(choice.delta, "thinking_blocks"):
                thinking_blocks = choice.delta.thinking_blocks or []
                if len(thinking_blocks) > 0:
                    thinking_block = thinking_blocks[0]
                    if thinking_block["type"] == "thinking":
                        thinking = thinking_block.get("thinking") or ""
                        signature = thinking_block.get("signature") or ""

                        assert isinstance(thinking, str)
                        assert isinstance(signature, str)

                        return "thinking", ChatCompletionThinkingBlock(
                            type="thinking", thinking=thinking, signature=signature
                        )
            # OpenAI-compatible reasoning backends (e.g. vLLM/SGLang reasoning
            # parsers) populate ``reasoning_content`` without ``thinking_blocks``.
            # ``Delta`` deletes the ``thinking_blocks`` attribute when unset, so the
            # branch above is skipped entirely; open a ``thinking`` block here so the
            # matching ``thinking_delta`` stream is not emitted into a text block.
            elif isinstance(choice, StreamingChoices) and getattr(choice.delta, "reasoning_content", None):
                return "thinking", ChatCompletionThinkingBlock(type="thinking", thinking="", signature="")

        return "text", TextBlock(type="text", text="")

    def _translate_streaming_openai_chunk_to_anthropic(
        self, choices: list[OpenAIStreamingChoice | StreamingChoices]
    ) -> tuple[
        StreamingContentBlockDeltaType,
        ContentTextBlockDelta | ContentJsonBlockDelta | ContentThinkingBlockDelta | ContentThinkingSignatureBlockDelta,
    ]:
        text: str = ""
        reasoning_content: str = ""
        reasoning_signature: str = ""
        partial_json: str | None = None
        for choice in choices:
            if choice.delta.content is not None and len(choice.delta.content) > 0:
                text += choice.delta.content
            if choice.delta.tool_calls:
                partial_json = ""
                for tool in choice.delta.tool_calls:
                    if tool.function is not None and tool.function.arguments is not None:
                        partial_json = (partial_json or "") + tool.function.arguments
            elif isinstance(choice, StreamingChoices) and hasattr(choice.delta, "thinking_blocks"):
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
            elif isinstance(choice, StreamingChoices) and hasattr(choice.delta, "reasoning_content"):
                if choice.delta.reasoning_content is not None:
                    reasoning_content += choice.delta.reasoning_content

        if partial_json is not None:
            return "input_json_delta", ContentJsonBlockDelta(type="input_json_delta", partial_json=partial_json)
        elif reasoning_signature:
            return "signature_delta", ContentThinkingSignatureBlockDelta(
                type="signature_delta", signature=reasoning_signature
            )
        elif reasoning_content:
            return "thinking_delta", ContentThinkingBlockDelta(type="thinking_delta", thinking=reasoning_content)
        else:
            return "text_delta", ContentTextBlockDelta(type="text_delta", text=text)

    def translate_streaming_openai_response_to_anthropic(
        self,
        response: ModelResponse,
        current_content_block_index: int,
        applied_edits: list[AppliedEdit] | None = None,
    ) -> ContentBlockDelta | MessageBlockDelta:
        ## base case - final chunk w/ finish reason
        if response.choices[0].finish_reason is not None:
            delta: Final = MessageDelta(
                stop_reason=self._translate_openai_finish_reason_to_anthropic(response.choices[0].finish_reason),
            )
            if getattr(response, "usage", None) is not None:
                litellm_usage_chunk: Usage | None = response.usage
            elif hasattr(response, "_hidden_params") and "usage" in response._hidden_params:
                litellm_usage_chunk = response._hidden_params["usage"]
            else:
                litellm_usage_chunk = None
            if litellm_usage_chunk is not None:
                usage_delta = self._translate_openai_usage_to_anthropic_usage_delta(litellm_usage_chunk)
            else:
                usage_delta = UsageDelta(input_tokens=0, output_tokens=0)
            message_block: Final = MessageBlockDelta(
                type="message_delta",
                delta=delta,
                usage=usage_delta,
            )
            if applied_edits:
                message_block["context_management"] = ContextManagementResponse(applied_edits=list(applied_edits))
            return message_block
        (
            type_of_content,
            content_block_delta,
        ) = self._translate_streaming_openai_chunk_to_anthropic(choices=response.choices)
        return ContentBlockDelta(
            type="content_block_delta",
            index=current_content_block_index,
            delta=content_block_delta,
        )
