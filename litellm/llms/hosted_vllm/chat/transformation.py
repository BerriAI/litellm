"""
Translate from OpenAI's `/v1/chat/completions` to VLLM's `/v1/chat/completions`
"""

from typing import Any, Coroutine, Dict, List, Literal, Optional, Tuple, Union, cast, overload

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    _get_image_mime_type_from_url,
)
from litellm.litellm_core_utils.prompt_templates.factory import _parse_mime_type
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionFileObject,
    ChatCompletionToolMessage,
    ChatCompletionVideoObject,
    ChatCompletionVideoUrlObject,
)

from ....utils import _remove_additional_properties, _remove_strict_from_schema
from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class HostedVLLMChatConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> List[str]:
        params = super().get_supported_openai_params(model)
        params.extend(["reasoning_effort", "thinking"])
        return params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        _tools = non_default_params.pop("tools", None)
        if _tools is not None:
            # remove 'additionalProperties' from tools
            _tools = _remove_additional_properties(_tools)
            # remove 'strict' from tools
            _tools = _remove_strict_from_schema(_tools)
        if _tools is not None:
            non_default_params["tools"] = _tools

        # Handle thinking parameter - convert Anthropic-style to OpenAI-style reasoning_effort
        # vLLM is OpenAI-compatible, so it understands reasoning_effort, not thinking
        # Reference: https://github.com/BerriAI/litellm/issues/19761
        thinking = non_default_params.pop("thinking", None)
        if thinking is not None and isinstance(thinking, dict):
            if thinking.get("type") == "enabled":
                # Only convert if reasoning_effort not already set
                if "reasoning_effort" not in non_default_params:
                    budget_tokens = thinking.get("budget_tokens", 0)
                    # Map budget_tokens to reasoning_effort level
                    # Same logic as Anthropic adapter (translate_anthropic_thinking_to_reasoning_effort)
                    if budget_tokens >= 10000:
                        non_default_params["reasoning_effort"] = "high"
                    elif budget_tokens >= 5000:
                        non_default_params["reasoning_effort"] = "medium"
                    elif budget_tokens >= 2000:
                        non_default_params["reasoning_effort"] = "low"
                    else:
                        non_default_params["reasoning_effort"] = "minimal"

        return super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("HOSTED_VLLM_API_BASE")  # type: ignore
        dynamic_api_key = (
            api_key or get_secret_str("HOSTED_VLLM_API_KEY") or "fake-api-key"
        )  # vllm does not require an api key
        return api_base, dynamic_api_key

    def _is_video_file(self, content_item: ChatCompletionFileObject) -> bool:
        """
        Check if the file is a video

        - format: video/<extension>
        - file_data: base64 encoded video data
        - file_id: infer mp4 from extension
        """
        file = content_item.get("file", {})
        format = file.get("format")
        file_data = file.get("file_data")
        file_id = file.get("file_id")
        if content_item.get("type") != "file":
            return False
        if format and format.startswith("video/"):
            return True
        elif file_data:
            mime_type = _parse_mime_type(file_data)
            if mime_type and mime_type.startswith("video/"):
                return True
        elif file_id:
            mime_type = _get_image_mime_type_from_url(file_id)
            if mime_type and mime_type.startswith("video/"):
                return True
        return False

    def _convert_file_to_video_url(
        self, content_item: ChatCompletionFileObject
    ) -> ChatCompletionVideoObject:
        file = content_item.get("file", {})
        file_id = file.get("file_id")
        file_data = file.get("file_data")

        if file_id:
            return ChatCompletionVideoObject(
                type="video_url", video_url=ChatCompletionVideoUrlObject(url=file_id)
            )
        elif file_data:
            return ChatCompletionVideoObject(
                type="video_url", video_url=ChatCompletionVideoUrlObject(url=file_data)
            )
        raise ValueError("file_id or file_data is required")

    @staticmethod
    def _extract_tool_result_content(content: Any) -> str:
        """
        Extract text content from an Anthropic tool_result content field.

        The content field can be:
        - a string
        - a list of content blocks (each with type "text" and a "text" field)
        - missing/None (empty string)
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return " ".join(text_parts) if text_parts else ""
        return str(content)

    @overload
    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[Any, Any, List[AllMessageValues]]:
        ...

    @overload
    def _transform_messages(
        self,
        messages: List[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> List[AllMessageValues]:
        ...

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: bool = False
    ) -> Union[List[AllMessageValues], Coroutine[Any, Any, List[AllMessageValues]]]:
        """
        Support translating:
        - video files from file_id or file_data to video_url
        - thinking_blocks on assistant messages to content blocks
        - Anthropic tool_result content blocks to OpenAI tool messages

        When Claude Code sends tool results (e.g. from WebFetch) through LiteLLM
        to hosted_vllm, they arrive as Anthropic-format tool_result blocks inside
        user messages. vLLM (OpenAI-compatible) expects these as separate messages
        with role="tool". This method extracts tool_result blocks and converts them
        to proper OpenAI tool messages.

        Fixes: https://github.com/BerriAI/litellm/issues/24491
        """
        transformed_messages: List[AllMessageValues] = []
        for message in messages:
            if message["role"] == "assistant":
                thinking_blocks = message.pop("thinking_blocks", None)  # type: ignore
                if thinking_blocks:
                    new_content: list = [
                        {"type": block["type"], "thinking": block.get("thinking", "")}
                        if block.get("type") == "thinking"
                        else {"type": block["type"], "data": block.get("data", "")}
                        for block in thinking_blocks
                    ]
                    existing_content = message.get("content")
                    if isinstance(existing_content, str):
                        new_content.append({"type": "text", "text": existing_content})
                    elif isinstance(existing_content, list):
                        new_content.extend(existing_content)
                    message["content"] = new_content  # type: ignore
                transformed_messages.append(message)
            elif message["role"] == "user":
                message_content = message.get("content")
                if message_content and isinstance(message_content, list):
                    # Separate Anthropic tool_result blocks from regular content
                    tool_messages: List[AllMessageValues] = []
                    remaining_content: List[Dict[str, Any]] = []

                    for content_item in message_content:
                        if isinstance(content_item, dict) and content_item.get("type") == "tool_result":
                            # Convert Anthropic tool_result to OpenAI tool message
                            tool_content = self._extract_tool_result_content(
                                content_item.get("content", "")
                            )
                            tool_msg: ChatCompletionToolMessage = {
                                "role": "tool",
                                "tool_call_id": content_item.get("tool_use_id", ""),
                                "content": tool_content,
                            }
                            tool_messages.append(tool_msg)  # type: ignore[arg-type]
                        else:
                            remaining_content.append(content_item)

                    # Add extracted tool messages before the user message
                    if tool_messages:
                        transformed_messages.extend(tool_messages)

                    if remaining_content:
                        message["content"] = remaining_content  # type: ignore
                        # Handle video file replacements on remaining content
                        for idx, ci in enumerate(remaining_content):
                            if ci.get("type") == "file":
                                file_item = cast(ChatCompletionFileObject, ci)
                                if self._is_video_file(file_item):
                                    remaining_content[idx] = self._convert_file_to_video_url(file_item)  # type: ignore
                        transformed_messages.append(message)
                    elif not tool_messages:
                        # No content at all, keep the message as-is
                        transformed_messages.append(message)
                    # If only tool_results were present and no remaining content,
                    # skip the now-empty user message
                else:
                    # String content or empty — handle video files in the old path
                    if message_content and isinstance(message_content, list):
                        replaced_content_items: List[
                            Tuple[int, ChatCompletionFileObject]
                        ] = []
                        for idx, content_item in enumerate(message_content):
                            if content_item.get("type") == "file":
                                content_item = cast(ChatCompletionFileObject, content_item)
                                if self._is_video_file(content_item):
                                    replaced_content_items.append((idx, content_item))
                        for idx, content_item in replaced_content_items:
                            message_content[idx] = self._convert_file_to_video_url(
                                content_item
                            )
                    transformed_messages.append(message)
            else:
                transformed_messages.append(message)

        # Replace original messages list in-place for compatibility with callers
        messages.clear()
        messages.extend(transformed_messages)

        if is_async:
            return super()._transform_messages(
                messages, model, is_async=cast(Literal[True], True)
            )
        else:
            return super()._transform_messages(
                messages, model, is_async=cast(Literal[False], False)
            )
