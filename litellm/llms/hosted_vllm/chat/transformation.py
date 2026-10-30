"""
Translate from OpenAI's `/v1/chat/completions` to VLLM's `/v1/chat/completions`
"""

import json
from collections.abc import Coroutine, Iterable, Mapping, Sequence
from typing import Final, Literal, cast, overload

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_image_mime_type_from_url,
)
from litellm.litellm_core_utils.prompt_templates.factory import parse_mime_type
from litellm.litellm_core_utils.reasoning_effort_utils import (
    reasoning_effort_from_thinking,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionAssistantToolCall,
    ChatCompletionFileObject,
    ChatCompletionUserMessage,
    ChatCompletionVideoObject,
    ChatCompletionVideoUrlObject,
    OpenAIMessageContentListBlock,
)

from ....utils import remove_additional_properties, remove_strict_from_schema
from ...openai.chat.gpt_transformation import OpenAIGPTConfig

_THINKING_KEYS: Final = frozenset({"thinking_blocks"})
_THINKING_AND_REASONING_KEYS: Final = frozenset({"thinking_blocks", "reasoning_content"})


def _content_part_type(part: object) -> object:
    mapping: Final = as_str_mapping(part)
    return None if mapping is None else mapping.get("type")


def _tool_use_as_tool_call(part: Mapping[str, object]) -> ChatCompletionAssistantToolCall:
    tool_input: Final = part.get("input", {})
    return cast(  # cast-ok: id and name are forwarded as the caller sent them
        ChatCompletionAssistantToolCall,
        {
            "id": part.get("id"),
            "type": "function",
            "function": {
                "name": part.get("name"),
                "arguments": tool_input if isinstance(tool_input, str) else json.dumps(tool_input),
            },
        },
    )


def _tool_call_ids(tool_calls: Iterable[object]) -> frozenset[object]:
    mappings: Final = tuple(mapping for mapping in map(as_str_mapping, tool_calls) if mapping is not None)
    return frozenset(mapping.get("id") for mapping in mappings if mapping.get("id") is not None)


def _tool_calls_update(
    existing: Sequence[ChatCompletionAssistantToolCall] | None,
    from_content: tuple[ChatCompletionAssistantToolCall, ...],
) -> Mapping[str, Sequence[ChatCompletionAssistantToolCall]]:
    if not from_content:
        return {}
    if not isinstance(existing, list):
        return {"tool_calls": list(from_content)}
    existing_ids: Final = _tool_call_ids(existing)
    new_tool_calls: Final = tuple(tool_call for tool_call in from_content if tool_call["id"] not in existing_ids)
    return {"tool_calls": [*existing, *new_tool_calls]} if new_tool_calls else {}


def _vllm_assistant_message(message: ChatCompletionAssistantMessage) -> AllMessageValues:
    dropped_keys: Final = (
        _THINKING_KEYS if isinstance(message.get("reasoning_content"), str) else _THINKING_AND_REASONING_KEYS
    )
    kept: Final = {key: value for key, value in message.items() if key not in dropped_keys}
    content: Final = message.get("content")
    if not isinstance(content, list):
        return cast(AllMessageValues, kept)  # cast-ok: only keys vLLM rejects are removed
    parts: Final[tuple[object, ...]] = tuple(content)
    mappings: Final = tuple(mapping for mapping in map(as_str_mapping, parts) if mapping is not None)
    texts: Final = tuple(
        cast(str, mapping.get("text", ""))  # cast-ok: a text part carries str text, and join rejects anything else
        for mapping in mappings
        if mapping.get("type") == "text"
    )
    tool_calls: Final = tuple(
        _tool_use_as_tool_call(mapping) for mapping in mappings if mapping.get("type") == "tool_use"
    )
    kept_parts: Final = [part for part in parts if _content_part_type(part) != "tool_use"]
    has_structured_content: Final = any(_content_part_type(part) not in ("text", "tool_use") for part in parts)
    return cast(  # cast-ok: content and tool_calls keep the shapes vLLM takes
        AllMessageValues,
        {
            **kept,
            "content": kept_parts if has_structured_content else "\n".join(texts),
            **_tool_calls_update(message.get("tool_calls"), tool_calls),
        },
    )


class HostedVLLMChatConfig(OpenAIGPTConfig):
    def _convert_custom_tools_to_function_tools(self, tools: list[dict[str, object]]) -> list[dict[str, object]]:
        """
        vLLM chat completions currently accepts only OpenAI function tools.
        Convert custom tools into function tools so request validation does not fail.
        """
        converted_tools: Final[list[dict[str, object]]] = []
        for idx, tool in enumerate(tools):
            if not isinstance(tool, dict):
                converted_tools.append(tool)
                continue

            if tool.get("type") != "custom":
                converted_tools.append(tool)
                continue

            custom_tool = tool.get("custom", {})
            if not isinstance(custom_tool, dict):
                custom_tool = {}

            tool_name = custom_tool.get("name") or tool.get("name") or f"custom_tool_{idx}"
            tool_description = custom_tool.get("description") or tool.get("description")
            tool_parameters = custom_tool.get("input_schema") or tool.get("input_schema")

            if not isinstance(tool_parameters, dict):
                tool_parameters = {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Raw tool input payload.",
                        }
                    },
                    "required": ["input"],
                }

            function_definition: dict[str, object] = {
                "name": str(tool_name),
                "parameters": tool_parameters,
            }
            if isinstance(tool_description, str):
                function_definition["description"] = tool_description

            converted_tools.append({"type": "function", "function": function_definition})

        return converted_tools

    def get_supported_openai_params(self, model: str) -> list[str]:
        params: Final = super().get_supported_openai_params(model)
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
            _tools = remove_additional_properties(_tools)
            _tools = remove_strict_from_schema(_tools)
            if isinstance(_tools, list):
                _tools = self._convert_custom_tools_to_function_tools(_tools)
        if _tools is not None:
            non_default_params["tools"] = _tools

        thinking: Final = non_default_params.pop("thinking", None)
        thinking_effort: Final = (
            None
            if "reasoning_effort" in non_default_params
            else reasoning_effort_from_thinking(thinking, model=model, drop_params=drop_params)
        )
        if thinking_effort is not None:
            non_default_params["reasoning_effort"] = thinking_effort

        return super().map_openai_params(non_default_params, optional_params, model, drop_params)

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        api_base = api_base or get_secret_str("HOSTED_VLLM_API_BASE")
        dynamic_api_key: Final = api_key or get_secret_str("HOSTED_VLLM_API_KEY") or "fake-api-key"
        return api_base, dynamic_api_key

    def get_openai_compatible_provider_info(
        self,
        api_base: str | None,
        api_key: str | None,
    ) -> tuple[str | None, str | None]:
        return self._get_openai_compatible_provider_info(api_base, api_key)

    def _is_video_file(self, content_item: ChatCompletionFileObject) -> bool:
        file: Final = content_item.get("file", {})
        format: Final = file.get("format")
        file_data: Final = file.get("file_data")
        file_id: Final = file.get("file_id")
        if content_item.get("type") != "file":
            return False
        if format and format.startswith("video/"):
            return True
        elif file_data:
            mime_type = parse_mime_type(file_data)
            if mime_type and mime_type.startswith("video/"):
                return True
        elif file_id:
            mime_type = get_image_mime_type_from_url(file_id)
            if mime_type and mime_type.startswith("video/"):
                return True
        return False

    def _convert_file_to_video_url(self, content_item: ChatCompletionFileObject) -> ChatCompletionVideoObject:
        file: Final = content_item.get("file", {})
        file_id: Final = file.get("file_id")
        file_data: Final = file.get("file_data")

        if file_id:
            return ChatCompletionVideoObject(type="video_url", video_url=ChatCompletionVideoUrlObject(url=file_id))
        elif file_data:
            return ChatCompletionVideoObject(type="video_url", video_url=ChatCompletionVideoUrlObject(url=file_data))
        raise ValueError("file_id or file_data is required")

    @overload
    def _transform_messages(
        self, messages: list[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[object, object, list[AllMessageValues]]: ...

    @overload
    def _transform_messages(
        self,
        messages: list[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> list[AllMessageValues]: ...

    def _transform_messages(
        self, messages: list[AllMessageValues], model: str, is_async: bool = False
    ) -> list[AllMessageValues] | Coroutine[object, object, list[AllMessageValues]]:
        """
        Support translating:
        - video files from file_id or file_data to video_url
        - thinking_blocks and non-string reasoning_content on assistant messages
          are removed, and content lists are converted to strings for vLLM compatibility
        """
        vllm_messages: Final = [self._vllm_message(message) for message in messages]
        if is_async:
            return super()._transform_messages(vllm_messages, model, is_async=True)
        return super()._transform_messages(vllm_messages, model, is_async=False)

    def _vllm_message(self, message: AllMessageValues) -> AllMessageValues:
        if message["role"] == "assistant":
            return _vllm_assistant_message(message)
        if message["role"] == "user":
            return self._vllm_user_message(message)
        return message

    def _vllm_user_message(self, message: ChatCompletionUserMessage) -> AllMessageValues:
        content: Final = message.get("content")
        if not content or not isinstance(content, list):
            return message
        return cast(  # cast-ok: only video file parts change, to the video_url part vLLM takes
            AllMessageValues, {**message, "content": [self._with_video_url(part) for part in content]}
        )

    def _with_video_url(self, part: OpenAIMessageContentListBlock) -> OpenAIMessageContentListBlock:
        if part.get("type") != "file":
            return part
        file_part: Final = cast(ChatCompletionFileObject, part)  # cast-ok: the type tag says this is a file part
        return self._convert_file_to_video_url(file_part) if self._is_video_file(file_part) else part
