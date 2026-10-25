"""
Translate from OpenAI's `/v1/chat/completions` to VLLM's `/v1/chat/completions`
"""

import json
from collections.abc import Coroutine, Mapping
from typing import TYPE_CHECKING, Final, Literal, cast, overload

import httpx
from pydantic import JsonValue

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
    ChatCompletionAssistantToolCall,
    ChatCompletionFileObject,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionVideoObject,
    ChatCompletionVideoUrlObject,
)
from litellm.types.utils import Choices, ModelResponse

from ....utils import remove_additional_properties, remove_strict_from_schema
from ...openai.chat.gpt_transformation import OpenAIGPTConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

_VLLM_CHOICE_FIELDS: Final = ("stop_reason", "token_ids")


def _vllm_choice_fields(choice: Choices) -> Mapping[str, object]:
    provider_fields: Final[Mapping[str, object]] = (
        choice.provider_specific_fields or {} if "provider_specific_fields" in choice else {}
    )
    return {field: provider_fields[field] for field in _VLLM_CHOICE_FIELDS if field in provider_fields}


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
            else reasoning_effort_from_thinking(thinking, drop_params=drop_params)
        )
        if thinking_effort is not None:
            non_default_params["reasoning_effort"] = thinking_effort

        return super().map_openai_params(non_default_params, optional_params, model, drop_params)

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict[str, JsonValue],  # mutable-ok: BaseConfig signature
        messages: list[AllMessageValues],  # mutable-ok: BaseConfig signature
        optional_params: dict[str, JsonValue],  # mutable-ok: BaseConfig signature
        litellm_params: dict[str, JsonValue],  # mutable-ok: BaseConfig signature
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        response: Final = super().transform_response(  # pyright: ignore[reportUnknownMemberType]  # OpenAIGPTConfig.transform_response takes bare dicts
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )
        for choice in response.choices:
            for field, value in _vllm_choice_fields(choice).items():
                choice[field] = value
        return response

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
        for message in messages:
            if message["role"] == "assistant":
                message.pop("thinking_blocks", None)
                if not isinstance(message.get("reasoning_content"), str):
                    message.pop("reasoning_content", None)
                existing_content = message.get("content")
                if isinstance(existing_content, list):
                    text_parts = []
                    tool_calls: list[ChatCompletionAssistantToolCall] = []
                    content_blocks: list[object] = []
                    has_structured_content = False
                    for c in existing_content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            text_parts.append(c.get("text", ""))
                            content_blocks.append(c)
                        elif isinstance(c, dict) and c.get("type") == "tool_use":
                            tool_input = c.get("input", {})
                            tool_calls.append(
                                ChatCompletionAssistantToolCall(
                                    id=c.get("id"),
                                    type="function",
                                    function=ChatCompletionToolCallFunctionChunk(
                                        name=c.get("name"),
                                        arguments=(
                                            tool_input
                                            if isinstance(
                                                tool_input,
                                                str,
                                            )
                                            else json.dumps(tool_input)
                                        ),
                                    ),
                                )
                            )
                        else:
                            content_blocks.append(c)
                            has_structured_content = True
                    if tool_calls:
                        existing_tool_calls = message.get("tool_calls")
                        if isinstance(existing_tool_calls, list):
                            existing_tool_call_ids = {
                                tool_call.get("id")
                                for tool_call in existing_tool_calls
                                if isinstance(tool_call, dict) and tool_call.get("id") is not None
                            }
                            new_tool_calls = [
                                tool_call
                                for tool_call in tool_calls
                                if tool_call.get("id") not in existing_tool_call_ids
                            ]
                            if new_tool_calls:
                                message["tool_calls"] = existing_tool_calls + new_tool_calls
                        else:
                            message["tool_calls"] = tool_calls
                    content_str = "\n".join(text_parts)
                    new_content = content_blocks if has_structured_content else content_str
                    message["content"] = new_content
            elif message["role"] == "user":
                message_content = message.get("content")
                if message_content and isinstance(message_content, list):
                    replaced_content_items: list[tuple[int, ChatCompletionFileObject]] = []
                    for idx, content_item in enumerate(message_content):
                        if content_item.get("type") == "file":
                            content_item = cast(ChatCompletionFileObject, content_item)
                            if self._is_video_file(content_item):
                                replaced_content_items.append((idx, content_item))
                    for idx, content_item in replaced_content_items:
                        message_content[idx] = self._convert_file_to_video_url(content_item)

        if is_async:
            return super()._transform_messages(messages, model, is_async=cast(Literal[True], True))
        else:
            return super()._transform_messages(messages, model, is_async=cast(Literal[False], False))
