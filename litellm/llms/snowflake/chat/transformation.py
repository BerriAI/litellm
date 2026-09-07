"""
Snowflake Cortex REST API — Chat Transformation

Routes to native Cortex REST API endpoints based on model:
  - Claude models → POST /api/v2/cortex/v1/messages (Anthropic format)
  - All other models → POST /api/v2/cortex/v1/chat/completions (OpenAI format)

Ref: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api
"""

import copy
import json
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Protocol, TypedDict

import httpx
from typing_extensions import ReadOnly

from litellm.litellm_core_utils.prompt_templates.factory import (
    anthropic_process_openai_file_message,
    convert_to_anthropic_tool_result,
    create_anthropic_image_param,
    select_anthropic_content_block_type_for_file,
)
from litellm.llms.anthropic.chat.handler import ModelResponseIterator as AnthropicStreamParser
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.anthropic.common_utils import normalize_cache_control_in_anthropic_payload
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolCallChunk, ChatCompletionToolMessage
from litellm.types.utils import (
    Choices,
    GenericStreamingChunk,
    Message,
    ModelResponse,
    ModelResponseStream,
)

from ...base_llm.base_model_iterator import BaseModelResponseIterator
from ...openai_like.chat.transformation import OpenAIGPTConfig
from ..utils import SnowflakeBaseConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

ANTHROPIC_VERSION: Final = "2023-06-01"

_CLAUDE_MODEL_PREFIXES: Final = (
    "claude-",
    "claude_",
)


class _AnthropicContentBlock(TypedDict, total=False):
    type: ReadOnly[str]
    text: ReadOnly[str]
    id: ReadOnly[str]
    name: ReadOnly[str]
    input: ReadOnly[Mapping[str, object]]


class _AnthropicUsageBlock(TypedDict, total=False):
    input_tokens: ReadOnly[int]
    output_tokens: ReadOnly[int]


class _AnthropicMessagesResponse(TypedDict, total=False):
    id: ReadOnly[str]
    model: ReadOnly[str]
    stop_reason: ReadOnly[str]
    content: ReadOnly[Sequence[_AnthropicContentBlock]]
    usage: ReadOnly[_AnthropicUsageBlock]


class _ChatCompletionsResponse(Protocol):
    """Response view that decodes the Cortex chat-completions body as a field mapping."""

    def json(self) -> Mapping[str, object]: ...


class _MessagesResponse(Protocol):
    """Response view that decodes the Cortex messages body in Anthropic shape."""

    def json(self) -> _AnthropicMessagesResponse: ...


def _decoded_chat_completions(response: _ChatCompletionsResponse) -> Mapping[str, object]:
    return response.json()


def _decoded_messages(response: _MessagesResponse) -> _AnthropicMessagesResponse:
    return response.json()


def _is_claude_model(model: str) -> bool:
    """Return True if model name (after stripping snowflake/ prefix) is a Claude model."""
    name: Final = model.lower().removeprefix("snowflake/")
    return any(name.startswith(p) for p in _CLAUDE_MODEL_PREFIXES)


def _convert_image_url_to_anthropic(block: Mapping[str, object]) -> object:
    """One OpenAI ``image_url`` block in the native shape Cortex accepts.

    Cortex documents base64 sources only, so remote URLs are inlined the way every
    other base64-only Anthropic dialect (Bedrock invoke, Vertex) inlines them, and
    pdf/text data URIs become document blocks rather than malformed image blocks.
    """
    image_url: Final = block.get("image_url")
    url: Final = image_url if isinstance(image_url, str) else _image_url_field(image_url, "url")
    if not url:
        return block

    converted: Final = (
        anthropic_process_openai_file_message({"type": "file", "file": {"file_data": url}})
        if select_anthropic_content_block_type_for_file(_data_uri_media_type(url)) == "document"
        else create_anthropic_image_param(
            image_url if isinstance(image_url, dict) else url,  # mutable-ok: caller's JSON block
            format=_image_url_field(image_url, "format"),
            is_bedrock_invoke=True,
        )
    )
    cache_control: Final = block.get("cache_control")
    if cache_control is None:
        return converted
    return {**converted, "cache_control": cache_control}  # mutable-ok: JSON wire block


def _image_url_field(image_url: object, key: str) -> str | None:
    value: Final = image_url.get(key) if isinstance(image_url, dict) else None
    return value if isinstance(value, str) else None


def _data_uri_media_type(url: str) -> str:
    match: Final = re.match(r"data:([^;,]+)", url)
    return match.group(1) if match else ""


def _convert_image_url_blocks_to_anthropic(content: object) -> object:
    if not isinstance(content, list):
        return content
    return [  # mutable-ok: JSON wire blocks
        _convert_image_url_to_anthropic(block)
        if isinstance(block, Mapping) and block.get("type") == "image_url"
        else block
        for block in content
    ]


def _convert_tool_result_to_anthropic(
    content: object, tool_call_id: str, cache_control: object
) -> Mapping[str, object]:
    """The Anthropic ``tool_result`` block for one OpenAI tool message.

    Delegating to the shared converter keeps image, document and per-block cache
    breakpoints identical to every other Anthropic dialect; only the plain-string
    and non-list shapes it does not model are handled here.
    """
    if not isinstance(content, list):
        plain: Final[dict[str, object]] = {  # mutable-ok: JSON wire block
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": content if isinstance(content, str) else json.dumps(content),
        }
        return {**plain, "cache_control": cache_control} if cache_control is not None else plain
    converted: Final = convert_to_anthropic_tool_result(
        ChatCompletionToolMessage(role="tool", tool_call_id=tool_call_id, content=content),
        force_base64=True,
    )
    if cache_control is None:
        return converted
    return {**converted, "cache_control": cache_control}  # mutable-ok: JSON wire block


def _signed_thinking_blocks(msg: object) -> list[dict[str, object]]:  # mutable-ok: JSON wire blocks
    """The assistant turn's thinking blocks that can legally be echoed back.

    Only signed blocks round-trip: Cortex rejects a thinking block whose signature is
    missing, which is what an unsigned block from a non-thinking turn would produce.
    """
    blocks: Final = msg.get("thinking_blocks") if isinstance(msg, dict) else getattr(msg, "thinking_blocks", None)
    if not isinstance(blocks, list):
        return []  # mutable-ok: JSON wire blocks
    return [  # mutable-ok: JSON wire blocks
        dict(block)
        for block in blocks
        if isinstance(block, Mapping) and (block.get("signature") or block.get("type") == "redacted_thinking")
    ]


def _clean_input_schema(schema: object) -> object:  # mutable-ok: JSON schema copy
    return (
        {key: value for key, value in schema.items() if key != "$schema"}
        if isinstance(schema, Mapping)
        else schema  # mutable-ok: JSON schema copy
    )  # mutable-ok: JSON schema copy


class SnowflakeConfig(SnowflakeBaseConfig, OpenAIGPTConfig):
    """
    Snowflake Cortex REST API — unified provider.

    Auto-routes based on model name:
      - Claude models → /api/v2/cortex/v1/messages (Anthropic Messages format)
      - All others   → /api/v2/cortex/v1/chat/completions (OpenAI format)

    Auth:
        PAT:  api_key="pat/<token>"  →  X-Snowflake-Authorization-Token-Type: PROGRAMMATIC_ACCESS_TOKEN
        JWT:  api_key="<jwt>"        →  X-Snowflake-Authorization-Token-Type: KEYPAIR_JWT
    """

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list[str]:
        params: Final = [
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "top_p",
            "stream",
            "tools",
            "tool_choice",
        ]
        if _is_claude_model(model):
            params.append("thinking")
        return params

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        api_base = self._get_api_base(api_base, optional_params)
        if _is_claude_model(model):
            return f"{api_base}/cortex/v1/messages"
        return f"{api_base}/cortex/v1/chat/completions"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        headers = super().validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )
        if _is_claude_model(model):
            headers["anthropic-version"] = ANTHROPIC_VERSION
        return headers

    def _transform_tools_to_anthropic(self, tools: list[dict]) -> list[dict]:
        """
        Convert tools from OpenAI format to Anthropic format.

        OpenAI: {"type": "function", "function": {"name": ..., "parameters": {...}}}
        Anthropic: {"name": ..., "description": ..., "input_schema": {...}}
        """
        anthropic_tools: Final = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                anthropic_tool: dict[str, object] = {
                    "name": func.get("name", ""),
                }
                if "description" in func:
                    anthropic_tool["description"] = func["description"]
                if "parameters" in func:
                    anthropic_tool["input_schema"] = _clean_input_schema(func["parameters"])
                else:
                    anthropic_tool["input_schema"] = {
                        "type": "object",
                        "properties": {},
                    }
                anthropic_tools.append(anthropic_tool)
            else:
                anthropic_tools.append(
                    {**tool, "input_schema": _clean_input_schema(tool["input_schema"])}  # mutable-ok: JSON wire tool
                    if "input_schema" in tool
                    else tool
                )
        return anthropic_tools

    def _extract_system_and_messages(  # mutable-ok: JSON wire messages
        self, messages: list[AllMessageValues]
    ) -> tuple[list[dict] | None, list[dict]]:
        """
        Split messages into system prompt and conversation turns for Anthropic format.

        - system messages → collected and joined (preserves guardrail prompts)
        - assistant messages with tool_calls → tool_use content blocks
        - tool role messages → user role with tool_result content blocks
        """
        system_parts: Final[list[dict]] = []  # mutable-ok: JSON wire messages
        conversation: Final[list[dict]] = []  # mutable-ok: JSON wire messages

        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content: Any = msg.get("content", "")
                msg_cache_control: object = msg.get("cache_control")
            else:
                role = getattr(msg, "role", "")
                content = getattr(msg, "content", "")
                msg_cache_control = getattr(msg, "cache_control", None)

            if role == "system":
                if isinstance(content, str) and content:
                    system_parts.append({"type": "text", "text": content})  # mutable-ok: JSON wire system block
                elif isinstance(content, list):
                    system_parts.extend(
                        {  # mutable-ok: JSON wire system block
                            "type": "text",
                            "text": block.get("text", ""),
                            **(
                                {"cache_control": block["cache_control"]} if "cache_control" in block else {}
                            ),  # mutable-ok: JSON wire block
                        }
                        for block in content
                        if isinstance(block, Mapping) and block.get("type") == "text"
                    )
            elif role == "assistant":
                tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
                thinking_blocks = _signed_thinking_blocks(msg)
                if tool_calls:
                    content_blocks: list[dict[str, object]] = list(thinking_blocks)  # mutable-ok: JSON wire blocks
                    if content:
                        content_blocks.append({"type": "text", "text": content})
                    for tc in tool_calls:
                        func = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})
                        tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                        func_name = func.get("name", "") if isinstance(func, dict) else getattr(func, "name", "")
                        func_args = (
                            func.get("arguments", "{}") if isinstance(func, dict) else getattr(func, "arguments", "{}")
                        )
                        try:
                            input_data = json.loads(func_args) if isinstance(func_args, str) else func_args
                        except (json.JSONDecodeError, TypeError):
                            input_data = {}
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc_id,
                                "name": func_name,
                                "input": input_data,
                            }
                        )
                    conversation.append({"role": "assistant", "content": content_blocks})
                elif thinking_blocks:
                    thinking_content = (
                        [
                            *thinking_blocks,
                            *copy.deepcopy(content),
                        ]
                        if isinstance(content, list)
                        else [*thinking_blocks, *([{"type": "text", "text": content}] if content else [])]
                    )  # rebind-ok: loop-local normalized content
                    conversation.append({"role": "assistant", "content": thinking_content})
                else:
                    conversation.append({"role": "assistant", "content": content})
            elif role == "tool":
                tool_call_id_value = (
                    msg.get("tool_call_id", "") if isinstance(msg, dict) else getattr(msg, "tool_call_id", "")
                )
                tool_call_id = (
                    tool_call_id_value if isinstance(tool_call_id_value, str) else ""
                )  # rebind-ok: normalized loop value
                tool_result_block = _convert_tool_result_to_anthropic(content, tool_call_id, msg_cache_control)
                if (
                    conversation
                    and conversation[-1]["role"] == "user"
                    and isinstance(conversation[-1]["content"], list)
                    and conversation[-1]["content"]
                    and conversation[-1]["content"][0].get("type") == "tool_result"
                ):
                    conversation[-1]["content"].append(tool_result_block)
                else:
                    conversation.append(
                        {"role": "user", "content": [tool_result_block]}  # mutable-ok: JSON wire message
                    )  # mutable-ok: JSON wire message
            else:
                conversation.append(  # mutable-ok: JSON wire message
                    {  # mutable-ok: JSON wire message
                        "role": role,
                        "content": _convert_image_url_blocks_to_anthropic(content),
                    }  # mutable-ok: JSON wire message
                )

        system: Final[list[dict] | None] = system_parts if system_parts else None  # mutable-ok: JSON wire messages
        return system, conversation

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        stream: Final[bool] = optional_params.pop("stream", False) or False
        extra_body: Final = optional_params.pop("extra_body", {})

        if _is_claude_model(model):
            return self._transform_request_anthropic(model, messages, optional_params, stream, extra_body)
        return self._transform_request_openai(model, messages, optional_params, stream, extra_body)

    def _transform_request_openai(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        stream: bool,
        extra_body: dict,
    ) -> dict:
        """OpenAI format for /chat/completions endpoint."""
        max_tokens: Final = optional_params.pop("max_tokens", None)
        max_completion_tokens: Final = optional_params.pop("max_completion_tokens", None)
        resolved_max: Final = max_completion_tokens or max_tokens

        body: Final[dict] = {
            "model": model.removeprefix("snowflake/"),
            "messages": messages,
            "stream": stream,
            **optional_params,
            **extra_body,
        }

        if resolved_max is not None:
            body["max_completion_tokens"] = resolved_max

        return body

    def _transform_tool_choice_to_anthropic(self, tool_choice: Any) -> dict[str, Any]:
        """
        Convert tool_choice from OpenAI format to Anthropic format.

        OpenAI string values: "auto", "required", "none"
        OpenAI dict: {"type": "function", "function": {"name": "..."}}
        Anthropic: {"type": "auto"}, {"type": "any"}, {"type": "tool", "name": "..."}
        """
        if isinstance(tool_choice, str):
            mapping: Final = {
                "auto": {"type": "auto"},
                "required": {"type": "any"},
                "none": {"type": "none"},
            }
            return mapping.get(tool_choice, {"type": "auto"})
        elif isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                func: Final = tool_choice.get("function", {})
                return {"type": "tool", "name": func.get("name", "")}
            return tool_choice
        return {"type": "auto"}

    def _transform_request_anthropic(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        stream: bool,
        extra_body: dict,
    ) -> dict:
        """Anthropic Messages format for /messages endpoint."""
        passthrough_system: Final = optional_params.pop("system", None)
        extracted_system, conversation = self._extract_system_and_messages(messages)
        system: Final = passthrough_system if passthrough_system is not None else extracted_system

        if "tools" in optional_params:
            optional_params["tools"] = self._transform_tools_to_anthropic(optional_params["tools"])

        if "tool_choice" in optional_params:
            optional_params["tool_choice"] = self._transform_tool_choice_to_anthropic(optional_params["tool_choice"])

        max_completion_tokens: Final = optional_params.pop("max_completion_tokens", None)
        if max_completion_tokens and "max_tokens" not in optional_params:
            optional_params["max_tokens"] = max_completion_tokens

        model_name: Final = model.removeprefix("snowflake/")

        body: Final[dict[str, object]] = normalize_cache_control_in_anthropic_payload(  # mutable-ok: JSON wire body
            {  # mutable-ok: JSON wire body
                "model": model_name,
                "messages": conversation,
                "stream": stream,
                **optional_params,
                **extra_body,  # mutable-ok: JSON wire body
            }
        )
        if system is not None:
            body["system"] = normalize_cache_control_in_anthropic_payload(  # mutable-ok: JSON wire payload
                {"system": system}  # mutable-ok: JSON wire payload
            )["system"]

        if "max_tokens" not in body:
            body["max_tokens"] = 4096  # reasonable default; Anthropic API max varies by model

        return body

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: object,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        if _is_claude_model(model):
            return self._transform_response_anthropic(
                model, raw_response, model_response, logging_obj, request_data, messages
            )
        return self._transform_response_openai(model, raw_response, model_response, logging_obj, request_data, messages)

    def _transform_response_openai(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: list[AllMessageValues],
    ) -> ModelResponse:
        """Parse standard OpenAI chat completions response."""
        response_json: Final = _decoded_chat_completions(raw_response)

        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=response_json,
            additional_args={"complete_input_dict": request_data},
        )

        returned_response: Final = ModelResponse(**response_json)
        returned_response.model = "snowflake/" + (returned_response.model or "")

        if model is not None:
            returned_response._hidden_params["model"] = model

        return returned_response

    def _transform_response_anthropic(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: list[AllMessageValues],
    ) -> ModelResponse:
        """Parse Anthropic Messages response into OpenAI format."""
        response_json: Final = _decoded_messages(raw_response)

        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=response_json,
            additional_args={"complete_input_dict": request_data},
        )

        anthropic_config: Final = AnthropicConfig()
        text_content, _, thinking_blocks, reasoning_content, tool_calls, _, _, _ = (
            anthropic_config.extract_response_content(completion_response=dict(response_json))
        )

        _stop_reason_map: Final = {
            "end_turn": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
            "stop_sequence": "stop",
        }
        finish_reason: Final = _stop_reason_map.get(response_json.get("stop_reason", "end_turn"), "stop")

        message: Final = Message(
            content=text_content or None,
            role="assistant",
            tool_calls=tool_calls or None,
            thinking_blocks=thinking_blocks,
            reasoning_content=reasoning_content,
        )

        choice: Final = Choices(
            finish_reason=finish_reason,
            index=0,
            message=message,
        )

        # Cortex reports prompt-cache creation/read counts alongside input_tokens; the
        # shared calculator folds them into prompt_tokens_details so cached input is
        # visible and billed at its own rate.
        usage: Final = anthropic_config.calculate_usage(
            usage_object=response_json.get("usage", {}),
            reasoning_content=reasoning_content,
            completion_response=dict(response_json),
        )

        model_response.choices = [choice]
        model_response.usage = usage
        model_response.model = "snowflake/" + response_json.get("model", model)
        model_response.id = response_json.get("id", "")

        if model is not None:
            model_response._hidden_params["model"] = model

        return model_response

    def get_model_response_iterator(
        self,
        streaming_response: object,
        sync_stream: bool,
        json_mode: bool | None = False,
    ) -> "SnowflakeStreamingHandler":
        return SnowflakeStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


class SnowflakeStreamingHandler(BaseModelResponseIterator):
    """
    Parse streaming events from both Snowflake endpoints.

    - /chat/completions: OpenAI SSE format (has "choices" key)
    - /messages: Anthropic SSE format (has "type" key like content_block_delta)
    """

    def __init__(
        self,
        streaming_response: object,
        sync_stream: bool,
        json_mode: bool | None = False,
    ):
        super().__init__(streaming_response=streaming_response, sync_stream=sync_stream)
        # Cortex streams the Anthropic SSE dialect on /messages, so its events are parsed
        # by Anthropic's own parser: thinking deltas, signatures and prompt-cache usage
        # all arrive the way they do on every other Anthropic-dialect provider.
        self._anthropic_parser: Final = AnthropicStreamParser(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )

    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk | ModelResponseStream:
        if "choices" in chunk:
            return self._parse_openai_chunk(chunk)
        return self._anthropic_parser.chunk_parser(chunk)

    def _parse_openai_chunk(self, chunk: dict) -> GenericStreamingChunk:
        choices: Final = chunk.get("choices", [])
        if not choices:
            return GenericStreamingChunk(
                text="",
                is_finished=False,
                finish_reason="",
                usage=None,
                index=0,
                tool_use=None,
            )

        choice: Final = choices[0]
        delta: Final = choice.get("delta", {})
        finish_reason: Final = choice.get("finish_reason") or ""
        text: Final = delta.get("content") or ""

        tool_use = None
        tool_calls: Final = delta.get("tool_calls")
        if tool_calls:
            tc: Final = tool_calls[0]
            func: Final = tc.get("function", {})
            tool_use = ChatCompletionToolCallChunk(
                id=tc.get("id", ""),
                type="function",
                function={
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", ""),
                },
                index=tc.get("index", 0),
            )

        return GenericStreamingChunk(
            text=text,
            is_finished=finish_reason != "",
            finish_reason=finish_reason,
            usage=None,
            index=choice.get("index", 0),
            tool_use=tool_use,
        )
