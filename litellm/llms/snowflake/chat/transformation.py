"""
Snowflake Cortex REST API — Chat Transformation

Routes to native Cortex REST API endpoints based on model:
  - Claude models → POST /api/v2/cortex/v1/messages (Anthropic format)
  - All other models → POST /api/v2/cortex/v1/chat/completions (OpenAI format)

Ref: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api
"""

import json
from typing import TYPE_CHECKING, Any, Final

import httpx

from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolCallChunk
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    ChatCompletionUsageBlock,
    Choices,
    Function,
    GenericStreamingChunk,
    Message,
    ModelResponse,
    Usage,
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


def _is_claude_model(model: str) -> bool:
    """Return True if model name (after stripping snowflake/ prefix) is a Claude model."""
    name: Final = model.lower().removeprefix("snowflake/")
    return any(name.startswith(p) for p in _CLAUDE_MODEL_PREFIXES)


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
                anthropic_tool: dict[str, Any] = {
                    "name": func.get("name", ""),
                }
                if "description" in func:
                    anthropic_tool["description"] = func["description"]
                if "parameters" in func:
                    anthropic_tool["input_schema"] = func["parameters"]
                else:
                    anthropic_tool["input_schema"] = {
                        "type": "object",
                        "properties": {},
                    }
                anthropic_tools.append(anthropic_tool)
            else:
                anthropic_tools.append(tool)
        return anthropic_tools

    def _extract_system_and_messages(self, messages: list[AllMessageValues]) -> tuple[str | None, list[dict]]:
        """
        Split messages into system prompt and conversation turns for Anthropic format.

        - system messages → collected and joined (preserves guardrail prompts)
        - assistant messages with tool_calls → tool_use content blocks
        - tool role messages → user role with tool_result content blocks
        """
        system_parts: Final[list[str]] = []
        conversation: Final[list[dict]] = []

        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content: Any = msg.get("content", "")
            else:
                role = getattr(msg, "role", "")
                content = getattr(msg, "content", "")

            if role == "system":
                if isinstance(content, str) and content:
                    system_parts.append(content)
                elif isinstance(content, list):
                    system_parts.append("\n".join(b.get("text", "") for b in content if b.get("type") == "text"))
            elif role == "assistant":
                tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
                if tool_calls:
                    content_blocks: list[dict[str, Any]] = []
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
                else:
                    conversation.append({"role": "assistant", "content": content})
            elif role == "tool":
                tool_call_id = (
                    msg.get("tool_call_id", "") if isinstance(msg, dict) else getattr(msg, "tool_call_id", "")
                )
                tool_content = content if isinstance(content, str) else json.dumps(content)
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": tool_content,
                }
                if (
                    conversation
                    and conversation[-1]["role"] == "user"
                    and isinstance(conversation[-1]["content"], list)
                    and conversation[-1]["content"]
                    and conversation[-1]["content"][0].get("type") == "tool_result"
                ):
                    conversation[-1]["content"].append(tool_result_block)
                else:
                    conversation.append({"role": "user", "content": [tool_result_block]})
            else:
                conversation.append({"role": role, "content": content})

        system: Final[str | None] = "\n\n".join(system_parts) if system_parts else None
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
        system, conversation = self._extract_system_and_messages(messages)

        if "tools" in optional_params:
            optional_params["tools"] = self._transform_tools_to_anthropic(optional_params["tools"])

        if "tool_choice" in optional_params:
            optional_params["tool_choice"] = self._transform_tool_choice_to_anthropic(optional_params["tool_choice"])

        max_completion_tokens: Final = optional_params.pop("max_completion_tokens", None)
        if max_completion_tokens and "max_tokens" not in optional_params:
            optional_params["max_tokens"] = max_completion_tokens

        model_name: Final = model.removeprefix("snowflake/")

        body: Final[dict[str, Any]] = {
            "model": model_name,
            "messages": conversation,
            "stream": stream,
            **optional_params,
            **extra_body,
        }

        if system is not None:
            body["system"] = system

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
        encoding: Any,
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
        response_json: Final = raw_response.json()

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
        response_json: Final = raw_response.json()

        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=response_json,
            additional_args={"complete_input_dict": request_data},
        )

        text_content = ""
        tool_calls: Final = []

        for block in response_json.get("content", []):
            if block.get("type") == "text":
                text_content += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=block.get("id", ""),
                        type="function",
                        function=Function(
                            name=block.get("name", ""),
                            arguments=json.dumps(block.get("input", {})),
                        ),
                    )
                )

        _stop_reason_map: Final = {
            "end_turn": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
            "stop_sequence": "stop",
        }
        finish_reason: Final = _stop_reason_map.get(response_json.get("stop_reason", "end_turn"), "stop")

        message: Final = Message(content=text_content or None, role="assistant")
        if tool_calls:
            message.tool_calls = tool_calls

        choice: Final = Choices(
            finish_reason=finish_reason,
            index=0,
            message=message,
        )

        usage_data: Final = response_json.get("usage", {})
        usage: Final = Usage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
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
        streaming_response: Any,
        sync_stream: bool,
        json_mode: bool | None = False,
    ) -> Any:
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
        streaming_response: Any,
        sync_stream: bool,
        json_mode: bool | None = False,
    ):
        super().__init__(streaming_response=streaming_response, sync_stream=sync_stream)
        self._tool_index = 0
        self._tool_id = ""
        self._tool_name = ""
        self._input_tokens = 0

    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        if "choices" in chunk:
            return self._parse_openai_chunk(chunk)
        return self._parse_anthropic_chunk(chunk)

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

    def _parse_anthropic_chunk(self, chunk: dict) -> GenericStreamingChunk:
        event_type: Final = chunk.get("type", "")

        if event_type == "message_start":
            message: Final = chunk.get("message", {})
            usage_data = message.get("usage", {})
            self._input_tokens = usage_data.get("input_tokens", 0)
            return GenericStreamingChunk(
                text="",
                is_finished=False,
                finish_reason="",
                usage=None,
                index=0,
                tool_use=None,
            )

        elif event_type == "content_block_delta":
            delta = chunk.get("delta", {})
            delta_type: Final = delta.get("type", "")

            if delta_type == "text_delta":
                return GenericStreamingChunk(
                    text=delta.get("text", ""),
                    is_finished=False,
                    finish_reason="",
                    usage=None,
                    index=chunk.get("index", 0),
                    tool_use=None,
                )
            elif delta_type == "input_json_delta":
                return GenericStreamingChunk(
                    text="",
                    is_finished=False,
                    finish_reason="",
                    usage=None,
                    index=chunk.get("index", 0),
                    tool_use=ChatCompletionToolCallChunk(
                        id=self._tool_id,
                        type="function",
                        function={
                            "name": self._tool_name,
                            "arguments": delta.get("partial_json", ""),
                        },
                        index=self._tool_index,
                    ),
                )

        elif event_type == "content_block_start":
            content_block: Final = chunk.get("content_block", {})
            if content_block.get("type") == "tool_use":
                self._tool_id = content_block.get("id", "")
                self._tool_name = content_block.get("name", "")
                self._tool_index = chunk.get("index", 0)
                return GenericStreamingChunk(
                    text="",
                    is_finished=False,
                    finish_reason="",
                    usage=None,
                    index=chunk.get("index", 0),
                    tool_use=ChatCompletionToolCallChunk(
                        id=self._tool_id,
                        type="function",
                        function={"name": self._tool_name, "arguments": ""},
                        index=self._tool_index,
                    ),
                )

        elif event_type == "message_delta":
            delta = chunk.get("delta", {})
            stop_reason: Final = delta.get("stop_reason", "")
            usage_data = chunk.get("usage", {})
            _stop_map: Final = {
                "end_turn": "stop",
                "max_tokens": "length",
                "tool_use": "tool_calls",
                "stop_sequence": "stop",
            }
            usage = None
            if usage_data or self._input_tokens:
                output_t: Final = usage_data.get("output_tokens", 0)
                input_t: Final = self._input_tokens or usage_data.get("input_tokens", 0)
                usage = ChatCompletionUsageBlock(
                    prompt_tokens=input_t,
                    completion_tokens=output_t,
                    total_tokens=input_t + output_t,
                )
            return GenericStreamingChunk(
                text="",
                is_finished=True,
                finish_reason=_stop_map.get(stop_reason, "stop"),
                usage=usage,
                index=0,
                tool_use=None,
            )

        elif event_type == "message_stop":
            return GenericStreamingChunk(
                text="",
                is_finished=True,
                finish_reason="stop",
                usage=None,
                index=0,
                tool_use=None,
            )

        return GenericStreamingChunk(
            text="",
            is_finished=False,
            finish_reason="",
            usage=None,
            index=0,
            tool_use=None,
        )
