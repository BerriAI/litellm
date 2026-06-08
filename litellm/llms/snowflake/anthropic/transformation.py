"""
Snowflake Cortex REST API — Anthropic-Compatible Endpoint

Routes to the native Anthropic-compatible endpoint:
  POST /api/v2/cortex/v1/messages

Use this config for Claude models when you need Claude-specific features:
  - Extended thinking / reasoning (thinking parameter)
  - Prompt caching (cache_control on messages/system)
  - Anthropic tool format

For standard usage (all models, no Claude-specific features), use the
default SnowflakeConfig which routes to /api/v2/cortex/v1/chat/completions.

Ref: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx

from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    Choices,
    Message,
    ModelResponse,
    Usage,
)

from ..utils import SnowflakeBaseConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

ANTHROPIC_VERSION = "2023-06-01"

_CLAUDE_MODEL_PREFIXES = (
    "claude-",
    "claude_",
)


def _is_claude_model(model: str) -> bool:
    """Return True if model name (after stripping snowflake/ prefix) is a Claude model."""
    name = model.lower().removeprefix("snowflake/")
    return any(name.startswith(p) for p in _CLAUDE_MODEL_PREFIXES)


class SnowflakeCortexAnthropicConfig(SnowflakeBaseConfig):
    """
    Snowflake Cortex REST API — Anthropic Messages endpoint.

    Endpoint: POST /api/v2/cortex/v1/messages

    Designed for Claude models. Accepts and returns Anthropic Messages API
    format. Supports thinking, prompt caching, and extended tool formats.

    Usage in litellm:
        import litellm
        response = litellm.completion(
            model="snowflake/claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="pat/<your-pat>",
            api_base="https://<account>.snowflakecomputing.com",
            custom_llm_provider="snowflake-anthropic",  # routes to this config
        )
    """

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "temperature",
            "max_tokens",
            "top_p",
            "stream",
            "tools",
            "tool_choice",
            "thinking",
        ]

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Returns the Anthropic-compatible Cortex REST API endpoint.

            https://{account}.snowflakecomputing.com/api/v2/cortex/v1/messages
        """
        api_base = self._get_api_base(api_base, optional_params)
        return f"{api_base}/cortex/v1/messages"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Set Snowflake auth headers + Anthropic version header.
        """
        headers = super().validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )
        headers["anthropic-version"] = ANTHROPIC_VERSION
        return headers

    def _transform_tools_to_anthropic(self, tools: List[Dict]) -> List[Dict]:
        """
        Convert tools from OpenAI format to Anthropic format.

        OpenAI: {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
        Anthropic: {"name": ..., "description": ..., "input_schema": {...}}
        """
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                anthropic_tool: Dict[str, Any] = {
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

    def _extract_system_and_messages(
        self, messages: List[AllMessageValues]
    ) -> tuple[Optional[Union[str, List[Dict]]], List[Dict]]:
        """
        Split messages into system prompt and conversation turns.

        Anthropic's /messages endpoint takes system as a top-level param,
        not inside the messages array.

        Handles tool-use messages:
        - assistant messages with tool_calls → converted to Anthropic tool_use content blocks
        - tool role messages → converted to user role with tool_result content blocks
        """
        system: Optional[Union[str, List[Dict]]] = None
        conversation: List[Dict] = []

        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", "")
                content = getattr(msg, "content", "")

            if role == "system":
                system = content
            elif role == "assistant":
                tool_calls = (
                    msg.get("tool_calls")
                    if isinstance(msg, dict)
                    else getattr(msg, "tool_calls", None)
                )
                if tool_calls:
                    content_blocks: List[Dict[str, Any]] = []
                    if content:
                        content_blocks.append({"type": "text", "text": content})
                    for tc in tool_calls:
                        func = (
                            tc.get("function", {})
                            if isinstance(tc, dict)
                            else getattr(tc, "function", {})
                        )
                        tc_id = (
                            tc.get("id", "")
                            if isinstance(tc, dict)
                            else getattr(tc, "id", "")
                        )
                        func_name = (
                            func.get("name", "")
                            if isinstance(func, dict)
                            else getattr(func, "name", "")
                        )
                        func_args = (
                            func.get("arguments", "{}")
                            if isinstance(func, dict)
                            else getattr(func, "arguments", "{}")
                        )
                        try:
                            input_data = (
                                json.loads(func_args)
                                if isinstance(func_args, str)
                                else func_args
                            )
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
                    conversation.append(
                        {"role": "assistant", "content": content_blocks}
                    )
                else:
                    conversation.append({"role": "assistant", "content": content})
            elif role == "tool":
                tool_call_id = (
                    msg.get("tool_call_id", "")
                    if isinstance(msg, dict)
                    else getattr(msg, "tool_call_id", "")
                )
                tool_content = (
                    content if isinstance(content, str) else json.dumps(content)
                )
                conversation.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call_id,
                                "content": tool_content,
                            }
                        ],
                    }
                )
            else:
                conversation.append({"role": role, "content": content})

        return system, conversation

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform to Anthropic Messages API format.

        Key differences from OpenAI format:
        - system message → top-level "system" field
        - max_tokens is required
        - tools use Anthropic format (input_schema, not parameters)
        """
        stream: bool = optional_params.pop("stream", False) or False
        extra_body = optional_params.pop("extra_body", {})

        system, conversation = self._extract_system_and_messages(messages)

        if "tools" in optional_params:
            optional_params["tools"] = self._transform_tools_to_anthropic(
                optional_params["tools"]
            )

        model_name = model.removeprefix("snowflake/")

        body: Dict[str, Any] = {
            "model": model_name,
            "messages": conversation,
            "stream": stream,
            **optional_params,
            **extra_body,
        }

        if system is not None:
            body["system"] = system

        if "max_tokens" not in body:
            body["max_tokens"] = 1024

        return body

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Transform Anthropic Messages response to OpenAI ChatCompletion format.

        Anthropic response:
        {
            "id": "msg_...",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }

        Output: standard OpenAI ModelResponse
        """
        response_json = raw_response.json()

        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=response_json,
            additional_args={"complete_input_dict": request_data},
        )

        text_content = ""
        tool_calls = []

        for block in response_json.get("content", []):
            if block.get("type") == "text":
                text_content += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )

        _stop_reason_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
            "stop_sequence": "stop",
        }
        finish_reason = _stop_reason_map.get(
            response_json.get("stop_reason", "end_turn"), "stop"
        )

        message = Message(content=text_content or None, role="assistant")
        if tool_calls:
            message.tool_calls = tool_calls  # type: ignore

        choice = Choices(
            finish_reason=finish_reason,
            index=0,
            message=message,
        )

        usage_data = response_json.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0)
            + usage_data.get("output_tokens", 0),
        )

        model_response.choices = [choice]
        model_response.usage = usage
        model_response.model = "snowflake/" + response_json.get("model", model)
        model_response.id = response_json.get("id", "")

        if model is not None:
            model_response._hidden_params["model"] = model

        return model_response
