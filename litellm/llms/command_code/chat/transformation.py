"""
Transformation for the Command Code API.

Command Code (https://api.commandcode.ai) exposes a custom streaming
generation endpoint:

    POST {api_base}/alpha/generate

The API is not OpenAI-compatible: OpenAI-style params are nested under
``params``, the system prompt is a flat string, tool schemas use
``input_schema`` and the response is a newline-delimited stream of typed
JSON events. See litellm/types/llms/command_code.py for the wire types.
"""

from __future__ import annotations

import datetime
import json
import uuid
from typing import TYPE_CHECKING

import httpx

from litellm._logging import verbose_logger
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.command_code.chat.sse_iterator import CommandCodeSSEStreamIterator
from litellm.llms.command_code.common_utils import (
    COMMAND_CODE_API_BASE,
    COMMAND_CODE_DEFAULT_MAX_TOKENS,
    CommandCodeError,
    flatten_system_messages,
    get_command_code_headers,
    map_command_code_finish_reason,
    parse_stream_event_line,
    parse_tool_call_input,
    transform_messages_to_command_code,
    transform_tools_to_command_code,
    usage_from_finish_event,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.command_code import (
    CommandCodeConfigBlock,
    CommandCodeParamsBlock,
    CommandCodeRequestBody,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
    from litellm.utils import CustomStreamWrapper


class CommandCodeConfig(BaseConfig):
    """
    Configuration for the Command Code API.

    Supported OpenAI params: stream, max_tokens, max_completion_tokens,
    temperature, tools. Other params are not accepted by the generation
    endpoint and are not claimed as supported.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_supported_openai_params(self, model: str) -> list[str]:
        return [
            "stream",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "tools",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "stream":
                optional_params["stream"] = value
            elif param in ("max_tokens", "max_completion_tokens"):
                optional_params["max_tokens"] = value
            elif param == "temperature":
                optional_params["temperature"] = value
            elif param == "tools":
                optional_params["tools"] = value
        return optional_params

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
        api_key = api_key or get_secret_str("COMMANDCODE_API_KEY")
        if not api_key:
            raise CommandCodeError(
                status_code=401,
                message="Missing Command Code API key. Set the COMMANDCODE_API_KEY env var or pass api_key.",
            )
        headers.update(get_command_code_headers(api_key))
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        base = api_base or get_secret_str("COMMANDCODE_API_BASE") or COMMAND_CODE_API_BASE
        return f"{base.rstrip('/')}/alpha/generate"

    def _strip_model_prefix(self, model: str) -> str:
        if model.startswith("command_code/"):
            return model.split("/", 1)[1]
        return model

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """Build the nested config/params request body.

        The outer ``config`` block is CLI-workspace metadata that a gateway
        has no meaningful values for, so neutral defaults are sent.
        ``params.stream`` is always true: the generation endpoint streams
        events, and the non-streaming LiteLLM path assembles them in
        transform_response.
        """
        system_text, chat_messages = flatten_system_messages(messages)

        params = CommandCodeParamsBlock(
            model=self._strip_model_prefix(model),
            messages=transform_messages_to_command_code(chat_messages),
            tools=transform_tools_to_command_code(optional_params.get("tools") or []),
            system=system_text,
            max_tokens=optional_params.get("max_tokens") or COMMAND_CODE_DEFAULT_MAX_TOKENS,
            stream=True,
        )
        if "temperature" in optional_params:
            params["temperature"] = optional_params["temperature"]

        request_body = CommandCodeRequestBody(
            config=CommandCodeConfigBlock(
                workingDir="/tmp",
                date=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
                environment="terminal",
                structure=[],
                isGitRepo=False,
                currentBranch="",
                mainBranch="",
                gitStatus="",
                recentCommits=[],
            ),
            memory=None,
            taste=None,
            skills=None,
            threadId=str(uuid.uuid4()),
            params=params,
        )
        return dict(request_body)

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
        """Assemble a single ModelResponse from the buffered event stream.

        The generation endpoint always streams newline-delimited events;
        for the non-streaming path the full body is drained here.
        """
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ChatCompletionMessageToolCall] = []
        finish_reason = "stop"
        usage = None

        for line in raw_response.text.splitlines():
            event = parse_stream_event_line(line)
            if event is None:
                continue
            event_type = event.get("type")
            if event_type == "text-delta":
                text_parts.append(event.get("text") or "")
            elif event_type == "reasoning-delta":
                reasoning_parts.append(event.get("text") or "")
            elif event_type == "tool-call":
                arguments = event.get("input")
                if arguments is None:
                    arguments = event.get("args")
                if arguments is None:
                    arguments = event.get("arguments")
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=event.get("toolCallId") or "",
                        type="function",
                        function=Function(
                            name=event.get("toolName") or "",
                            arguments=json.dumps(parse_tool_call_input(arguments)),
                        ),
                    )
                )
            elif event_type == "finish":
                finish_reason = map_command_code_finish_reason(event.get("finishReason"))
                total_usage = event.get("totalUsage")
                if isinstance(total_usage, dict):
                    usage = usage_from_finish_event(total_usage)
            elif event_type == "error":
                error = event.get("error")
                if isinstance(error, dict):
                    error_message = error.get("message") or "Command Code stream error"
                elif isinstance(error, str):
                    error_message = error
                else:
                    error_message = "Command Code stream error"
                raise CommandCodeError(status_code=raw_response.status_code, message=error_message)

        message = Message(
            role="assistant",
            content="".join(text_parts) if text_parts else None,
            reasoning_content="".join(reasoning_parts) if reasoning_parts else None,
            tool_calls=tool_calls or None,
        )
        model_response.choices = [Choices(index=0, message=message, finish_reason=finish_reason)]
        model_response.model = model
        if usage is not None:
            setattr(model_response, "usage", usage)
        return model_response

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return CommandCodeError(status_code=status_code, message=error_message, headers=headers)

    @property
    def has_custom_stream_wrapper(self) -> bool:
        return True

    @property
    def supports_stream_param_in_request_body(self) -> bool:
        """Streaming is set via ``params.stream`` inside transform_request,
        not as a top-level request body field."""
        return False

    def should_fake_stream(
        self,
        model: str | None,
        stream: bool | None,
        custom_llm_provider: str | None = None,
    ) -> bool:
        """Command Code has native streaming support."""
        return False

    def get_streaming_response(
        self,
        model: str,
        raw_response: httpx.Response,
    ) -> CommandCodeSSEStreamIterator:
        return CommandCodeSSEStreamIterator(response=raw_response, model=model)

    def get_sync_custom_stream_wrapper(
        self,
        model: str,
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        api_base: str,
        headers: dict,
        data: dict,
        messages: list,
        client: HTTPHandler | AsyncHTTPHandler | None = None,
        json_mode: bool | None = None,
        signed_json_body: bytes | None = None,
    ) -> CustomStreamWrapper:
        from litellm.llms.custom_httpx.http_handler import (
            HTTPHandler,
            _get_httpx_client,
        )
        from litellm.utils import CustomStreamWrapper

        if client is None or not isinstance(client, HTTPHandler):
            client = _get_httpx_client(params={})

        verbose_logger.debug(f"Making sync streaming request to: {api_base}")

        response = client.post(
            api_base,
            headers=headers,
            data=json.dumps(data),
            stream=True,
            logging_obj=logging_obj,
        )

        if response.status_code != 200:
            raise CommandCodeError(status_code=response.status_code, message=str(response.read()))

        completion_stream = self.get_streaming_response(model=model, raw_response=response)

        streaming_response = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
        )

        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response="first stream response received",
            additional_args={"complete_input_dict": data},
        )

        return streaming_response

    async def get_async_custom_stream_wrapper(
        self,
        model: str,
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        api_base: str,
        headers: dict,
        data: dict,
        messages: list,
        client: AsyncHTTPHandler | None = None,
        json_mode: bool | None = None,
        signed_json_body: bytes | None = None,
    ) -> CustomStreamWrapper:
        from litellm.llms.custom_httpx.http_handler import (
            AsyncHTTPHandler,
            get_async_httpx_client,
        )
        from litellm.types.utils import LlmProviders
        from litellm.utils import CustomStreamWrapper

        if client is None or not isinstance(client, AsyncHTTPHandler):
            client = get_async_httpx_client(llm_provider=LlmProviders.COMMAND_CODE, params={})

        verbose_logger.debug(f"Making async streaming request to: {api_base}")

        response = await client.post(
            api_base,
            headers=headers,
            data=json.dumps(data),
            stream=True,
            logging_obj=logging_obj,
        )

        if response.status_code != 200:
            raise CommandCodeError(status_code=response.status_code, message=str(await response.aread()))

        completion_stream = self.get_streaming_response(model=model, raw_response=response)

        streaming_response = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
        )

        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response="first stream response received",
            additional_args={"complete_input_dict": data},
        )

        return streaming_response
