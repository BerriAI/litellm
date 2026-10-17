import os
from collections.abc import Mapping
from typing import Final, Literal

from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.deepeval.api import Api, Endpoints, HttpMethods
from litellm.integrations.deepeval.types import (
    BaseApiSpan,
    SpanApiType,
    TraceApi,
    TraceSpanApiStatus,
)
from litellm.integrations.deepeval.utils import (
    to_zod_compatible_iso,
    validate_environment,
)

_STR_MAPPING: Final = TypeAdapter(Mapping[str, object])
_OBJECT_TUPLE: Final = TypeAdapter(tuple[object, ...])
_REDACTED: Final = "redacted-by-litellm"
_TEXT_PART_TYPES: Final = frozenset({"text", "output_text"})
_RESPONSES_TOOL_CALL_TYPES: Final = frozenset({"function_call", "custom_tool_call"})


class _ToolFunction(TypedDict):
    name: ReadOnly[str]
    arguments: ReadOnly[str]


class _ToolCall(TypedDict):
    id: ReadOnly[str]
    type: ReadOnly[Literal["function"]]
    function: ReadOnly[_ToolFunction]


class _AssistantMessage(TypedDict):
    role: ReadOnly[Literal["assistant"]]
    content: ReadOnly[str | None]
    tool_calls: ReadOnly[tuple[_ToolCall, ...]]


def _as_mapping(value: object) -> Mapping[str, object] | None:
    return _STR_MAPPING.validate_python(value) if isinstance(value, dict) else None


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _dict_items(value: object) -> tuple[Mapping[str, object], ...]:
    items: Final = _OBJECT_TUPLE.validate_python(value) if isinstance(value, list) else ()
    return tuple(mapping for item in items if (mapping := _as_mapping(item)) is not None)


def _part_text(part: Mapping[str, object]) -> str | None:
    return _as_str(part.get("text")) if part.get("type") in _TEXT_PART_TYPES else None


def _content_text(content: object) -> str:
    """A plain string as is, or the joined text parts of a content-part list."""
    if isinstance(content, str):
        return content
    return "".join(text for part in _dict_items(content) if (text := _part_text(part)) is not None)


def _tool_call(call_id: object, name: object, arguments: object, redact: bool) -> _ToolCall:
    function: Final[_ToolFunction] = {
        "name": _as_str(name) or "",
        "arguments": _REDACTED if redact else (_as_str(arguments) or ""),
    }
    tool_call: Final[_ToolCall] = {"id": _as_str(call_id) or "", "type": "function", "function": function}
    return tool_call


def _chat_tool_call(tool_call: Mapping[str, object], redact: bool) -> _ToolCall:
    function: Final = _as_mapping(tool_call.get("function")) or {}
    return _tool_call(tool_call.get("id"), function.get("name"), function.get("arguments"), redact)


def _responses_tool_call(item: Mapping[str, object], redact: bool) -> _ToolCall:
    arguments_key: Final = "input" if item.get("type") == "custom_tool_call" else "arguments"
    call_id: Final = item.get("call_id") or item.get("id")
    return _tool_call(call_id, item.get("name"), item.get(arguments_key), redact)


def _span_output(text: str, tool_calls: tuple[_ToolCall, ...], redact: bool) -> _AssistantMessage | str:
    span_text: Final = _REDACTED if redact and text else text
    if not tool_calls:
        return span_text
    message: Final[_AssistantMessage] = {"role": "assistant", "content": span_text or None, "tool_calls": tool_calls}
    return message


def _chat_output(choice: Mapping[str, object], redact: bool) -> _AssistantMessage | str | None:
    message: Final = _as_mapping(choice.get("message")) or {}
    content: Final = message.get("content")
    raw_tool_calls: Final = _dict_items(message.get("tool_calls"))
    tool_calls: Final = tuple(_chat_tool_call(tool_call, redact) for tool_call in raw_tool_calls)
    if content is None and not tool_calls:
        return None
    return _span_output(_content_text(content), tool_calls, redact)


def _responses_output(output_items: tuple[Mapping[str, object], ...], redact: bool) -> _AssistantMessage | str:
    messages: Final = tuple(item for item in output_items if item.get("type") == "message")
    message_texts: Final = tuple(_content_text(item.get("content")) for item in messages)
    tool_items: Final = tuple(item for item in output_items if item.get("type") in _RESPONSES_TOOL_CALL_TYPES)
    tool_calls: Final = tuple(_responses_tool_call(item, redact) for item in tool_items)
    text: Final = "\n\n".join(message_text for message_text in message_texts if message_text)
    return _span_output(text, tool_calls, redact) if text or tool_calls else "NO_OUTPUT"


def _success_output(response: object, redact: bool) -> _AssistantMessage | str | None:
    """Chat completions read ``choices``; Responses API payloads read ``output``."""
    response_dict: Final = _as_mapping(response) or {}
    choices: Final = _dict_items(response_dict.get("choices"))
    if choices:
        return _chat_output(choices[0], redact)
    output_items: Final = _dict_items(response_dict.get("output"))
    return _responses_output(output_items, redact) if output_items else "NO_OUTPUT"


# This file includes the custom callbacks for LiteLLM Proxy
# Once defined, these can be passed in proxy_config.yaml
class DeepEvalLogger(CustomLogger):
    """Logs litellm traces to DeepEval's platform."""

    def __init__(self, *args, **kwargs):
        api_key: Final = os.getenv("CONFIDENT_API_KEY")
        self.litellm_environment = os.getenv("LITELM_ENVIRONMENT", "development")
        validate_environment(self.litellm_environment)
        if not api_key:
            raise ValueError("Please set 'CONFIDENT_API_KEY=<>' in your environment variables.")
        self.api = Api(api_key=api_key)
        super().__init__(*args, **kwargs)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Logs a success event to DeepEval's platform."""
        self._sync_event_handler(kwargs, response_obj, start_time, end_time, is_success=True)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Logs a failure event to DeepEval's platform."""
        self._sync_event_handler(kwargs, response_obj, start_time, end_time, is_success=False)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Logs a failure event to DeepEval's platform."""
        await self._async_event_handler(kwargs, response_obj, start_time, end_time, is_success=False)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Logs a success event to DeepEval's platform."""
        await self._async_event_handler(kwargs, response_obj, start_time, end_time, is_success=True)

    def _prepare_trace_api(self, kwargs, response_obj, start_time, end_time, is_success):
        _start_time: Final = to_zod_compatible_iso(start_time)
        _end_time: Final = to_zod_compatible_iso(end_time)
        _standard_logging_object: Final = kwargs.get("standard_logging_object", {})
        base_api_span: Final = self._create_base_api_span(
            kwargs,
            standard_logging_object=_standard_logging_object,
            start_time=_start_time,
            end_time=_end_time,
            is_success=is_success,
        )
        trace_api: Final = self._create_trace_api(
            base_api_span,
            standard_logging_object=_standard_logging_object,
            start_time=_start_time,
            end_time=_end_time,
            litellm_environment=self.litellm_environment,
        )

        body = {}

        try:
            body = trace_api.model_dump(by_alias=True, exclude_none=True)
        except AttributeError:
            # Pydantic version below 2.0
            body = trace_api.dict(by_alias=True, exclude_none=True)
        return body

    def _sync_event_handler(self, kwargs, response_obj, start_time, end_time, is_success):
        body: Final = self._prepare_trace_api(kwargs, response_obj, start_time, end_time, is_success)
        try:
            response: Final = self.api.send_request(
                method=HttpMethods.POST,
                endpoint=Endpoints.TRACING_ENDPOINT,
                body=body,
            )
        except Exception as e:
            raise e
        verbose_logger.debug("DeepEvalLogger: sync_log_failure_event: Api response %s", response)

    async def _async_event_handler(self, kwargs, response_obj, start_time, end_time, is_success):
        body: Final = self._prepare_trace_api(kwargs, response_obj, start_time, end_time, is_success)
        response: Final = await self.api.a_send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.TRACING_ENDPOINT,
            body=body,
        )

        verbose_logger.debug("DeepEvalLogger: async_event_handler: Api response %s", response)

    def _create_base_api_span(self, kwargs, standard_logging_object, start_time, end_time, is_success):
        # extract usage
        usage: Final = standard_logging_object.get("response", {}).get("usage", {})
        if is_success:
            output = _success_output(standard_logging_object.get("response", {}), redact=self.turn_off_message_logging)
        else:
            output = str(standard_logging_object.get("error_string", ""))
        return BaseApiSpan(
            uuid=standard_logging_object.get("id", uuid.uuid4()),
            name=("litellm_success_callback" if is_success else "litellm_failure_callback"),
            status=(TraceSpanApiStatus.SUCCESS if is_success else TraceSpanApiStatus.ERRORED),
            type=SpanApiType.LLM,
            traceUuid=standard_logging_object.get("trace_id", uuid.uuid4()),
            startTime=str(start_time),
            endTime=str(end_time),
            input=kwargs.get("input", "NO_INPUT"),
            output=output,
            model=standard_logging_object.get("model", None),
            inputTokenCount=usage.get("prompt_tokens", None) if is_success else None,
            outputTokenCount=(usage.get("completion_tokens", None) if is_success else None),
        )

    def _create_trace_api(
        self,
        base_api_span,
        standard_logging_object,
        start_time,
        end_time,
        litellm_environment,
    ):
        return TraceApi(
            uuid=standard_logging_object.get("trace_id", uuid.uuid4()),
            baseSpans=[],
            agentSpans=[],
            llmSpans=[base_api_span],
            retrieverSpans=[],
            toolSpans=[],
            startTime=str(start_time),
            endTime=str(end_time),
            environment=litellm_environment,
        )
