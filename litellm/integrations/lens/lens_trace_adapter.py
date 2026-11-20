"""What a gateway trace needs so Lens shows an agent's run the way the agent's own trace would.

The gateway only sees model calls. The agent's name comes from the caller's user agent, and a tool step is
rebuilt from the tool result a model call carries in its prompt, paired by call id with the tool call an
earlier reply made.
"""

import asyncio
import json
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from typing import Final, TypeVar

from opentelemetry.context import Context
from opentelemetry.sdk.trace import Span as RecordingSpan
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import Span
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource
from typing_extensions import override

import litellm
from litellm.integrations.lens.policy import lens_logging_enabled_for_call
from litellm.integrations.lens.remote import LensConnection
from litellm.integrations.otel.logger import OpenTelemetryV2
from litellm.integrations.otel.model.config import ExporterSpec, OpenTelemetryV2Config
from litellm.integrations.otel.model.metadata import LLMCallEvent
from litellm.integrations.otel.model.utils import to_ns
from litellm.types.utils import CaptureMessageContent

_KNOWN_AGENTS: Final[tuple[tuple[str, str], ...]] = (
    ("claude-cli", "claude-code"),
    ("claude-code", "claude-code"),
    ("codex", "codex"),
)


def agent_name(user_agent: str | None) -> str | None:
    if not user_agent:
        return None
    lowered: Final = user_agent.strip().lower()
    known: Final = next((name for marker, name in _KNOWN_AGENTS if marker in lowered), None)
    return known or lowered.split("/")[0].split(" ")[0] or None


@dataclass(frozen=True, slots=True)
class ToolExchange:
    call_id: str
    name: str
    arguments: str
    result: str


class _Function(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    name: str | None = None
    arguments: object = None


class _ToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str | None = None
    function: _Function = _Function()


class _Block(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    type: str | None = None
    id: str | None = None
    name: str | None = None
    input: object = None
    tool_use_id: str | None = None
    content: object = None


class _Item(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    type: str | None = None
    role: str | None = None
    call_id: str | None = None
    name: str | None = None
    arguments: object = None
    output: object = None
    tool_call_id: str | None = None
    tool_calls: tuple[_ToolCall, ...] | None = None
    content: object = None


_Model: Final = TypeVar("_Model", bound=BaseModel)
_ITEMS: Final[TypeAdapter[tuple[object, ...]]] = TypeAdapter(tuple[object, ...])
_BLOCKS: Final[TypeAdapter[tuple[object, ...]]] = TypeAdapter(tuple[object, ...])


@dataclass(frozen=True, slots=True)
class _Turn:
    calls: tuple[tuple[str, str, str], ...] = ()
    results: tuple[tuple[str, str], ...] = ()
    from_model: bool = False


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _blocks(content: object) -> tuple[_Block, ...]:
    if not isinstance(content, list):
        return ()
    try:
        raw: Final = _BLOCKS.validate_python(content)
    except ValidationError:
        return ()
    return tuple(block for block in (_parsed(_Block, item) for item in raw) if block is not None)


def _parsed(model: type[_Model], value: object) -> _Model | None:
    try:
        return model.model_validate(value)
    except ValidationError:
        return None


def _turn(item: _Item) -> _Turn:
    if item.type == "function_call":
        calls: Final = ((item.call_id, item.name or "", _text(item.arguments)),) if item.call_id else ()
        return _Turn(calls=calls, from_model=True)
    if item.type == "function_call_output":
        return _Turn(results=((item.call_id, _text(item.output)),) if item.call_id else ())
    blocks: Final = _blocks(item.content)
    if item.role == "assistant":
        return _Turn(
            calls=(
                *(
                    (call.id, call.function.name or "", _text(call.function.arguments))
                    for call in item.tool_calls or ()
                    if call.id
                ),
                *(
                    (block.id, block.name or "", _text(block.input))
                    for block in blocks
                    if block.type == "tool_use" and block.id
                ),
            ),
            from_model=True,
        )
    if item.role == "tool":
        return _Turn(results=((item.tool_call_id, _text(item.content)),) if item.tool_call_id else ())
    return _Turn(
        results=tuple(
            (block.tool_use_id, _text(block.content))
            for block in blocks
            if block.type == "tool_result" and block.tool_use_id
        )
    )


def tool_exchanges(messages: object) -> tuple[ToolExchange, ...]:
    """The tool results that arrived since the model's last turn, each with the call that asked for it.

    A prompt repeats the whole conversation, so only the results after the last model turn are new to this
    call. Earlier ones were reported by the call that first carried them.
    """
    if not isinstance(messages, list):
        return ()
    try:
        raw: Final = _ITEMS.validate_python(messages)
    except ValidationError:
        return ()
    items: Final = tuple(_parsed(_Item, value) for value in raw)
    turns: Final = tuple(_turn(item) for item in items if item is not None)
    last_model_turn: Final = max((index for index, turn in enumerate(turns) if turn.from_model), default=-1)
    calls: Final[Mapping[str, tuple[str, str]]] = {
        call_id: (name, arguments) for call_id, name, arguments in chain.from_iterable(turn.calls for turn in turns)
    }
    results: Final = chain.from_iterable(turn.results for turn in turns[last_model_turn + 1 :])
    answered: Final = tuple(answer for answer in results if answer[0] in calls)
    return tuple(
        ToolExchange(call_id=call_id, name=calls[call_id][0] or "tool", arguments=calls[call_id][1], result=result)
        for call_id, result in answered
    )


class _Metadata(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    user_agent: str | None = None


class _Payload(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    metadata: _Metadata = _Metadata()
    messages: object = None


@dataclass(frozen=True, slots=True)
class AgentCall:
    agent: str | None = None
    tool_exchanges: tuple[ToolExchange, ...] = ()


def agent_call(standard_logging_payload: object) -> AgentCall:
    payload: Final = _parsed(_Payload, standard_logging_payload)
    if payload is None:
        return AgentCall()
    return AgentCall(agent=agent_name(payload.metadata.user_agent), tool_exchanges=tool_exchanges(payload.messages))


_AGENT_NAME: Final = "gen_ai.agent.name"
_SESSION_ATTRIBUTE: Final = "gen_ai.conversation.id"
_TOOL_STEP_TRACER: Final = "litellm.lens.tool_steps"
_CALLING_AGENT: Final[ContextVar[str | None]] = ContextVar("lens_calling_agent", default=None)
LENS_TRACE_PATH: Final = "/v1/traces"


LENS_TRACE_CALLBACK: Final = "lens"


class _LensTraceConfig(OpenTelemetryV2Config):
    """Built from its arguments alone, so the OTEL_* variables a gateway sets for its own tracing never reach Lens."""

    @classmethod
    @override
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)


class _LensTraceLogger(OpenTelemetryV2):
    """Model-call spans only, on a private provider, so a gateway's own OTel callbacks are left as they are."""

    @override
    def _init_otel_logger_on_litellm_proxy(self) -> None:
        return None

    @override
    def log_pre_api_call(self, model: str, messages: object, kwargs: Mapping[str, object]) -> None:
        if lens_logging_enabled_for_call(kwargs):
            super().log_pre_api_call(model, messages, kwargs)

    @override
    def _close_llm_call(
        self,
        kwargs: Mapping[str, object],
        start_time: datetime | float | None,
        end_time: datetime | float | None,
    ) -> Span | None:
        call: Final = LLMCallEvent.from_dict(kwargs)
        if not lens_logging_enabled_for_call(kwargs):
            self._release_carrier(self._open_llm_calls.pop(call.call_id, None) if call.call_id else None)
            return None
        made_by: Final = agent_call(call.payload)
        carrier: Final = self._open_llm_calls.get(call.call_id) if call.call_id else None
        if made_by.agent and carrier is not None and carrier.span is not None:
            carrier.span.set_attribute(_AGENT_NAME, made_by.agent)
        token: Final = _CALLING_AGENT.set(made_by.agent)
        try:
            span: Final = super()._close_llm_call(kwargs, start_time, end_time)
        finally:
            _CALLING_AGENT.reset(token)
        if span is not None and call.session_id:
            self._emit_tool_steps(made_by, call.session_id, to_ns(start_time))
        return span

    def _emit_tool_steps(self, made_by: AgentCall, session_id: str, at_ns: int | None) -> None:
        """One zero-length step per tool result, at the start of the model call whose prompt carries it."""
        tracer: Final = self.tracer_provider.get_tracer(_TOOL_STEP_TRACER)
        for exchange in made_by.tool_exchanges:
            tracer.start_span(
                f"execute_tool {exchange.name}",
                start_time=at_ns,
                attributes={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": exchange.name,
                    "gen_ai.tool.call.id": exchange.call_id,
                    "gen_ai.tool.call.arguments": exchange.arguments,
                    "gen_ai.tool.call.result": exchange.result,
                    _SESSION_ATTRIBUTE: session_id,
                    **({_AGENT_NAME: made_by.agent} if made_by.agent else {}),
                },
            ).end(end_time=at_ns)


class _AgentNameProcessor(SpanProcessor):
    """Names the calling agent on a model-call span that is only created when the call closes."""

    @override
    def on_start(self, span: RecordingSpan, parent_context: Context | None = None) -> None:
        agent: Final = _CALLING_AGENT.get()
        if agent:
            span.set_attribute(_AGENT_NAME, agent)


@asynccontextmanager
async def export_traces(connection: LensConnection) -> AsyncGenerator[None, None]:
    trace_logger: Final = _LensTraceLogger(
        config=_LensTraceConfig(
            exporters=[
                ExporterSpec(
                    kind="otlp_http",
                    traces_endpoint=connection.endpoint(LENS_TRACE_PATH),
                    headers=f"Authorization=Bearer {connection.token}",
                    resource_attributes={"lens.session.capture": "true", "lens.session.attribute": _SESSION_ATTRIBUTE},
                )
            ],
            capture_message_content=CaptureMessageContent.SPAN_ONLY,
        ),
        callback_name=LENS_TRACE_CALLBACK,
    )
    trace_logger.tracer_provider.add_span_processor(_AgentNameProcessor())
    manager: Final = litellm.logging_callback_manager
    manager.add_litellm_callback(trace_logger)
    try:
        yield None
    finally:
        manager.remove_callback_from_all_lists(trace_logger)
        await asyncio.to_thread(trace_logger.tracer_provider.shutdown)
