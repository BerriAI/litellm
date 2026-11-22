"""What a gateway trace needs so Lens shows an agent's run the way the agent's own trace would.

The gateway only sees model calls. The agent's name comes from the caller's user agent, or from its system prompt
when the agent sends its provider SDK's user agent. The task is the text the user typed, without the setup blocks
an agent puts in front of it. A tool step is rebuilt from the tool result a model call carries in its prompt, paired
by call id with the tool call an earlier reply made. A known agent that sends no session id gets one derived from
its key, system prompt and task, so its model calls are one run.

An agent that traces itself names the span each model call belongs to in the request's ``traceparent``. The gateway
puts its model-call span under that span, in the agent's own trace, and rebuilds no tool steps there: the agent's
spans, which it sends through the gateway's ``/v1/traces``, already hold them.
"""

import asyncio
import hashlib
import json
import re
import time
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from types import MappingProxyType
from typing import Final, TypeVar

from opentelemetry.context import Context
from opentelemetry.sdk.trace import Span as RecordingSpan
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import Span, get_current_span
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource
from typing_extensions import override

import litellm
from litellm.integrations.lens.remote import LensConnection
from litellm.integrations.otel.logger import OpenTelemetryV2
from litellm.integrations.otel.model.config import ExporterSpec, OpenTelemetryV2Config
from litellm.integrations.otel.model.metadata import LLMCallEvent, RequestIdentity
from litellm.integrations.otel.model.utils import to_ns
from litellm.integrations.otel.plumbing.context import extract_traceparent
from litellm.litellm_core_utils.harness_context import (
    CODEX_MARKERS,
    SYSTEM_REMINDER_MARKERS,
    MarkerPairs,
    without_harness_context,
)
from litellm.types.utils import CaptureMessageContent


@dataclass(frozen=True, slots=True)
class _Harness:
    """A coding agent Lens knows: one row here is all a new one needs."""

    name: str
    integration: str
    context_markers: MarkerPairs
    user_agent_markers: tuple[str, ...] = ()


_HARNESSES: Final = (
    _Harness(
        name="claude-code",
        integration="claude-code",
        context_markers=SYSTEM_REMINDER_MARKERS,
        user_agent_markers=("claude-cli", "claude-code"),
    ),
    _Harness(
        name="codex-cli", integration="openai-codex", context_markers=CODEX_MARKERS, user_agent_markers=("codex",)
    ),
    _Harness(name="opencode", integration="opencode", context_markers=SYSTEM_REMINDER_MARKERS),
    _Harness(name="pi", integration="pi", context_markers=SYSTEM_REMINDER_MARKERS),
)
_INTEGRATIONS: Final[Mapping[str, str]] = MappingProxyType(
    {harness.name: harness.integration for harness in _HARNESSES}
)
_CONTEXT_MARKERS: Final[Mapping[str, MarkerPairs]] = MappingProxyType(
    {harness.name: harness.context_markers for harness in _HARNESSES}
)
_PI_SYSTEM_PROMPT: Final = "operating inside pi, a coding agent harness"
_LEADING_SETUP_BLOCKS: Final = re.compile(r"(?:\s*<([A-Za-z][\w-]*)>.*?</\1>)*", re.DOTALL)
_TASK_CHARS: Final = 2000


def agent_name(user_agent: str | None) -> str | None:
    if not user_agent:
        return None
    lowered: Final = user_agent.strip().lower()
    known: Final = next(
        (harness.name for harness in _HARNESSES if any(marker in lowered for marker in harness.user_agent_markers)),
        None,
    )
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
    text: str | None = None


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
_FIELDS: Final = TypeAdapter(dict[str, object])


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


def _texts(content: object) -> tuple[str, ...]:
    if isinstance(content, str):
        return (content,)
    return tuple(block.text for block in _blocks(content) if block.text)


def _without_setup_blocks(text: str) -> str:
    """The text after every leading ``<tag>...</tag>`` block, which is how agents send their setup."""
    return _LEADING_SETUP_BLOCKS.sub("", text, count=1).strip()


def _conversation(messages: object) -> tuple[_Item, ...]:
    if not isinstance(messages, list):
        return ()
    try:
        raw: Final = _ITEMS.validate_python(messages)
    except ValidationError:
        return ()
    return tuple(item for item in (_parsed(_Item, entry) for entry in raw) if item is not None)


def _role_texts(messages: object, roles: frozenset[str]) -> tuple[str, ...]:
    spoken: Final = tuple(item for item in _conversation(messages) if item.role in roles)
    return tuple(chain.from_iterable(_texts(item.content) for item in spoken))


def user_task(messages: object) -> str | None:
    """What the user typed: the first user text that is more than an agent's setup blocks."""
    typed: Final = (_without_setup_blocks(text) for text in _role_texts(messages, frozenset({"user"})))
    return next((task[:_TASK_CHARS] for task in typed if task), None)


def _system_prompt(messages: object) -> str:
    return "\n".join(_role_texts(messages, frozenset({"system", "developer"})))


def _agent(user_agent: str | None, system_prompt: str) -> str | None:
    named: Final = agent_name(user_agent)
    if named in _INTEGRATIONS:
        return named
    return "pi" if _PI_SYSTEM_PROMPT in system_prompt else named


def _derived_session(key_hash: str | None, system_prompt: str, task: str, first_reply: str | None) -> str:
    parts: Final = (key_hash or "", system_prompt, task, first_reply or "")
    digest: Final = hashlib.sha256("\x00".join(parts).encode()).hexdigest()
    return f"derived-{digest[:32]}"


class _Metadata(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    user_agent: str | None = None
    user_api_key_hash: str | None = None


class _Choice(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    message: _Item = _Item()


class _Response(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    choices: tuple[_Choice, ...] = ()


class _Payload(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    metadata: _Metadata = _Metadata()
    messages: object = None
    response: object = None


def _reply_mark(reply: _Item) -> str | None:
    """What tells one conversation's first reply from another's: its first tool call id, else its text."""
    called: Final = tuple(call.id for call in reply.tool_calls or () if call.id)
    used: Final = tuple(block.id for block in _blocks(reply.content) if block.type == "tool_use" and block.id)
    return next(iter((*called, *used)), None) or "".join(_texts(reply.content)).strip() or None


def _first_reply_mark(messages: object, response: object) -> str | None:
    """The first reply of the conversation: in the request once it has one, else this call's response."""
    said: Final = next((item for item in _conversation(messages) if item.role == "assistant"), None)
    if said is not None:
        return _reply_mark(said)
    answered: Final = _parsed(_Response, response)
    return _reply_mark(answered.choices[0].message) if answered and answered.choices else None


@dataclass(frozen=True, slots=True)
class AgentCall:
    agent: str | None = None
    tool_exchanges: tuple[ToolExchange, ...] = ()
    integration: str | None = None
    task: str | None = None
    derived_session: str | None = None

    def attributes(self, session_id: str | None) -> Mapping[str, str]:
        """The facts every span of this call carries; ``session_id`` is the caller's own, which wins."""
        session: Final = None if session_id else self.derived_session
        return {
            **({_AGENT_NAME: self.agent} if self.agent else {}),
            **({_INTEGRATION: self.integration} if self.integration else {}),
            **({_TASK: self.task} if self.task else {}),
            **({_SESSION_ATTRIBUTE: session} if session else {}),
        }


def agent_call(standard_logging_payload: object) -> AgentCall:
    payload: Final = _parsed(_Payload, standard_logging_payload)
    if payload is None:
        return AgentCall()
    system_prompt: Final = _system_prompt(payload.messages)
    agent: Final = _agent(payload.metadata.user_agent, system_prompt)
    integration: Final = _INTEGRATIONS.get(agent) if agent else None
    task: Final = user_task(payload.messages)
    return AgentCall(
        agent=agent,
        tool_exchanges=tool_exchanges(payload.messages),
        integration=integration,
        task=task,
        derived_session=(
            _derived_session(
                payload.metadata.user_api_key_hash,
                system_prompt,
                task,
                _first_reply_mark(payload.messages, payload.response),
            )
            if integration and task
            else None
        ),
    )


def as_user_saw_it(standard_logging_payload: object) -> object:
    """The logging payload with the messages its agent added for the model, and never showed its user, removed."""
    payload: Final = _parsed(_Payload, standard_logging_payload)
    if payload is None or payload.messages is None:
        return standard_logging_payload
    try:
        fields: Final = _FIELDS.validate_python(standard_logging_payload)
    except ValidationError:
        return standard_logging_payload
    markers: Final = _CONTEXT_MARKERS.get(agent_name(payload.metadata.user_agent) or "", SYSTEM_REMINDER_MARKERS)
    return {**fields, "messages": without_harness_context(payload.messages, markers)}


_AGENT_NAME: Final = "gen_ai.agent.name"
_INTEGRATION: Final = "langsmith.metadata.ls_integration"
_TASK: Final = "lens.task"
_SESSION_ATTRIBUTE: Final = "gen_ai.conversation.id"
_TOOL_STEP_TRACER: Final = "litellm.lens.tool_steps"
_CALL_FACTS: Final[ContextVar[Mapping[str, str]]] = ContextVar("lens_call_facts", default=MappingProxyType({}))
LENS_TRACE_PATH: Final = "/v1/traces"


LENS_TRACE_CALLBACK: Final = "lens"


_AGENT_TRACE: Final[ContextVar[Context | None]] = ContextVar("lens_agent_trace", default=None)


class _ProxyRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    headers: Mapping[str, str] | None = None


class _CallParams(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    proxy_server_request: _ProxyRequest = _ProxyRequest()


def agent_trace_of(kwargs: Mapping[str, object]) -> Context | None:
    """Where the calling agent's own trace says this model call belongs, when its request said so."""
    params: Final = _parsed(_CallParams, kwargs.get("litellm_params"))
    named: Final = extract_traceparent(params.proxy_server_request.headers or {}) if params else None
    return named if named is not None and get_current_span(named).get_span_context().is_valid else None


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
        """A call the agent placed in its own trace gets its span at close, under the span the agent named."""
        call: Final = LLMCallEvent.from_dict(kwargs)
        if agent_trace_of(kwargs) is None or call.is_no_upstream_call or call.call_id is None:
            return super().log_pre_api_call(model, messages, kwargs)
        if call.call_id not in self._open_llm_calls:
            self._defer_llm_call(call.call_id, to_ns(call.upstream_start_seconds) or time.time_ns())
        return None

    @override
    def _seed_identity_baggage(self, identity: RequestIdentity, model: str | None, context: Context) -> Context:
        return super()._seed_identity_baggage(identity, model, _AGENT_TRACE.get() or context)

    @override
    def _span_session(self, call: LLMCallEvent) -> str | None:
        """Lens files a span that names a session under that session; one in the agent's trace must stay there."""
        return call.session_id if _AGENT_TRACE.get() is None else None

    @override
    def _close_llm_call(
        self,
        kwargs: Mapping[str, object],
        start_time: datetime | float | None,
        end_time: datetime | float | None,
    ) -> Span | None:
        shown: Final = {**kwargs, "standard_logging_object": as_user_saw_it(kwargs.get("standard_logging_object"))}
        call: Final = LLMCallEvent.from_dict(shown)
        made_by: Final = agent_call(call.payload)
        carrier: Final = self._open_llm_calls.get(call.call_id) if call.call_id else None
        agent_trace: Final = agent_trace_of(kwargs)
        named: Final = made_by.attributes(call.session_id)
        facts: Final = (
            named
            if agent_trace is None
            else MappingProxyType({key: value for key, value in named.items() if key != _SESSION_ATTRIBUTE})
        )
        if carrier is not None and carrier.span is not None:
            carrier.span.set_attributes(facts)
        token: Final = _CALL_FACTS.set(facts)
        placed: Final = _AGENT_TRACE.set(agent_trace)
        try:
            span: Final = super()._close_llm_call(shown, start_time, end_time)
        finally:
            _AGENT_TRACE.reset(placed)
            _CALL_FACTS.reset(token)
        session_id: Final = call.session_id or made_by.derived_session
        if span is not None and session_id and agent_trace is None:
            self._emit_tool_steps(made_by, session_id, to_ns(start_time))
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
                    **made_by.attributes(session_id),
                    _SESSION_ATTRIBUTE: session_id,
                },
            ).end(end_time=at_ns)


class _CallFactsProcessor(SpanProcessor):
    """Puts the calling agent's facts on a model-call span that is only created when the call closes."""

    @override
    def on_start(self, span: RecordingSpan, parent_context: Context | None = None) -> None:
        span.set_attributes(_CALL_FACTS.get())


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
    trace_logger.tracer_provider.add_span_processor(_CallFactsProcessor())
    manager: Final = litellm.logging_callback_manager
    manager.add_litellm_callback(trace_logger)
    try:
        yield None
    finally:
        manager.remove_callback_from_all_lists(trace_logger)
        await asyncio.to_thread(trace_logger.tracer_provider.shutdown)
