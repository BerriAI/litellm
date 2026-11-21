"""What a gateway trace needs so Lens shows an agent's run the way the agent's own trace would.

The gateway only sees model calls. The agent's name comes from the caller's user agent, or from its system prompt
when the agent sends its provider SDK's user agent. The task is the text the user typed, without the setup blocks
an agent puts in front of it. A tool step is rebuilt from the tool result a model call carries in its prompt, paired
by call id with the tool call an earlier reply made. A known agent that sends no session id gets one derived from
its key, system prompt and task, so its model calls are one run.

A coding agent's own trace is one trace per message the user typed: an interaction span that holds the message,
with the model calls and tool steps that answer it underneath, and a subagent's calls under the tool call that
launched it. A gateway trace of the same agent is given that shape, so Lens shows both the same way.
"""

import asyncio
import hashlib
import json
import re
from collections import OrderedDict
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeVar

from opentelemetry.context import Context
from opentelemetry.sdk.trace import Span as RecordingSpan
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.trace import (
    INVALID_SPAN,
    NonRecordingSpan,
    Span,
    SpanContext,
    TraceFlags,
    set_span_in_context,
)
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource
from typing_extensions import override

import litellm
from litellm.integrations.lens.remote import LensConnection
from litellm.integrations.otel.logger import OpenTelemetryV2, _LLMCallSpan
from litellm.integrations.otel.model.config import ExporterSpec, OpenTelemetryV2Config
from litellm.integrations.otel.model.metadata import LLMCallEvent, RequestIdentity
from litellm.integrations.otel.model.utils import to_ns
from litellm.types.utils import CaptureMessageContent

if TYPE_CHECKING:
    from litellm.caching.redis_cache import RedisCache


@dataclass(frozen=True, slots=True)
class _Harness:
    """A coding agent Lens knows: one row here is all a new one needs."""

    name: str
    integration: str
    user_agent_markers: tuple[str, ...] = ()


_HARNESSES: Final = (
    _Harness(name="claude-code", integration="claude-code", user_agent_markers=("claude-cli", "claude-code")),
    _Harness(name="codex-cli", integration="openai-codex", user_agent_markers=("codex",)),
    _Harness(name="opencode", integration="opencode"),
    _Harness(name="pi", integration="pi"),
)
_INTEGRATIONS: Final[Mapping[str, str]] = MappingProxyType(
    {harness.name: harness.integration for harness in _HARNESSES}
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


def _typed(item: _Item) -> str:
    """What the user wrote in one message, empty for tool results and for an agent's setup blocks."""
    if item.role != "user":
        return ""
    return "\n\n".join(text for text in (_without_setup_blocks(part) for part in _texts(item.content)) if text)


@dataclass(frozen=True, slots=True)
class Interaction:
    """One typed message and the work that answers it, named after the span an agent's own trace gives it."""

    key: str
    sequence: int
    prompt: str
    opens: bool

    @property
    def trace_id(self) -> int:
        return int(self.key, 16)

    @property
    def span_id(self) -> int:
        return int(self.key[:16], 16) or 1


def interaction(session_id: str, messages: object) -> Interaction | None:
    """The interaction a model call belongs to, read from its request alone so every gateway worker agrees.

    A prompt repeats the whole conversation, so the newest typed message is the one being answered and the
    number of typed messages is its place in the session. The call opens the interaction when that message
    comes after the model's last turn, and continues it when only tool results do.
    """
    items: Final = _conversation(messages)
    typed: Final = tuple((index, text) for index, item in enumerate(items) if (text := _typed(item)))
    if not typed:
        return None
    last_model_turn: Final = max((index for index, item in enumerate(items) if _turn(item).from_model), default=-1)
    index, prompt = typed[-1]
    digest: Final = hashlib.sha256("\x00".join((session_id, str(len(typed)), prompt)).encode()).hexdigest()
    return Interaction(key=digest[:32], sequence=len(typed), prompt=prompt, opens=index > last_model_turn)


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
    requester_custom_headers: Mapping[str, str] | None = None


class _Parameters(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    tools: tuple[object, ...] | None = None


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
    model_parameters: object = None


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
    request: object = field(default=None, compare=False, repr=False)
    subagent: str | None = None
    """The id a subagent's requests carry, absent on the agent's own thread."""
    has_tools: bool = False

    @property
    def beside_thread(self) -> bool:
        """A request the agent makes beside its thread, such as naming the session or suggesting a reply:
        it declares no tools, or its newest message is one the agent wrote for that purpose."""
        newest: Final = next((text for item in reversed(_conversation(self.request)) if (text := _typed(item))), "")
        return not self.has_tools or newest.startswith(_SIDE_REQUEST_PREFIXES)

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
    parameters: Final = _parsed(_Parameters, payload.model_parameters)
    return AgentCall(
        request=payload.messages,
        subagent=(payload.metadata.requester_custom_headers or {}).get(_SUBAGENT_HEADER) or None,
        has_tools=bool(parameters and parameters.tools),
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


_SUBAGENT_HEADER: Final = "x-claude-code-agent-id"
_SIDE_REQUEST_PREFIXES: Final = ("[SUGGESTION MODE:",)
_AGENT_NAME: Final = "gen_ai.agent.name"
_INTEGRATION: Final = "langsmith.metadata.ls_integration"
_TASK: Final = "lens.task"
_SESSION_ATTRIBUTE: Final = "gen_ai.conversation.id"
_TRACE_KEY: Final = "lens.trace.key"
_ACTOR: Final = "agent_id"
_INTERACTION_SPAN: Final[Mapping[str, str]] = MappingProxyType({"claude-code": "claude_code.interaction"})
_REMEMBERED_SESSIONS: Final = 2048
_REMEMBERED_STEPS: Final = 64
_TOOL_STEP_TRACER: Final = "litellm.lens.tool_steps"
_CALL_FACTS: Final[ContextVar[Mapping[str, str]]] = ContextVar("lens_call_facts", default=MappingProxyType({}))
_CALL_PARENT: Final[ContextVar[SpanContext | None]] = ContextVar("lens_call_parent", default=None)
_NEXT_SPAN_ID: Final[ContextVar[int | None]] = ContextVar("lens_next_span_id", default=None)


class _InteractionIds(RandomIdGenerator):
    """Gives an interaction span the id its model calls were told to parent to."""

    @override
    def generate_span_id(self) -> int:
        return _NEXT_SPAN_ID.get() or super().generate_span_id()


@dataclass(frozen=True, slots=True)
class _Open:
    """The interaction a session is in, with the tool steps reported for it so far."""

    interaction: Interaction
    steps: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True, slots=True)
class _Placement:
    """Where a call goes: the trace Lens files it under and the span it hangs from."""

    key: str | None = None
    parent: SpanContext | None = None
    actor: str | None = None
    opening: Interaction | None = None
    interaction: Interaction | None = None


_OPEN: Final[TypeAdapter[_Open]] = TypeAdapter(_Open)
_SHARED_KEY: Final = "lens:interaction:{session}"
_SHARED_SECONDS: Final = 6 * 60 * 60


def _shared_cache() -> "RedisCache | None":
    """The Redis every gateway worker reaches, when the proxy was given one."""
    from litellm.proxy import proxy_server

    return proxy_server.redis_usage_cache


def _span_context(turn: Interaction, span_id: int) -> SpanContext:
    return SpanContext(turn.trace_id, span_id, is_remote=True, trace_flags=TraceFlags(TraceFlags.SAMPLED))


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

    _open_interactions: OrderedDict[str, _Open]  # mutable-ok: bounded per-worker memory of each session's turn

    @override
    async def async_log_success_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: datetime | float | None,
        end_time: datetime | float | None,
    ) -> None:
        session_id: Final = await self._recall(kwargs)
        await super().async_log_success_event(kwargs, response_obj, start_time, end_time)
        await self._share(session_id)

    @override
    async def async_log_failure_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: datetime | float | None,
        end_time: datetime | float | None,
    ) -> None:
        session_id: Final = await self._recall(kwargs)
        await super().async_log_failure_event(kwargs, response_obj, start_time, end_time)
        await self._share(session_id)

    async def _recall(self, kwargs: Mapping[str, object]) -> str | None:
        """Bring in the interaction another gateway worker opened for this call's session, so a subagent's
        call or a side request finds it whichever worker it lands on. Without a shared cache each worker
        only knows the interactions it opened itself."""
        call: Final = LLMCallEvent.from_dict(kwargs)
        made_by: Final = agent_call(call.payload)
        session_id: Final = call.session_id or made_by.derived_session
        shared: Final = _shared_cache()
        if not session_id or not made_by.integration or shared is None:
            return None
        try:
            stored: Final = await shared.async_get_cache(_SHARED_KEY.format(session=session_id))
            opened: Final = _OPEN.validate_json(stored) if isinstance(stored, (str, bytes)) else None
        except (ValidationError, OSError, RuntimeError):
            return session_id
        if opened is not None:
            self._open_interactions[session_id] = opened
        return session_id

    async def _share(self, session_id: str | None) -> None:
        shared: Final = _shared_cache()
        opened: Final = self._open_interactions.get(session_id) if session_id else None
        if shared is None or opened is None or session_id is None:
            return
        try:
            await shared.async_set_cache(
                _SHARED_KEY.format(session=session_id), _OPEN.dump_json(opened).decode(), ttl=_SHARED_SECONDS
            )
        except (OSError, RuntimeError):
            return

    @override
    def log_pre_api_call(self, model: str, messages: object, kwargs: Mapping[str, object]) -> None:
        """Only note when the call started: its span is created at close, once its interaction is known."""
        call: Final = LLMCallEvent.from_dict(kwargs)
        if call.is_no_upstream_call or call.call_id is None or call.call_id in self._open_llm_calls:
            return
        started: Final = to_ns(call.upstream_start_seconds) or to_ns(datetime.now())
        self._store_open_call(call.call_id, _LLMCallSpan(span=None, start_time_ns=started))

    @override
    def _seed_identity_baggage(self, identity: RequestIdentity, model: str | None, context: Context) -> Context:
        """The one place the parent of a model-call span passes through: hang it from its interaction."""
        parent: Final = _CALL_PARENT.get()
        placed: Final = context if parent is None else set_span_in_context(NonRecordingSpan(parent), context)
        return super()._seed_identity_baggage(identity, model, placed)

    def _place(self, made_by: AgentCall, session_id: str | None) -> _Placement:
        """The interaction a call belongs to. A call on the agent's own thread says so in its request; a
        subagent's call and a side request do not, so they join the interaction their session is in."""
        if not session_id or not made_by.integration:
            return _Placement(key=session_id)
        remembered: Final = self._open_interactions.get(session_id)
        if made_by.subagent or made_by.beside_thread:
            if remembered is None:
                return _Placement(key=session_id, actor=made_by.subagent)
            turn: Final = remembered.interaction
            launched_by: Final = next(
                (span_id for span_id, result in remembered.steps if made_by.subagent and made_by.subagent in result),
                turn.span_id,
            )
            return _Placement(
                key=turn.key, parent=_span_context(turn, launched_by), actor=made_by.subagent, interaction=turn
            )
        answered: Final = interaction(session_id, made_by.request)
        if answered is None:
            return _Placement(key=session_id)
        if remembered is None or remembered.interaction.key != answered.key:
            self._open_interactions[session_id] = _Open(interaction=answered)
        self._open_interactions.move_to_end(session_id)
        if len(self._open_interactions) > _REMEMBERED_SESSIONS:
            self._open_interactions.popitem(last=False)
        return _Placement(
            key=answered.key,
            parent=_span_context(answered, answered.span_id),
            opening=answered if answered.opens else None,
            interaction=answered,
        )

    @override
    def _close_llm_call(
        self,
        kwargs: Mapping[str, object],
        start_time: datetime | float | None,
        end_time: datetime | float | None,
    ) -> Span | None:
        call: Final = LLMCallEvent.from_dict(kwargs)
        made_by: Final = agent_call(call.payload)
        session_id: Final = call.session_id or made_by.derived_session
        placement: Final = self._place(made_by, session_id)
        facts: Final = {
            **made_by.attributes(call.session_id),
            **({_TRACE_KEY: placement.key} if placement.key else {}),
            **({_ACTOR: placement.actor or session_id} if session_id else {}),
        }
        token: Final = _CALL_FACTS.set(facts)
        parent: Final = _CALL_PARENT.set(placement.parent)
        try:
            span: Final = super()._close_llm_call(kwargs, start_time, end_time)
        finally:
            _CALL_PARENT.reset(parent)
            _CALL_FACTS.reset(token)
        if span is not None and session_id:
            if placement.opening is not None:
                self._emit_interaction(placement.opening, made_by, session_id, to_ns(start_time), to_ns(end_time))
            self._emit_tool_steps(made_by, session_id, placement, to_ns(start_time))
        return span

    def _emit_interaction(
        self, turn: Interaction, made_by: AgentCall, session_id: str, start_ns: int | None, end_ns: int | None
    ) -> None:
        """The span an agent's own trace opens for each typed message, emitted by the call that carries it."""
        pinned: Final = _NEXT_SPAN_ID.set(turn.span_id)
        try:
            self.tracer_provider.get_tracer(_TOOL_STEP_TRACER).start_span(
                _INTERACTION_SPAN.get(made_by.agent or "", "interaction"),
                context=set_span_in_context(INVALID_SPAN),
                start_time=start_ns,
                attributes={
                    "gen_ai.operation.name": "invoke_agent",
                    "span.type": "interaction",
                    "user_prompt": turn.prompt,
                    "user_prompt_length": len(turn.prompt),
                    "interaction.sequence": turn.sequence,
                    "gen_ai.prompt": turn.prompt,
                    "session.id": session_id,
                    **made_by.attributes(session_id),
                    _SESSION_ATTRIBUTE: session_id,
                    _TRACE_KEY: turn.key,
                    _ACTOR: session_id,
                },
            ).end(end_time=end_ns)
        finally:
            _NEXT_SPAN_ID.reset(pinned)

    def _emit_tool_steps(self, made_by: AgentCall, session_id: str, placement: _Placement, at_ns: int | None) -> None:
        """One zero-length step per tool result, at the start of the model call whose prompt carries it."""
        tracer: Final = self.tracer_provider.get_tracer(_TOOL_STEP_TRACER)
        context: Final = set_span_in_context(NonRecordingSpan(placement.parent)) if placement.parent else None
        steps: Final = tuple(
            (
                exchange,
                tracer.start_span(
                    f"execute_tool {exchange.name}",
                    context=context,
                    start_time=at_ns,
                    attributes={
                        "gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.name": exchange.name,
                        "gen_ai.tool.call.id": exchange.call_id,
                        "gen_ai.tool.call.arguments": exchange.arguments,
                        "gen_ai.tool.call.result": exchange.result,
                        **made_by.attributes(session_id),
                        _SESSION_ATTRIBUTE: session_id,
                        **({_TRACE_KEY: placement.key} if placement.key else {}),
                        _ACTOR: placement.actor or session_id,
                    },
                ),
            )
            for exchange in made_by.tool_exchanges
        )
        for _, step in steps:
            step.end(end_time=at_ns)
        remembered: Final = self._open_interactions.get(session_id)
        placed: Final = placement.interaction.key if placement.interaction else None
        if not steps or placement.actor or remembered is None or remembered.interaction.key != placed:
            return
        reported: Final = tuple((step.get_span_context().span_id, exchange.result) for exchange, step in steps)
        self._open_interactions[session_id] = _Open(
            interaction=remembered.interaction, steps=(*remembered.steps, *reported)[-_REMEMBERED_STEPS:]
        )


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
                    resource_attributes={"lens.session.capture": "true", "lens.session.attribute": _TRACE_KEY},
                )
            ],
            capture_message_content=CaptureMessageContent.SPAN_ONLY,
        ),
        callback_name=LENS_TRACE_CALLBACK,
    )
    trace_logger.tracer_provider.add_span_processor(_CallFactsProcessor())
    trace_logger.tracer_provider.id_generator = _InteractionIds()
    trace_logger._open_interactions = OrderedDict()
    manager: Final = litellm.logging_callback_manager
    manager.add_litellm_callback(trace_logger)
    try:
        yield None
    finally:
        manager.remove_callback_from_all_lists(trace_logger)
        await asyncio.to_thread(trace_logger.tracer_provider.shutdown)
