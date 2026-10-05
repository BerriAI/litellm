"""The harness engine: validation, sessions, turns, approvals, files, usage and results.

Adapters only translate a runtime's native protocol into events. Everything that must behave
the same across harnesses (timeouts, max_turns, approvals, FileChange, structured output,
usage and cost) lives here.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine, Generator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Any,
    Final,
    get_args,
)

from pydantic import BaseModel, ValidationError

import litellm
from litellm.constants import HARNESS_EVENT_QUEUE_MAX_SIZE
from litellm.harness.context import ApprovalHandler, GatewayTarget, SessionContext
from litellm.harness.endpoint import ModelEndpoint
from litellm.harness.errors import (
    CapabilityUnsupported,
    HarnessError,
    HarnessInstallFailed,
    OptionsMismatch,
    OutputInvalid,
    SessionClosed,
    StateIncompatible,
)
from litellm.harness.handlers import get_harness_config, get_harness_handler
from litellm.harness.handlers.base import BaseHarnessHandler
from litellm.harness.options import HarnessOptions
from litellm.harness.sandbox.base import Sandbox
from litellm.harness.sandbox.snapshot import build_file_changes, capture_text_contents
from litellm.harness.types import (
    Approval,
    Capabilities,
    Done,
    Event,
    FileChange,
    Harness,
    PermissionMode,
    Result,
    State,
    StopReason,
    Text,
    ToolCall,
    Usage,
    require_harness,
)
from litellm.llms.base_llm.harness.transformation import BaseHarnessConfig
from litellm.llms.base_llm.harness.utils import last_json_object

PERMISSION_MODES: Final = frozenset(get_args(PermissionMode))
SKILL_FILE: Final = "SKILL.md"
# Adapter errors that mean "misconfigured", not "the runtime crashed": re-raised to the caller.
verbose_logger: Final = logging.getLogger("LiteLLM")

PROPAGATED_ERRORS: Final = (HarnessInstallFailed, CapabilityUnsupported)


@dataclass(frozen=True)
class SessionConfig:
    """Every per-session parameter a caller can pass, already normalized."""

    harness: Harness
    sandbox: Sandbox
    model: str | None = None
    gateway: GatewayTarget | None = None
    api_key: str | None = None
    api_base: str | None = None
    instructions: str | None = None
    tools: Sequence[Callable[..., object]] = ()
    skills: Sequence[str] = ()
    disable_tools: Sequence[str] = ()
    permissions: PermissionMode = "full"
    on_approval: ApprovalHandler | None = None
    output: type[BaseModel] | None = None
    max_turns: int | None = None
    timeout: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    options: HarnessOptions | None = None
    install: bool = False


LITELLM_PROXY_PREFIX: Final = "litellm_proxy/"


def resolve_model_route(
    model: str | None, api_key: str | None, api_base: str | None
) -> tuple[str | None, GatewayTarget | None]:
    """(model sent to the runtime, gateway or None).

    `litellm_proxy/<group>` (or `litellm.use_litellm_proxy = True`) routes every model call
    through the LiteLLM AI Gateway, using api_base/api_key or LITELLM_PROXY_API_BASE /
    LITELLM_PROXY_API_KEY. Anything else is called directly through the LiteLLM SDK.
    """
    prefixed = model is not None and model.startswith(LITELLM_PROXY_PREFIX)
    if not prefixed and not litellm.use_litellm_proxy:
        return model, None
    group = model[len(LITELLM_PROXY_PREFIX) :] if prefixed and model is not None else model
    base = (api_base or os.environ.get("LITELLM_PROXY_API_BASE") or "").strip()
    key = (api_key or os.environ.get("LITELLM_PROXY_API_KEY") or "").strip()
    if not base:
        raise ValueError("litellm_proxy/ models need the gateway URL: pass api_base= or set LITELLM_PROXY_API_BASE")
    if not key:
        raise ValueError("litellm_proxy/ models need a gateway virtual key: pass api_key= or set LITELLM_PROXY_API_KEY")
    return group, GatewayTarget(api_base=base.rstrip("/"), api_key=key)


def _normalize_skill(skill: str | os.PathLike[str]) -> str:
    path = os.path.abspath(os.fspath(skill))
    if not os.path.isfile(os.path.join(path, SKILL_FILE)):
        raise ValueError(f"Skill folder {path!r} has no {SKILL_FILE}")
    return path


def _normalize_skills(skills: Sequence[str | os.PathLike[str]]) -> tuple[str, ...]:
    return tuple(_normalize_skill(skill) for skill in skills)


def _check_basic(config: SessionConfig) -> None:
    if config.permissions not in PERMISSION_MODES:
        raise ValueError(f"permissions must be one of {sorted(PERMISSION_MODES)}, got {config.permissions!r}")
    if config.max_turns is not None and config.max_turns < 1:
        raise ValueError("max_turns must be >= 1")
    if config.timeout is not None and config.timeout <= 0:
        raise ValueError("timeout must be > 0")
    if config.install:
        raise CapabilityUnsupported("install=True is not supported yet; put the runtime binary on PATH in the sandbox")


def _check_options(config: SessionConfig, harness_config: BaseHarnessConfig) -> None:
    if config.options is None or isinstance(config.options, harness_config.options_type):
        return
    raise OptionsMismatch(
        f"{type(config.options).__name__} cannot be used with Harness.{config.harness.name}; "
        f"use {harness_config.options_type.__name__}"
    )


def _check_capabilities(config: SessionConfig, caps: Capabilities, interactive: bool) -> None:
    name = f"Harness.{config.harness.name}"
    if config.permissions not in caps.permission_modes:
        raise CapabilityUnsupported(
            f"{name} does not support permissions={config.permissions!r}; supported: {sorted(caps.permission_modes)}"
        )
    if config.permissions == "ask":
        if not caps.tool_approval:
            raise CapabilityUnsupported(f"{name} does not support tool approvals")
        if config.on_approval is None and not interactive:
            raise ValueError("permissions='ask' needs on_approval=, or use stream() and answer Approval events")
    if config.output is not None and not caps.structured_output:
        raise CapabilityUnsupported(f"{name} does not support output=")
    if config.tools and not caps.custom_tools:
        raise CapabilityUnsupported(f"{name} does not support custom tools=")
    if config.skills and not caps.skills:
        raise CapabilityUnsupported(f"{name} does not support skills=")
    if config.disable_tools and not caps.tool_filtering:
        raise CapabilityUnsupported(f"{name} does not support disable_tools=")


def validate(config: SessionConfig, harness_config: BaseHarnessConfig, interactive: bool) -> None:
    """Raise before anything starts if the request cannot be served."""
    _check_basic(config)
    _check_options(config, harness_config)
    _check_capabilities(config, harness_config.capabilities, interactive)


def build_config(
    harness: Harness,
    *,
    sandbox: Sandbox,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    instructions: str | None = None,
    tools: Sequence[Callable[..., object]] = (),
    skills: Sequence[str | os.PathLike[str]] = (),
    disable_tools: Sequence[str] = (),
    permissions: PermissionMode = "full",
    on_approval: ApprovalHandler | None = None,
    output: type[BaseModel] | None = None,
    max_turns: int | None = None,
    timeout: float | None = None,
    metadata: Mapping[str, object] | None = None,
    options: HarnessOptions | None = None,
    install: bool = False,
) -> SessionConfig:
    """Normalize public keyword arguments into a SessionConfig."""
    resolved_harness = require_harness(harness)
    routed_model, gateway = resolve_model_route(model, api_key, api_base)
    return SessionConfig(
        harness=resolved_harness,
        sandbox=sandbox,
        model=routed_model,
        gateway=gateway,
        api_key=api_key,
        api_base=api_base,
        instructions=instructions,
        tools=tuple(tools),
        skills=tuple(_normalize_skills(skills)),
        disable_tools=tuple(disable_tools),
        permissions=permissions,
        on_approval=on_approval,
        output=output,
        max_turns=max_turns,
        timeout=timeout,
        metadata=MappingProxyType(dict(metadata or ())),
        options=options,
        install=install,
    )


def _context_for(config: SessionConfig) -> SessionContext:
    return SessionContext(
        harness=config.harness,
        sandbox=config.sandbox,
        session_id=uuid.uuid4().hex,
        model=config.model,
        gateway=config.gateway,
        api_key=config.api_key,
        api_base=config.api_base,
        instructions=config.instructions,
        tools=config.tools,
        skills=config.skills,
        disable_tools=config.disable_tools,
        permissions=config.permissions,
        on_approval=config.on_approval,
        output=config.output,
        max_turns=config.max_turns,
        timeout=config.timeout,
        metadata=config.metadata,
        options=config.options,
    )


def parse_output(output: type[BaseModel], output_json: str | None, text: str) -> tuple[BaseModel | None, str | None]:
    """Return (model, None) on success or (None, error message) on failure."""
    raw = output_json or last_json_object(text)
    if raw is None:
        return None, "no JSON object found in the final answer"
    try:
        return output.model_validate_json(raw), None
    except ValidationError as e:
        return None, str(e)


@dataclass
class _End:
    """Sentinel the producer puts on the queue when the handler turn is over."""

    reason: StopReason | None = None
    error: BaseException | None = None


class TurnControl:
    """Lets a stream consumer cancel the running turn."""

    def __init__(self) -> None:
        self.cancelled = False
        self.producer: asyncio.Task[None] | None = None

    def cancel(self) -> None:
        self.cancelled = True
        if self.producer is not None and not self.producer.done():
            self.producer.cancel()


async def _aclose(events: AsyncIterator[Event]) -> None:
    closer = getattr(events, "aclose", None)
    if closer is None:
        return
    try:
        await closer()
    except Exception:  # closing must not mask the turn's own outcome
        verbose_logger.debug("harness: error closing handler turn", exc_info=True)


async def pump_events(
    events: AsyncIterator[Event],
    queue: asyncio.Queue[Event | _End],
    max_turns: int | None,
) -> None:
    """Drive the handler turn in one task, enforcing max_turns on ToolCall events."""
    end = _End()
    tool_calls = 0
    try:
        async for event in events:
            if isinstance(event, ToolCall):
                tool_calls += 1
                if max_turns is not None and tool_calls > max_turns:
                    end = _End(reason="max_turns")
                    break
            # Backpressure: a runtime that streams faster than the consumer waits here.
            await queue.put(event)
    except asyncio.CancelledError:
        end = _End(reason="cancelled")
        raise
    except Exception as e:  # any runtime failure becomes stop_reason="runtime_error" (see _Turn._finish)
        verbose_logger.debug("harness: handler turn raised", exc_info=True)
        end = _End(error=e)
    finally:
        await _aclose(events)
        await _put_end(queue, end)


async def _put_end(queue: asyncio.Queue[Event | _End], end: _End) -> None:
    """Queue the end marker behind every event, waiting for room so no event is dropped.

    A cancelled turn has no consumer left to drain the queue, so only then is space made
    by discarding queued events.
    """
    if end.reason != "cancelled":
        try:
            await queue.put(end)
            return
        except asyncio.CancelledError:
            pass
    while queue.full():
        queue.get_nowait()
    queue.put_nowait(end)


async def call_approval_handler(handler: ApprovalHandler, approval: Approval) -> None:
    """Run on_approval (sync in a worker thread, or async) and resolve approval."""
    try:
        if inspect.iscoroutinefunction(handler):
            decision: object = await handler(approval)
        else:
            decision = await asyncio.to_thread(handler, approval)
            if inspect.isawaitable(decision):
                decision = await decision
    except Exception as e:  # a failing user callback denies the tool instead of crashing the turn
        verbose_logger.warning("harness: on_approval raised for tool %s; denying", approval.tool, exc_info=True)
        approval.deny(f"on_approval raised: {e}")
        return
    if decision:
        approval.allow()
    else:
        approval.deny("denied by on_approval")


class _Turn:
    """One prompt -> events -> Done cycle on a started session."""

    def __init__(
        self,
        session: AsyncSession,
        prompt: str,
        control: TurnControl,
        interactive: bool,
    ) -> None:
        self.session = session
        self.ctx = session.ctx
        self.prompt = prompt
        self.control = control
        self.interactive = interactive
        self.queue: asyncio.Queue[Event | _End] = asyncio.Queue(maxsize=HARNESS_EVENT_QUEUE_MAX_SIZE)
        self.events: list[Event] = []  # mutable-ok: per-turn accumulator the runtime appends events to
        self.text_parts: list[str] = []  # mutable-ok: per-turn accumulator of streamed text deltas
        self.emitted_files: set[tuple[str, str]] = set()  # mutable-ok: per-turn record of emitted FileChanges
        self.approval_tasks: list[asyncio.Future[None]] = []  # mutable-ok: per-turn in-flight approval tasks
        self.stop_reason: StopReason = "done"
        self.error_text: str | None = None
        self.before: Mapping[str, str] = MappingProxyType({})
        self.before_contents: Mapping[str, bytes] = MappingProxyType({})
        self.usage_before: tuple[int, int, int, float] = (0, 0, 0, 0.0)
        self.deadline: float | None = None

    async def _begin(self) -> None:
        sandbox = self.ctx.sandbox
        self.before = await sandbox.snapshot()
        self.before_contents = await capture_text_contents(sandbox, self.before)
        self.usage_before = self.session.usage_counters()
        self.ctx.final_text = ""
        self.ctx.output_json = None
        if self.ctx.timeout is not None:
            self.deadline = asyncio.get_running_loop().time() + self.ctx.timeout

    def _start_producer(self) -> None:
        events = self.session.handler.turn(self.ctx, self.prompt)
        self.control.producer = asyncio.ensure_future(pump_events(events, self.queue, self.ctx.max_turns))
        if self.control.cancelled:
            self.control.producer.cancel()

    async def _stop_producer(self) -> None:
        producer = self.control.producer
        live_producer = (producer,) if producer is not None and not producer.done() else ()
        pending = (*(task for task in self.approval_tasks if not task.done()), *live_producer)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.wait(pending)

    async def _next_item(self) -> Event | _End:
        if self.deadline is None:
            return await self.queue.get()
        remaining = self.deadline - asyncio.get_running_loop().time()
        try:
            if remaining <= 0:
                raise asyncio.TimeoutError
            return await asyncio.wait_for(self.queue.get(), remaining)
        except asyncio.TimeoutError:
            await self._stop_producer()
            return _End(reason="timeout")

    def _finish(self, end: _End) -> None:
        if end.error is not None:
            if isinstance(end.error, PROPAGATED_ERRORS):
                raise end.error
            self.stop_reason = "runtime_error"
            self.error_text = f"{type(end.error).__name__}: {end.error}"
            verbose_logger.warning("harness %s runtime error: %s", self.ctx.harness.value, self.error_text)
            return
        if self.control.cancelled:
            self.stop_reason = "cancelled"
        elif end.reason is not None:
            self.stop_reason = end.reason

    async def _on_approval(self, approval: Approval) -> None:
        handler = self.ctx.on_approval
        if handler is not None:
            self.approval_tasks.append(asyncio.ensure_future(call_approval_handler(handler, approval)))
        elif not self.interactive:
            approval.deny("no approval handler")

    async def _record(self, event: Event) -> None:
        if isinstance(event, Text):
            self.text_parts.append(event.delta)
        elif isinstance(event, FileChange):
            self.emitted_files.add((event.path, event.kind))
        elif isinstance(event, Approval):
            await self._on_approval(event)
        self.events.append(event)

    async def _drain(self) -> AsyncIterator[Event]:
        while True:
            item = await self._next_item()
            if isinstance(item, _End):
                self._finish(item)
                return
            if isinstance(item, Done):
                continue
            await self._record(item)
            yield item
            if isinstance(item, Approval) and self.ctx.on_approval is None:
                # The consumer asked for the next event without answering.
                item.deny("approval not answered")

    async def _file_changes(self) -> list[FileChange]:  # mutable-ok: becomes the public Result.files list
        sandbox = self.ctx.sandbox
        after = await sandbox.snapshot()
        files = await build_file_changes(sandbox, self.before, after, self.before_contents)
        seen = {  # mutable-ok: dedupe set grown while merging streamed FileChange events
            change.path for change in files
        }
        for event in self.events:
            if isinstance(event, FileChange) and event.path not in seen:
                files.append(event)
                seen.add(event.path)
        return files

    def _text(self) -> str:
        text = self.ctx.final_text or "".join(self.text_parts)
        if self.error_text is None:
            return text
        return f"{text}\n\n{self.error_text}" if text else self.error_text

    def _usage(self) -> tuple[Usage, float]:
        now = self.session.usage_counters()
        before = self.usage_before
        usage = Usage(
            input_tokens=now[0] - before[0],
            output_tokens=now[1] - before[1],
            calls=now[2] - before[2],
        )
        return usage, max(now[3] - before[3], 0.0)

    def _result(
        self,
        files: list[FileChange],  # mutable-ok: Result.files is a public list field
        output: BaseModel | None,
    ) -> Result:
        usage, cost = self._usage()
        return Result(
            text=self._text(),
            output=output,
            files=files,
            events=list(  # mutable-ok: Result.events is a public list field; copy detaches it from the accumulator
                self.events
            ),
            usage=usage,
            cost=cost,
            stop_reason=self.stop_reason,
            session_id=self.ctx.session_id,
        )

    def _output(self) -> tuple[BaseModel | None, str | None, str | None]:
        """(parsed output, raw text, error) for the structured-output check."""
        output_type = self.ctx.output
        if output_type is None or self.stop_reason != "done":
            return None, None, None
        text = self._text()
        parsed, error = parse_output(output_type, self.ctx.output_json, text)
        return parsed, self.ctx.output_json or text, error

    async def run(self) -> AsyncIterator[Event]:
        await self._begin()
        self._start_producer()
        try:
            async for event in self._drain():
                yield event
        finally:
            await self._stop_producer()
        if self.stop_reason != "done":
            await self.session.interrupt()
        files = await self._file_changes()
        for change in files:
            if (change.path, change.kind) not in self.emitted_files:
                self.events.append(change)
                yield change
        parsed, raw, error = self._output()
        result = self._result(files, parsed)
        self.session.record(result)
        yield Done(result)
        if error is not None:
            raise OutputInvalid(
                f"Final answer did not match {self.ctx.output.__name__ if self.ctx.output else 'output'}: {error}",
                raw=raw or "",
                result=result,
            )


class AsyncEventStream:
    """Async iterator of events for one turn. `.result` is set once Done is seen."""

    def __init__(self, source: AsyncIterator[Event], control: TurnControl) -> None:
        self._source = source
        self._control = control
        self._result: Result | None = None

    def __aiter__(self) -> AsyncEventStream:
        return self

    async def __anext__(self) -> Event:
        event = await self._source.__anext__()
        if isinstance(event, Done):
            self._result = event.result
        return event

    @property
    def result(self) -> Result | None:
        return self._result

    def cancel(self) -> None:
        """Stop the turn. The stream still ends with Done(stop_reason='cancelled')."""
        self._control.cancel()

    async def aclose(self) -> None:
        await _aclose(self._source)


async def _one_shot(session: AsyncSession, prompt: str, control: TurnControl) -> AsyncIterator[Event]:
    """Stream one turn on a fresh session and close it before Done is handed out."""
    try:
        async for event in session.turn_events(prompt, control, interactive=True):
            if isinstance(event, Done):
                await session.aclose()
            yield event
    finally:
        await session.aclose()


class AsyncSession:
    """A multi-turn conversation with one harness. Use `async with` or `await`."""

    def __init__(
        self,
        config: SessionConfig,
        *,
        resume_from: str | None = None,
        interactive: bool = True,
    ) -> None:
        self.config = config
        self.harness_config = get_harness_config(config.harness)
        validate(config, self.harness_config, interactive=interactive)
        if resume_from is not None and not self.harness_config.capabilities.resume:
            raise CapabilityUnsupported(f"Harness.{config.harness.name} does not support resume")
        self.ctx = _context_for(config)
        # Config-specific static checks (managed option keys, required model) before any I/O.
        self.harness_config.validate_environment(self.ctx)
        self.handler: BaseHarnessHandler = get_harness_handler(self.harness_config)
        self.results: list[Result] = []  # mutable-ok: session accumulator; each turn's Result is appended
        self._resume_from = resume_from
        self._native_id: str | None = resume_from
        self._started = False
        self._closed = False
        self._busy = False
        self._restart_needed = False

    def __await__(self) -> Generator[object, None, AsyncSession]:
        return self.start().__await__()

    async def __aenter__(self) -> AsyncSession:
        return await self.start()

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def _open_endpoint(self) -> None:
        if not self.harness_config.uses_model_endpoint or self.ctx.endpoint is not None:
            return
        endpoint = ModelEndpoint(
            self.config.harness,
            self.config.model,
            self.config.gateway,
            api_key=self.config.api_key,
            api_base=self.config.api_base,
            metadata=self.config.metadata,
        )
        await endpoint.__aenter__()
        self.ctx.endpoint = endpoint

    async def _launch(self) -> None:
        await self.handler.start(self.ctx)
        if self._native_id is not None and (self._resume_from is not None or self._restart_needed):
            await self.handler.resume(self.ctx, self._native_id)

    async def start(self) -> AsyncSession:
        if self._closed:
            raise SessionClosed("session is closed")
        if self._started:
            return self
        await self._open_endpoint()
        try:
            await self._launch()
        except BaseException:
            await self._close_endpoint()
            raise
        self._started = True
        return self

    async def interrupt(self) -> None:
        """Stop the runtime after a timeout / max_turns / cancel; next turn restarts it."""
        self._native_id = self.handler.native_session_id() or self._native_id
        try:
            await self.handler.stop(self.ctx)
        except Exception:  # the next turn restarts the runtime regardless
            verbose_logger.warning("harness: handler stop failed", exc_info=True)
        self._restart_needed = True

    async def _ensure_ready(self) -> None:
        if self._closed:
            raise SessionClosed("session is closed")
        if not self._started:
            await self.start()
        elif self._restart_needed:
            await self._launch()
        self._restart_needed = False

    async def _close_endpoint(self) -> None:
        endpoint = self.ctx.endpoint
        self.ctx.endpoint = None
        if endpoint is None:
            return
        try:
            await endpoint.__aexit__(None, None, None)
        except Exception:  # shutdown is best-effort cleanup
            verbose_logger.warning("harness: endpoint shutdown failed", exc_info=True)

    async def aclose(self) -> None:
        """Stop the runtime and the endpoint. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        if self._started:
            self._native_id = self.handler.native_session_id() or self._native_id
            try:
                await self.handler.stop(self.ctx)
            except Exception:  # still close the endpoint below
                verbose_logger.warning("harness: handler stop failed", exc_info=True)
        await self._close_endpoint()

    close = aclose

    def state(self) -> State:
        native = self._native_id
        if self._started and not self._closed:
            native = self.handler.native_session_id() or native
        return State(
            harness=self.config.harness,
            native_session_id=native,
            workdir=self.config.sandbox.workdir,
            model=self.config.model,
        )

    async def adetach(self) -> State:
        """Release local resources and return State to resume() later."""
        await self.aclose()
        return self.state()

    async def astop(self) -> State:
        """Stop the session for good and return its final State."""
        await self.aclose()
        return self.state()

    detach = adetach
    stop = astop

    def usage_counters(self) -> tuple[int, int, int, float]:
        """(input_tokens, output_tokens, calls, cost) so far, from endpoint or handler."""
        endpoint = self.ctx.endpoint
        if endpoint is not None:
            usage = endpoint.usage
            return (usage.input_tokens, usage.output_tokens, usage.calls, usage.cost)
        ctx = self.ctx
        return (ctx.input_tokens, ctx.output_tokens, ctx.calls, ctx.cost)

    def record(self, result: Result) -> None:
        self.results.append(result)

    async def turn_events(self, prompt: str, control: TurnControl, interactive: bool) -> AsyncIterator[Event]:
        if self._busy:
            raise HarnessError("a turn is already running on this session")
        self._busy = True
        try:
            await self._ensure_ready()
            async for event in _Turn(self, prompt, control, interactive).run():
                yield event
        finally:
            self._busy = False

    def astream(self, prompt: str) -> AsyncEventStream:
        control = TurnControl()
        return AsyncEventStream(self.turn_events(prompt, control, interactive=True), control)

    async def arun(self, prompt: str) -> Result:
        return await _collect(self.turn_events(prompt, TurnControl(), False))

    async def history(
        self,
    ) -> list[dict[str, Any]]:  # mutable-ok: public API returns OpenAI-format message dicts from the handler
        if not self.harness_config.capabilities.history:
            raise CapabilityUnsupported(f"Harness.{self.config.harness.name} does not expose history")
        await self._ensure_ready()
        return await self.handler.history(self.ctx)

    @property
    def cost(self) -> float:
        return sum(result.cost for result in self.results)

    @property
    def usage(self) -> Usage:
        return Usage(
            input_tokens=sum(r.usage.input_tokens for r in self.results),
            output_tokens=sum(r.usage.output_tokens for r in self.results),
            calls=sum(r.usage.calls for r in self.results),
        )

    @property
    def session_id(self) -> str:
        return self.ctx.session_id

    @property
    def closed(self) -> bool:
        return self._closed


async def _collect(events: AsyncIterator[Event]) -> Result:
    result: Result | None = None
    async for event in events:
        if isinstance(event, Done):
            result = event.result
    if result is None:
        raise HarnessError("turn ended without a result")
    return result


def aagent_session(
    harness: Harness,
    *,
    sandbox: Sandbox,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    instructions: str | None = None,
    tools: Sequence[Callable[..., object]] = (),
    skills: Sequence[str | os.PathLike[str]] = (),
    disable_tools: Sequence[str] = (),
    permissions: PermissionMode = "full",
    on_approval: ApprovalHandler | None = None,
    output: type[BaseModel] | None = None,
    max_turns: int | None = None,
    timeout: float | None = None,
    metadata: Mapping[str, object] | None = None,
    options: HarnessOptions | None = None,
    install: bool = False,
) -> AsyncSession:
    """A multi-turn agent session: `async with litellm.aagent_session(...) as s:`."""
    config = build_config(
        harness,
        sandbox=sandbox,
        model=model,
        api_key=api_key,
        api_base=api_base,
        instructions=instructions,
        tools=tools,
        skills=skills,
        disable_tools=disable_tools,
        permissions=permissions,
        on_approval=on_approval,
        output=output,
        max_turns=max_turns,
        timeout=timeout,
        metadata=metadata,
        options=options,
        install=install,
    )
    return AsyncSession(config)


async def arun_agent(
    harness: Harness,
    prompt: str,
    *,
    sandbox: Sandbox,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    instructions: str | None = None,
    tools: Sequence[Callable[..., object]] = (),
    skills: Sequence[str | os.PathLike[str]] = (),
    disable_tools: Sequence[str] = (),
    permissions: PermissionMode = "full",
    on_approval: ApprovalHandler | None = None,
    output: type[BaseModel] | None = None,
    max_turns: int | None = None,
    timeout: float | None = None,
    metadata: Mapping[str, object] | None = None,
    options: HarnessOptions | None = None,
    install: bool = False,
) -> Result:
    """Run one prompt to completion and return the Result."""
    config = build_config(
        harness,
        sandbox=sandbox,
        model=model,
        api_key=api_key,
        api_base=api_base,
        instructions=instructions,
        tools=tools,
        skills=skills,
        disable_tools=disable_tools,
        permissions=permissions,
        on_approval=on_approval,
        output=output,
        max_turns=max_turns,
        timeout=timeout,
        metadata=metadata,
        options=options,
        install=install,
    )
    async with AsyncSession(config, interactive=False) as session:
        return await session.arun(prompt)


def astream_agent(
    harness: Harness,
    prompt: str,
    *,
    sandbox: Sandbox,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    instructions: str | None = None,
    tools: Sequence[Callable[..., object]] = (),
    skills: Sequence[str | os.PathLike[str]] = (),
    disable_tools: Sequence[str] = (),
    permissions: PermissionMode = "full",
    on_approval: ApprovalHandler | None = None,
    output: type[BaseModel] | None = None,
    max_turns: int | None = None,
    timeout: float | None = None,
    metadata: Mapping[str, object] | None = None,
    options: HarnessOptions | None = None,
    install: bool = False,
) -> AsyncEventStream:
    """Stream events for one prompt. Validation errors raise here, before iteration."""
    session = aagent_session(
        harness,
        sandbox=sandbox,
        model=model,
        api_key=api_key,
        api_base=api_base,
        instructions=instructions,
        tools=tools,
        skills=skills,
        disable_tools=disable_tools,
        permissions=permissions,
        on_approval=on_approval,
        output=output,
        max_turns=max_turns,
        timeout=timeout,
        metadata=metadata,
        options=options,
        install=install,
    )
    control = TurnControl()
    return AsyncEventStream(_one_shot(session, prompt, control), control)


def _coerce_state(state: State | bytes) -> State:
    if isinstance(state, (bytes, bytearray)):
        return State.loads(bytes(state))
    if not isinstance(state, State):
        raise TypeError(f"state must be a State or bytes, got {type(state).__name__}")
    return state


def aagent_resume(
    state: State | bytes,
    *,
    sandbox: Sandbox,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    instructions: str | None = None,
    tools: Sequence[Callable[..., object]] = (),
    skills: Sequence[str | os.PathLike[str]] = (),
    disable_tools: Sequence[str] = (),
    permissions: PermissionMode = "full",
    on_approval: ApprovalHandler | None = None,
    output: type[BaseModel] | None = None,
    max_turns: int | None = None,
    timeout: float | None = None,
    metadata: Mapping[str, object] | None = None,
    options: HarnessOptions | None = None,
    install: bool = False,
) -> AsyncSession:
    """Continue a detached/stopped session from its State."""
    resolved = _coerce_state(state)
    if not resolved.native_session_id:
        raise StateIncompatible("State has no native session id to resume")
    config = build_config(
        resolved.harness,
        sandbox=sandbox,
        model=model if model is not None else resolved.model,
        api_key=api_key,
        api_base=api_base,
        instructions=instructions,
        tools=tools,
        skills=skills,
        disable_tools=disable_tools,
        permissions=permissions,
        on_approval=on_approval,
        output=output,
        max_turns=max_turns,
        timeout=timeout,
        metadata=metadata,
        options=options,
        install=install,
    )
    return AsyncSession(config, resume_from=resolved.native_session_id)


def agent_capabilities(harness: Harness) -> Capabilities:
    """What a harness supports (permission modes, structured output, tools...)."""
    return get_harness_config(require_harness(harness)).capabilities


def aagent(
    harness: Harness,
    prompt: str,
    *,
    sandbox: Sandbox,
    stream: bool = False,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    instructions: str | None = None,
    tools: Sequence[Callable[..., object]] = (),
    skills: Sequence[str | os.PathLike[str]] = (),
    disable_tools: Sequence[str] = (),
    permissions: PermissionMode = "full",
    on_approval: ApprovalHandler | None = None,
    output: type[BaseModel] | None = None,
    max_turns: int | None = None,
    timeout: float | None = None,
    metadata: Mapping[str, object] | None = None,
    options: HarnessOptions | None = None,
    install: bool = False,
) -> Coroutine[object, object, Result] | AsyncEventStream:
    """Run an agent harness on one prompt.

    `await litellm.aagent(...)` returns a Result. With stream=True it returns an async
    iterator of events instead: `async for event in litellm.aagent(..., stream=True)`.
    """
    kwargs: dict[str, Any] = {  # mutable-ok: forwarded as **kwargs to arun_agent/astream_agent
        "sandbox": sandbox,
        "model": model,
        "api_key": api_key,
        "api_base": api_base,
        "instructions": instructions,
        "tools": tools,
        "skills": skills,
        "disable_tools": disable_tools,
        "permissions": permissions,
        "on_approval": on_approval,
        "output": output,
        "max_turns": max_turns,
        "timeout": timeout,
        "metadata": metadata,
        "options": options,
        "install": install,
    }
    if stream:
        return astream_agent(harness, prompt, **kwargs)
    return arun_agent(harness, prompt, **kwargs)
