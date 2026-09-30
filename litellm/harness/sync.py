"""Sync API for litellm.harness: one daemon event-loop thread runs every async call."""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping, Sequence
from concurrent.futures import Future
from typing import (
    Any,
    TypeVar,
)

from pydantic import BaseModel

from litellm.harness.adapters.base import ApprovalHandler
from litellm.harness.options import HarnessOptions
from litellm.harness.runtime import (
    AsyncEventStream,
    AsyncSession,
    aresume,
    arun,
    asession,
    astream,
)
from litellm.harness.sandbox.base import Sandbox
from litellm.harness.types import (
    Done,
    Event,
    Gateway,
    Harness,
    PermissionMode,
    Result,
    State,
    Usage,
)

T = TypeVar("T")

IN_LOOP_MESSAGE = (
    "litellm.harness.{name}() cannot be called from a running event loop; "
    "use `await litellm.harness.a{name}(...)` instead"
)


class _LoopThread:
    """A single background event loop shared by every sync call in the process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._thread is None or not self._thread.is_alive():
                self._loop = asyncio.new_event_loop()
                self._thread = threading.Thread(
                    target=self._loop.run_forever,
                    name="litellm-harness-loop",
                    daemon=True,
                )
                self._thread.start()
            return self._loop

    def in_loop_thread(self) -> bool:
        return self._thread is not None and threading.current_thread() is self._thread

    def submit(self, coro: Coroutine[Any, Any, T]) -> Future[T]:
        return asyncio.run_coroutine_threadsafe(coro, self.loop())


_LOOP = _LoopThread()


def _ensure_sync_context(name: str) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(IN_LOOP_MESSAGE.format(name=name))


def run_sync(coro: Coroutine[Any, Any, T], name: str) -> T:
    """Run coro on the harness loop thread and block for its result."""
    try:
        _ensure_sync_context(name)
    except RuntimeError:
        coro.close()
        raise
    future = _LOOP.submit(coro)
    try:
        return future.result()
    except KeyboardInterrupt:
        future.cancel()
        raise


async def _anext(iterator: AsyncIterator[Event]) -> Event | None:
    try:
        return await iterator.__anext__()
    except StopAsyncIteration:
        return None


async def _aclose_stream(stream: AsyncEventStream) -> None:
    await stream.aclose()


class EventStream:
    """Sync iterator of events for one turn. `.result` is set once Done is seen."""

    def __init__(self, stream: AsyncEventStream, name: str = "stream") -> None:
        self._stream = stream
        self._name = name
        self._result: Result | None = None
        self._finished = False

    def __iter__(self) -> EventStream:
        return self

    def __next__(self) -> Event:
        if self._finished:
            raise StopIteration
        event = run_sync(_anext(self._stream), self._name)
        if event is None:
            self._finished = True
            raise StopIteration
        if isinstance(event, Done):
            self._result = event.result
        return event

    def __enter__(self) -> EventStream:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def result(self) -> Result | None:
        return self._result

    def cancel(self) -> None:
        """Stop the turn. Iteration still ends with Done(stop_reason='cancelled')."""
        _LOOP.loop().call_soon_threadsafe(self._stream.cancel)

    def close(self) -> None:
        """Abandon the stream and release the session behind it."""
        if self._finished:
            return
        self._finished = True
        run_sync(_aclose_stream(self._stream), self._name)


class Session:
    """Sync multi-turn session. Use as a context manager."""

    def __init__(self, inner: AsyncSession) -> None:
        self._inner = inner

    @property
    def aio(self) -> AsyncSession:
        """The underlying AsyncSession (runs on the harness loop thread)."""
        return self._inner

    def start(self) -> Session:
        run_sync(self._inner.start(), "session")
        return self

    def __enter__(self) -> Session:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def run(self, prompt: str) -> Result:
        return run_sync(self._inner.arun(prompt), "run")

    def stream(self, prompt: str) -> EventStream:
        return EventStream(self._inner.astream(prompt))

    def close(self) -> None:
        run_sync(self._inner.aclose(), "close")

    def detach(self) -> State:
        return run_sync(self._inner.adetach(), "detach")

    def stop(self) -> State:
        return run_sync(self._inner.astop(), "stop")

    def history(self) -> list[dict[str, Any]]:
        return run_sync(self._inner.history(), "history")

    @property
    def cost(self) -> float:
        return self._inner.cost

    @property
    def usage(self) -> Usage:
        return self._inner.usage

    @property
    def results(self) -> list[Result]:
        return list(self._inner.results)

    @property
    def session_id(self) -> str:
        return self._inner.session_id


def run(
    harness: Harness,
    prompt: str,
    *,
    sandbox: Sandbox,
    model: str | None = None,
    gateway: Gateway | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    instructions: str | None = None,
    tools: Sequence[Callable[..., Any]] = (),
    skills: Sequence[str | os.PathLike[str]] = (),
    disable_tools: Sequence[str] = (),
    permissions: PermissionMode = "full",
    on_approval: ApprovalHandler | None = None,
    output: type[BaseModel] | None = None,
    max_turns: int | None = None,
    timeout: float | None = None,
    metadata: Mapping[str, Any] | None = None,
    options: HarnessOptions | None = None,
    install: bool = False,
) -> Result:
    """Run one prompt to completion (blocking) and return the Result."""
    return run_sync(
        arun(
            harness,
            prompt,
            sandbox=sandbox,
            model=model,
            gateway=gateway,
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
        ),
        "run",
    )


def stream(
    harness: Harness,
    prompt: str,
    *,
    sandbox: Sandbox,
    model: str | None = None,
    gateway: Gateway | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    instructions: str | None = None,
    tools: Sequence[Callable[..., Any]] = (),
    skills: Sequence[str | os.PathLike[str]] = (),
    disable_tools: Sequence[str] = (),
    permissions: PermissionMode = "full",
    on_approval: ApprovalHandler | None = None,
    output: type[BaseModel] | None = None,
    max_turns: int | None = None,
    timeout: float | None = None,
    metadata: Mapping[str, Any] | None = None,
    options: HarnessOptions | None = None,
    install: bool = False,
) -> EventStream:
    """Stream events for one prompt (sync iterator). Validation errors raise here."""
    _ensure_sync_context("stream")
    inner = astream(
        harness,
        prompt,
        sandbox=sandbox,
        model=model,
        gateway=gateway,
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
    return EventStream(inner)


def session(
    harness: Harness,
    *,
    sandbox: Sandbox,
    model: str | None = None,
    gateway: Gateway | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    instructions: str | None = None,
    tools: Sequence[Callable[..., Any]] = (),
    skills: Sequence[str | os.PathLike[str]] = (),
    disable_tools: Sequence[str] = (),
    permissions: PermissionMode = "full",
    on_approval: ApprovalHandler | None = None,
    output: type[BaseModel] | None = None,
    max_turns: int | None = None,
    timeout: float | None = None,
    metadata: Mapping[str, Any] | None = None,
    options: HarnessOptions | None = None,
    install: bool = False,
) -> Session:
    """A sync multi-turn session: `with session(...) as s: s.run(...)`."""
    _ensure_sync_context("session")
    return Session(
        asession(
            harness,
            sandbox=sandbox,
            model=model,
            gateway=gateway,
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
    )


def resume(
    state: State | bytes,
    *,
    sandbox: Sandbox,
    model: str | None = None,
    gateway: Gateway | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    instructions: str | None = None,
    tools: Sequence[Callable[..., Any]] = (),
    skills: Sequence[str | os.PathLike[str]] = (),
    disable_tools: Sequence[str] = (),
    permissions: PermissionMode = "full",
    on_approval: ApprovalHandler | None = None,
    output: type[BaseModel] | None = None,
    max_turns: int | None = None,
    timeout: float | None = None,
    metadata: Mapping[str, Any] | None = None,
    options: HarnessOptions | None = None,
    install: bool = False,
) -> Session:
    """Continue a detached/stopped session (sync)."""
    _ensure_sync_context("resume")
    return Session(
        aresume(
            state,
            sandbox=sandbox,
            model=model,
            gateway=gateway,
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
    )
