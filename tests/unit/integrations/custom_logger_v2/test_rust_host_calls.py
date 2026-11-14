import asyncio
import contextvars
from collections.abc import Awaitable, Generator, Mapping
from pathlib import Path
from typing import Final

import pytest

from litellm.integrations.custom_logger_v2 import Field, Hook, Patch
from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.rust import HostedRust, RustLogger, rust_logger
from litellm.rust_bridge.execution import Await, Complete, Step

EXAMPLE_MANIFEST: Final = (
    Path(__file__).parents[4] / "litellm/rust/crates/callbacks-python/tests/round_trip/manifest.json"
)
_SEEN: Final[contextvars.ContextVar[str]] = contextvars.ContextVar("rust_host_calls_seen", default="unset")
_WRITTEN: Final[contextvars.ContextVar[str]] = contextvars.ContextVar("rust_host_calls_written", default="unset")


def _patched(model: object) -> Complete:
    return Complete({"patch": {"ops": [{"op": "set", "field": "model", "value": model}]}})


class Host:
    def __init__(self) -> None:
        self.tasks: list[asyncio.Task[object] | None] = []

    async def lookup(self) -> str:
        self.tasks.append(asyncio.current_task())
        _ = _WRITTEN.set("written by the host")
        return _SEEN.get()


class Execution:
    def __init__(self, host: Host) -> None:
        self.closed = False
        self._steps: Final = self._run(host)

    def _run(self, host: Host) -> Generator[Step, object]:
        answer: Final = yield Await(host.lookup())
        yield _patched(answer)

    def send(self, value: object) -> Step:
        return next(self._steps) if value is None else self._steps.send(value)

    def throw(self, error: BaseException) -> Step:
        return self._steps.throw(error)

    def close(self) -> None:
        self.closed = True


class Rust:
    def __init__(self) -> None:
        self.handled: list[str] = []
        self.started: list[tuple[str, object]] = []
        self.executions: list[Execution] = []

    def manifest(self) -> str:
        return EXAMPLE_MANIFEST.read_text()

    def request_state(self) -> object:
        return None

    def handle(
        self, hook: str, fields: Mapping[str, object], call: Mapping[str, object] | None, state: object, /
    ) -> Awaitable[object]:
        self.handled.append(hook)
        return self._unhosted()

    async def _unhosted(self) -> object:
        return _patched("unhosted").value

    def start(
        self, hook: str, fields: Mapping[str, object], call: Mapping[str, object] | None, state: object, host: object, /
    ) -> Execution:
        assert isinstance(host, Host)
        self.started.append((hook, host))
        execution: Final = Execution(host)
        self.executions.append(execution)
        return execution


class Unhosted:
    def manifest(self) -> str:
        return EXAMPLE_MANIFEST.read_text()

    def request_state(self) -> object:
        return None

    def handle(
        self, hook: str, fields: Mapping[str, object], call: Mapping[str, object] | None, state: object, /
    ) -> Awaitable[object]:
        return asyncio.sleep(0, result=None)


async def _pre_call(logger: RustLogger) -> object:
    return await logger.run_rust(
        Hook.pre_call,
        frozenset({Field.model}),
        object(),
        CallContext(hook=Hook.pre_call, call=None, state=None),
    )


async def test_a_logger_without_a_host_awaits_handle_as_before() -> None:
    rust: Final = Rust()

    outcome: Final = await _pre_call(rust_logger(rust))

    assert outcome == Patch().set(Field.model, "unhosted")
    assert (rust.handled, rust.started) == (["pre_call"], [])


async def test_a_logger_with_a_host_drives_the_execution_in_the_callers_task_and_context() -> None:
    rust: Final = Rust()
    host: Final = Host()
    _ = _SEEN.set("set by the caller")

    outcome: Final = await _pre_call(rust_logger(rust, host=host))

    assert outcome == Patch().set(Field.model, "set by the caller")
    assert rust.handled == []
    assert rust.started == [("pre_call", host)]
    assert host.tasks == [asyncio.current_task()]
    assert _WRITTEN.get() == "written by the host"
    assert rust.executions[0].closed


async def test_a_host_on_a_rust_that_cannot_serve_host_calls_is_a_type_error() -> None:
    logger: Final = rust_logger(Unhosted(), host=Host())

    assert not isinstance(Unhosted(), HostedRust)
    with pytest.raises(TypeError, match="Unhosted cannot serve host calls"):
        _ = await _pre_call(logger)
