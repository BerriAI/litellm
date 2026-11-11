import asyncio
import threading
from collections.abc import Awaitable, Callable, Generator
from typing import Final

import pytest

from litellm.integrations.custom_logger_v2 import CallContext, CustomLoggerV2, Hook, build_manifest, on
from litellm.integrations.custom_logger_v2.host import has_natives, register_native, reset_host_for_tests
from litellm.router_utils import fallback_event_handlers as handlers

EventLogger = Callable[[str, dict[str, object], Exception], Awaitable[None]]
EVENTS: Final = (
    ("log_success_fallback_event", "_log_success_fallback_event"),
    ("log_failure_fallback_event", "_log_failure_fallback_event"),
)


@pytest.fixture(autouse=True)
def fresh_host() -> Generator[None]:
    reset_host_for_tests()
    yield
    reset_host_for_tests()


class Recorded:
    def __init__(self, order: list[str], error: Exception | None = None) -> None:  # mutable-ok: the shared call log
        self.order = order
        self.error = error
        self.calls: list[tuple[str, dict[str, object], Exception]] = []  # mutable-ok: each call's arguments

    async def __call__(
        self, original_model_group: str, kwargs: dict[str, object], original_exception: Exception
    ) -> None:  # mutable-ok: the handler's kwargs dict
        self.calls.append((original_model_group, kwargs, original_exception))
        self.order.append("legacy")
        if self.error is not None:
            raise self.error


class Watcher(CustomLoggerV2):
    def __init__(self, order: list[str], done: threading.Event) -> None:  # mutable-ok: the shared call log
        self.order = order
        self.done = done

    @on(Hook.router_event)
    async def watch(self, view: object, ctx: CallContext[object]) -> None:
        self.order.append("native")
        self.done.set()


def _public(name: str) -> EventLogger:
    found: Final[EventLogger] = getattr(handlers, name)  # pyright: ignore[reportAny]  # the module's async handlers
    return found


@pytest.mark.parametrize(("public", "legacy"), EVENTS)
async def test_without_natives_the_legacy_handler_runs_once_with_the_same_arguments(
    public: str, legacy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: Final = Recorded([])
    monkeypatch.setattr(handlers, legacy, recorded)
    kwargs: Final[dict[str, object]] = {"model": "fallback"}  # mutable-ok: the handler's kwargs dict
    error: Final = RuntimeError("original")

    result: Final = await _public(public)("primary", kwargs, error)

    assert not has_natives(Hook.router_event)
    assert result is None
    assert len(recorded.calls) == 1
    group, seen_kwargs, seen_error = recorded.calls[0]
    assert group == "primary"
    assert seen_kwargs is kwargs
    assert seen_error is error


@pytest.mark.parametrize(("public", "legacy"), EVENTS)
async def test_without_natives_a_legacy_handler_error_reaches_the_caller_unchanged(
    public: str, legacy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    error: Final = ValueError("handler raised")
    monkeypatch.setattr(handlers, legacy, Recorded([], error))

    with pytest.raises(ValueError) as raised:
        await _public(public)("primary", {}, RuntimeError("original"))

    assert raised.value is error


@pytest.mark.parametrize(("public", "legacy"), EVENTS)
async def test_an_off_path_native_is_delivered_after_the_legacy_handler(
    public: str, legacy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: Final[list[str]] = []  # mutable-ok: the shared call log
    done: Final = threading.Event()
    register_native(build_manifest(Watcher(order, done)))
    monkeypatch.setattr(handlers, legacy, Recorded(order))

    await _public(public)("primary", {}, RuntimeError("original"))

    assert await asyncio.to_thread(done.wait, 30)
    assert order == ["legacy", "native"]
