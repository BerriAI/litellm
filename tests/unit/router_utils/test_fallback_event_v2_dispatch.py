import asyncio
import threading
from collections.abc import Awaitable, Callable, Generator
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_logger_v2 import CallContext, CustomLoggerV2, Field, Hook, on
from litellm.router_utils import fallback_event_handlers as handlers

EventLogger = Callable[[str, dict[str, object], Exception], Awaitable[None]]
EVENTS: Final = (
    ("log_success_fallback_event", "fallback_success"),
    ("log_failure_fallback_event", "fallback_failure"),
)
CALLBACK_LISTS: Final = (
    "callbacks",
    "success_callback",
    "failure_callback",
    "_async_success_callback",
    "_async_failure_callback",
)


@pytest.fixture(autouse=True)
def empty_callback_lists(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    for name in CALLBACK_LISTS:
        monkeypatch.setattr(litellm, name, [])
    yield


class Recorder(CustomLogger):
    def __init__(
        self, name: str, journal: list[object], error: Exception | None = None
    ) -> None:  # mutable-ok: the shared call log
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # CustomLogger's untyped **kwargs
        self.name = name
        self.journal = journal
        self.error = error
        self.calls: list[tuple[str, str, dict[str, object], Exception]] = []  # mutable-ok: each call's arguments

    async def _record(
        self, event: str, group: str, kwargs: dict[str, object], exception: Exception
    ) -> None:  # mutable-ok: the handler's kwargs dict
        self.journal.append(self.name)
        self.calls.append((event, group, kwargs, exception))
        await asyncio.sleep(0)
        if self.error is not None:
            raise self.error

    async def log_success_fallback_event(
        self, original_model_group: str, kwargs: dict[str, object], original_exception: Exception
    ) -> None:  # mutable-ok: the handler's kwargs dict
        await self._record("success", original_model_group, kwargs, original_exception)

    async def log_failure_fallback_event(
        self, original_model_group: str, kwargs: dict[str, object], original_exception: Exception
    ) -> None:  # mutable-ok: the handler's kwargs dict
        await self._record("failure", original_model_group, kwargs, original_exception)


class Watcher(CustomLoggerV2):
    def __init__(self, journal: list[object], done: threading.Event) -> None:  # mutable-ok: the shared call log
        self.journal = journal
        self.done = done

    @on(Hook.router_event, reads={Field.model, Field.outcome, Field.error})
    async def watch(self, view: object, ctx: CallContext[object]) -> None:
        self.journal.append(
            ("v2", getattr(view, "model"), getattr(view, "outcome"), str(getattr(view, "error")))  # noqa: B009  # the view's declared fields
        )
        self.done.set()


def _public(name: str) -> EventLogger:
    found: Final[EventLogger] = getattr(handlers, name)  # pyright: ignore[reportAny]  # the module's async handlers
    return found


def _with_inert(callbacks: list[object], inert: bool) -> list[object]:  # mutable-ok: a process list
    return [*callbacks, Watcher([], threading.Event())] if inert else callbacks


@pytest.mark.parametrize("inert", [False, True])
@pytest.mark.parametrize(("public", "_outcome"), EVENTS)
async def test_every_logger_in_the_merged_lists_runs_once_in_order_with_the_same_arguments(
    public: str, _outcome: str, inert: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal: Final[list[object]] = []  # mutable-ok: the shared call log
    first: Final = Recorder("first", journal)
    second: Final = Recorder("second", journal)
    third: Final = Recorder("third", journal)
    monkeypatch.setattr(litellm, "callbacks", _with_inert([first, "langfuse", second], inert))
    monkeypatch.setattr(litellm, "_async_success_callback", [second, third, first])
    kwargs: Final[dict[str, object]] = {"model": "fallback"}  # mutable-ok: the handler's kwargs dict
    error: Final = RuntimeError("original")

    result: Final = await _public(public)("primary", kwargs, error)

    assert result is None
    assert journal == ["first", "second", "third"]
    event: Final = "success" if public.startswith("log_success") else "failure"
    assert first.calls == [(event, "primary", kwargs, error)]
    assert first.calls[0][2] is kwargs
    assert first.calls[0][3] is error


@pytest.mark.parametrize("inert", [False, True])
@pytest.mark.parametrize(("public", "_outcome"), EVENTS)
async def test_a_logger_error_is_logged_and_the_next_logger_still_runs(
    public: str, _outcome: str, inert: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal: Final[list[object]] = []  # mutable-ok: the shared call log
    logged: Final[list[tuple[object, ...]]] = []  # mutable-ok: the router logger's error calls
    monkeypatch.setattr(handlers.verbose_router_logger, "error", lambda *args: logged.append(args))
    error: Final = ValueError("logger raised")
    monkeypatch.setattr(
        litellm, "callbacks", _with_inert([Recorder("broken", journal, error), Recorder("next", journal)], inert)
    )

    await _public(public)("primary", {}, RuntimeError("original"))

    assert journal == ["broken", "next"]
    assert logged == [(f"Error in {public}: %s", error)]


@pytest.mark.parametrize(("public", "outcome"), EVENTS)
async def test_a_v2_callback_in_the_lists_is_delivered_off_path_after_the_legacy_loggers(
    public: str, outcome: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal: Final[list[object]] = []  # mutable-ok: the shared call log
    done: Final = threading.Event()
    monkeypatch.setattr(litellm, "callbacks", [Recorder("before", journal), Watcher(journal, done)])
    monkeypatch.setattr(litellm, "_async_failure_callback", [Recorder("after", journal)])

    await _public(public)("primary", {"model": "fallback"}, RuntimeError("original"))

    assert await asyncio.to_thread(done.wait, 30)
    assert journal == ["before", "after", ("v2", "primary", outcome, "original")]
