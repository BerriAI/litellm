import contextvars
import datetime
from collections.abc import Callable, Coroutine
from concurrent.futures import Future
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.litellm_core_utils.call_completion import CallCompletion, PythonCompletion


class RecordingExecutor:
    def __init__(self) -> None:
        self.submissions: list[tuple[Callable[..., object], tuple[object, ...]]] = []

    def submit(self, function: Callable[..., object], *args: object) -> Future[object]:
        self.submissions.append((function, args))
        future: Final[Future[object]] = Future()
        future.set_result(function(*args))
        return future


class RecordingCompletion:
    def __init__(self) -> None:
        self.successes: list[object] = []
        self.failures: list[Exception] = []

    def success(
        self,
        result: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        self.successes.append(result)

    def failure(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        self.failures.append(exception)

    async def async_failure(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        self.failures.append(exception)


class RecordingLogging:
    def __init__(self, marker: contextvars.ContextVar[str], observed: list[tuple[object, str]]) -> None:
        self._marker = marker
        self._observed = observed
        self._defer_async_logging = False
        self._enqueue_deferred_logging: Callable[[], None] | None = None

    def success_handler(
        self,
        result: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        self._observed.append((result, self._marker.get()))

    def failure_handler(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...

    async def async_success_handler(
        self,
        result: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...

    async def async_failure_handler(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...

    def handle_sync_success_callbacks_for_async_calls(
        self,
        result: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...


def test_python_completion_preserves_sync_context_and_response_identity() -> None:
    marker: Final[contextvars.ContextVar[str]] = contextvars.ContextVar("marker")
    marker.set("request-context")
    executor: Final = RecordingExecutor()
    response: Final = object()
    observed: Final[list[tuple[object, str]]] = []
    logging_obj: Final = RecordingLogging(marker, observed)
    completion: Final = PythonCompletion(
        logging_obj,
        executor,
        async_call=False,
        internal_call=False,
        completion_with_fallbacks=False,
    )
    now: Final = datetime.datetime.now(datetime.timezone.utc)

    completion.success(response, now, now)

    assert observed == [(response, "request-context")]
    assert len(executor.submissions) == 1


@pytest.mark.asyncio
async def test_call_completion_attaches_once_and_forwards_final_objects() -> None:
    python_completion: Final = RecordingCompletion()
    native_completion: Final = RecordingCompletion()
    ignored_completion: Final = RecordingCompletion()
    completion: Final = CallCompletion(python_completion)
    response: Final = object()
    error: Final = ValueError("mapped failure")
    now: Final = datetime.datetime.now(datetime.timezone.utc)

    assert completion.python_implementation is python_completion
    assert completion.attach(native_completion)
    assert not completion.attach(ignored_completion)

    completion.success(response, now, now)
    completion.failure(error, "traceback", now, now)
    await completion.async_failure(error, "traceback", now, now)

    assert native_completion.successes == [response]
    assert native_completion.failures == [error, error]
    assert python_completion.successes == []
    assert ignored_completion.successes == []


@pytest.mark.asyncio
async def test_python_completion_retains_deferred_success_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    response: Final = object()
    logging_obj: Final = MagicMock()
    logging_obj._defer_async_logging = True
    logging_obj.async_success_handler = AsyncMock()
    completion: Final = PythonCompletion(
        logging_obj,
        RecordingExecutor(),
        async_call=True,
        internal_call=False,
        completion_with_fallbacks=False,
    )
    scheduled: Final[list[Coroutine[object, object, None]]] = []
    monkeypatch.setattr("asyncio.create_task", scheduled.append)
    now: Final = datetime.datetime.now(datetime.timezone.utc)

    completion.success(response, now, now)

    logging_obj._enqueue_deferred_logging()
    assert len(scheduled) == 1
    scheduled[0].close()
