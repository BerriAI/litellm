from __future__ import annotations

import asyncio
import contextvars
import datetime
from collections.abc import Awaitable, Callable, Coroutine
from concurrent.futures import Future
from typing import Protocol


class CompletionLogging(Protocol):
    def success_handler(
        self,
        result: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...

    def failure_handler(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...

    def async_success_handler(
        self,
        result: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> Coroutine[object, object, None]: ...

    def async_failure_handler(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> Awaitable[None]: ...

    def handle_sync_success_callbacks_for_async_calls(
        self,
        result: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...


class CompletionExecutor(Protocol):
    def submit(
        self,
        function: Callable[..., object],
        *args: object,
    ) -> Future[object]: ...


class Completion(Protocol):
    def success(
        self,
        result: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...

    def failure(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...

    async def async_failure(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...


class PythonCompletion:
    def __init__(
        self,
        logging_obj: CompletionLogging,
        executor: CompletionExecutor | None,
        *,
        async_call: bool,
        internal_call: bool,
        completion_with_fallbacks: bool,
    ) -> None:
        self._logging_obj = logging_obj
        self._executor = executor
        self._async_call = async_call
        self._internal_call = internal_call
        self._completion_with_fallbacks = completion_with_fallbacks

    def success(
        self,
        result: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        if not self._async_call:
            assert self._executor is not None
            context = contextvars.copy_context()
            self._executor.submit(
                context.run,
                self._logging_obj.success_handler,
                result,
                start_time,
                end_time,
            )
            return

        if not self._internal_call:
            if getattr(self._logging_obj, "_defer_async_logging", False):

                def enqueue_deferred_logging() -> None:
                    asyncio.create_task(self._dispatch_async_success(result, start_time, end_time))

                setattr(  # noqa: B010  # optional legacy logger field is absent from narrow test doubles
                    self._logging_obj,
                    "_enqueue_deferred_logging",
                    enqueue_deferred_logging,
                )
            else:
                asyncio.create_task(self._dispatch_async_success(result, start_time, end_time))

        self._logging_obj.handle_sync_success_callbacks_for_async_calls(
            result=result,
            start_time=start_time,
            end_time=end_time,
        )

    def failure(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        if self._async_call and self._internal_call:
            return
        self._logging_obj.failure_handler(exception, traceback_exception, start_time, end_time)

    async def async_failure(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        if not self._async_call or self._internal_call:
            return
        await self._logging_obj.async_failure_handler(exception, traceback_exception, start_time, end_time)

    async def _dispatch_async_success(
        self,
        result: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        if self._completion_with_fallbacks:
            return
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(  # pyright: ignore[reportUnknownMemberType]  # legacy worker lacks generic coroutine annotations
            async_coroutine=self._logging_obj.async_success_handler(
                result=result,
                start_time=start_time,
                end_time=end_time,
            )
        )
        self._logging_obj.handle_sync_success_callbacks_for_async_calls(
            result=result,
            start_time=start_time,
            end_time=end_time,
        )


class CallCompletion:
    def __init__(self, implementation: Completion) -> None:
        self._python_implementation: Completion | None = implementation
        self._implementation: Completion | None = implementation
        self._attached = False

    @property
    def python_implementation(self) -> Completion:
        assert self._python_implementation is not None
        return self._python_implementation

    def attach(self, implementation: Completion) -> bool:
        if self._attached:
            return False
        self._implementation = implementation
        self._attached = True
        return True

    def release(self) -> None:
        self._python_implementation = None
        self._implementation = None

    def success(
        self,
        result: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        implementation = self._implementation
        assert implementation is not None
        implementation.success(result, start_time, end_time)

    def failure(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        implementation = self._implementation
        assert implementation is not None
        implementation.failure(exception, traceback_exception, start_time, end_time)

    async def async_failure(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        implementation = self._implementation
        assert implementation is not None
        await implementation.async_failure(
            exception,
            traceback_exception,
            start_time,
            end_time,
        )
