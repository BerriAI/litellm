import asyncio
import contextvars
import datetime
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Final, Protocol, cast

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils import litellm_logging, logging_worker, thread_pool_executor
from litellm.rust_bridge.callbacks_legacy_python import setup
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse
from tests.test_litellm_rust.support.isolation import isolated_callback_registries, rebound

marker: Final = contextvars.ContextVar("callback-contract", default="outside")


@dataclass(frozen=True, slots=True)
class Event:
    name: str
    value: object
    thread: int
    task: int | None
    context: str


def current_task_id() -> int | None:
    try:
        task: Final = asyncio.current_task()
    except RuntimeError:
        return None
    return id(task) if task is not None else None


class Recorder(CustomLogger):
    def __init__(self, fail_at: str = "", cancellation: bool = False) -> None:
        super().__init__()
        self.events: tuple[Event, ...] = ()
        self.lock: Final = threading.Lock()
        self.fail_at: Final = fail_at
        self.failure: Final = (
            asyncio.CancelledError("callback cancelled") if cancellation else ValueError("callback failed")
        )

    def record(self, name: str, value: object = None) -> None:
        with self.lock:
            self.events = (
                *self.events,
                Event(name, value, threading.get_ident(), current_task_id(), marker.get()),
            )
        if name == self.fail_at:
            raise self.failure

    def log_pre_api_call(self, model: str, messages: object, kwargs: dict[str, object]) -> None:
        self.record("pre_api")

    def log_post_api_call(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime | None,
    ) -> None:
        self.record("post_api")

    def log_success_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        self.record("sync_success", response_obj)

    async def async_log_success_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        self.record("async_success", response_obj)

    def log_failure_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        self.record("sync_failure", kwargs["exception"])

    async def async_log_failure_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        self.record("async_failure", kwargs["exception"])

    async def async_pre_call_deployment_hook(self, kwargs: dict[str, object], call_type: object) -> None:
        self.record("deployment_pre")

    async def async_post_call_success_deployment_hook(
        self, request_data: dict[str, object], response: object, call_type: object
    ) -> None:
        self.record("deployment_success", response)

    async def async_post_call_failure_deployment_hook(
        self,
        request_data: Mapping[str, object],
        exception: Exception,
        call_type: object,
        fallback_depth: int | None = None,
    ) -> None:
        self.record("deployment_failure", exception)

    async def async_pre_request_hook(self, model: str, messages: object, kwargs: dict[str, object]) -> None:
        self.record("pre_request")


class RequestEditor(CustomLogger):
    def __init__(self, field: str, returns: str) -> None:
        super().__init__()
        self.field: Final = field
        self.returns: Final = returns
        self.seen: object = None

    async def async_pre_request_hook(
        self, model: str, messages: object, kwargs: dict[str, object]
    ) -> dict[str, object] | None:
        if self.field == "messages":
            typed: Final = cast(list[dict[str, object]], messages)
            typed[0]["content"] = "edited"
        edited: Final = {**kwargs, "tools": [{"name": "edited", "input_schema": {"type": "object"}}]}
        if self.field == "tools" and self.returns != "replacement":
            kwargs["tools"] = edited["tools"]  # rebind-ok: exercises in-place callback mutation
        view: Final = edited if self.field == "tools" and self.returns == "replacement" else kwargs
        self.seen = messages if self.field == "messages" else view["tools"]
        return None if self.returns == "none" else dict(view) if self.returns == "replacement" else view


class RequestObserver(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.messages: object = None
        self.tools: object = None

    async def async_pre_request_hook(self, model: str, messages: object, kwargs: dict[str, object]) -> None:
        self.messages = messages
        self.tools = kwargs.get("tools")


class WireEditor(CustomLogger):
    def __init__(self, field: str, replace: bool) -> None:
        super().__init__()
        self.field: Final = field
        self.replace: Final = replace
        self.retained: object = None

    def log_pre_api_call(self, model: str, messages: object, kwargs: dict[str, object]) -> None:
        additional: Final = cast(dict[str, object], kwargs["additional_args"])
        target: Final = cast(dict[str, object], additional[self.field])
        self.retained = target
        if self.replace:
            additional[self.field] = {"x-contract": "edited"}
        else:
            target["x-contract"] = "edited"


class WireObserver(CustomLogger):
    def __init__(self, field: str) -> None:
        super().__init__()
        self.field: Final = field
        self.seen: object = None

    def log_pre_api_call(self, model: str, messages: object, kwargs: dict[str, object]) -> None:
        additional: Final = cast(dict[str, object], kwargs["additional_args"])
        self.seen = additional[self.field]


class ResponseEditor(CustomLogger):
    def __init__(self, replacement: AnthropicMessagesResponse) -> None:
        super().__init__()
        self.replacement: Final = replacement

    async def async_post_call_success_deployment_hook(
        self, request_data: dict[str, object], response: object, call_type: object
    ) -> AnthropicMessagesResponse:
        return self.replacement


class ResponseBlocker(CustomGuardrail):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error: Final = error

    async def async_post_call_success_deployment_hook(
        self, request_data: dict[str, object], response: object, call_type: object
    ) -> None:
        raise self.error


class ClosableStream(AsyncIterator[object], Protocol):
    async def aclose(self) -> None: ...


class ClosableIterator(Iterator[object], Protocol):
    def close(self) -> None: ...


class Scenario:
    def __init__(
        self, kwargs: dict[str, object], asynchronous: bool, registration: str, fail_at: str, cancellation: bool
    ) -> None:
        self.stack: Final = ExitStack()
        self.stack.enter_context(isolated_callback_registries())
        self.stack.enter_context(rebound(litellm, "cache", None))
        self.stack.enter_context(rebound(litellm, "max_budget", None))
        price: Final = {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "max_tokens": 1024,
            "input_cost_per_token": 0.01,
            "output_cost_per_token": 0.02,
        }
        self.stack.enter_context(
            rebound(
                litellm,
                "model_cost",
                {
                    **litellm.model_cost,
                    str(kwargs["model"]): price,
                    f"anthropic/{kwargs['model']}": price,
                },
            )
        )
        self.executor: Final = ThreadPoolExecutor(max_workers=2)
        for module in (litellm_logging, thread_pool_executor, litellm.utils):
            self.stack.enter_context(rebound(module, "executor", self.executor))
        self.worker: Final = logging_worker.GLOBAL_LOGGING_WORKER
        self.runner: Final = asyncio.Runner()
        self.recorder: Final = Recorder(fail_at, cancellation)
        if registration in ("global", "both"):
            litellm.callbacks.append(self.recorder)
        prepared: Final = {**kwargs, **({"callbacks": [self.recorder]} if registration in ("request", "both") else {})}
        self.setup: Final = setup("anthropic_messages", (), prepared, datetime.datetime.now(), asynchronous)
        self.kwargs: Final = {**self.setup.kwargs, "litellm_logging_obj": self.setup.logger}
        self.provider_error: Final = litellm.BadRequestError(
            "provider failed", model=str(kwargs["model"]), llm_provider="anthropic"
        )
        self.output: object = None
        self.error: BaseException | None = None
        self.caller_thread: Final = threading.get_ident()
        self.caller_task: int | None = None
        self.chunks: tuple[object, ...] = ()
        self.before_release: tuple[str, ...] = ()
        self.before_finish: tuple[str, ...] = ()
        self.blocked: Final = ValueError("response blocked")
        self.deferred: bool = False
        self.releases: tuple[bool, ...] = ()
        self.editor: object = None
        self.observer: object = None

    def request_edit(self, field: str, returns: str) -> None:
        self.editor = RequestEditor(field, returns)
        self.observer = RequestObserver()
        litellm.callbacks.extend((self.editor, self.observer))

    def wire_edit(self, field: str, replace: bool) -> None:
        self.editor = WireEditor(field, replace)
        self.observer = WireObserver(field)
        litellm.callbacks.extend((self.editor, self.observer))
        self.setup.logger.dynamic_input_callbacks = [self.editor, self.observer, self.recorder]

    def response_edit(self, replacement: AnthropicMessagesResponse) -> None:
        self.editor = ResponseEditor(replacement)
        litellm.callbacks.insert(0, self.editor)

    def block_response(self) -> None:
        litellm.callbacks.append(ResponseBlocker(self.blocked))

    async def drain(self) -> None:
        self.worker.start()
        await asyncio.wait_for(self.worker.flush(), timeout=10)

    async def run_async(self, call: Callable[[], Awaitable[object]], finish: str) -> None:
        self.caller_task = id(asyncio.current_task())
        token: Final = marker.set("caller")
        self.setup.logger._defer_async_logging = self.deferred
        try:
            self.output = await call()
            if finish != "complete":
                stream: Final = cast(ClosableStream, self.output)
                self.chunks = (await anext(stream),)
                await self.drain()
                self.before_finish = tuple(event.name for event in self.recorder.events)
                if self.deferred:
                    self.setup.logger._on_deferred_stream_complete = self.enqueue
                if finish == "consume":
                    self.chunks = (*self.chunks, *[chunk async for chunk in stream])
                elif finish == "close":
                    await stream.aclose()
                    await stream.aclose()
                elif finish == "disconnect":
                    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

                    await ProxyBaseLLMRequestProcessing._finalize_streaming_generator_cleanup(
                        request=None, request_data=self.kwargs, response=stream, client_disconnected=True
                    )
        except BaseException as error:
            self.error = error
        finally:
            await self.drain()
            self.before_release = tuple(event.name for event in self.recorder.events)
            from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

            for accepted in self.releases:
                ProxyBaseLLMRequestProcessing._flush_deferred_async_logging(self.setup.logger, not accepted)
            await self.drain()
            marker.reset(token)

    async def enqueue(self, coroutine: object) -> None:
        await cast(Awaitable[object], coroutine)

    def execute(self, call: Callable[[], object], asynchronous: bool, finish: str) -> None:
        if asynchronous:
            self.runner.run(self.run_async(cast(Callable[[], Awaitable[object]], call), finish))
        else:
            token: Final = marker.set("caller")
            try:
                self.output = call()
                self.before_finish = tuple(event.name for event in self.recorder.events)
                if finish == "consume":
                    self.chunks = tuple(cast(Iterator[object], self.output))
                elif finish == "close":
                    iterator: Final = cast(ClosableIterator, self.output)
                    self.chunks = (next(iterator),)
                    iterator.close()
            except BaseException as error:
                self.error = error
            finally:
                marker.reset(token)
            self.runner.run(self.drain())
        self.executor.shutdown(wait=True)

    def close(self) -> None:
        try:
            self.runner.run(self.worker.stop())
            pending: Final = getattr(self.setup.logger, "_deferred_stream_complete_args", ()) or ()
            for coroutine in pending:
                coroutine.close()
            self.executor.shutdown(wait=True)
        finally:
            self.runner.close()
            self.stack.close()
