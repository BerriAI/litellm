import asyncio
import copy
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER, LoggingWorker


async def drain_logging(worker: LoggingWorker = GLOBAL_LOGGING_WORKER) -> None:
    await asyncio.sleep(0)
    worker.start()
    await asyncio.wait_for(worker.flush(), timeout=10)


@dataclass(frozen=True, slots=True)
class HookEvent:
    name: str
    call_type: str | None
    thread: threading.Thread
    loop: asyncio.AbstractEventLoop | None
    kwargs: object
    response: object


class RecordingLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self._events: list[HookEvent] = []
        self._condition = threading.Condition()

    @property
    def events(self) -> tuple[HookEvent, ...]:
        with self._condition:
            return tuple(self._events)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(event.name for event in self.events)

    def _record(self, name: str, kwargs: object = None, response: object = None) -> None:
        details: Final = kwargs if isinstance(kwargs, dict) else {}
        try:
            snapshot: Final = copy.deepcopy(details)
        except Exception:
            snapshot = dict(details)
        if "exception" in details:
            snapshot["exception"] = details["exception"]
        try:
            loop: Final = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        event: Final = HookEvent(
            name=name,
            call_type=details.get("call_type"),
            thread=threading.current_thread(),
            loop=loop,
            kwargs=snapshot,
            response=response,
        )
        with self._condition:
            self._events.append(event)
            self._condition.notify_all()

    def wait_for(self, name: str, count: int = 1, timeout: float = 10) -> tuple[HookEvent, ...]:
        deadline: Final = time.monotonic() + timeout
        with self._condition:
            while sum(event.name == name for event in self._events) < count:
                remaining: Final = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for {count} {name} events; saw {self.names}")
                self._condition.wait(remaining)
            return tuple(event for event in self._events if event.name == name)

    async def wait_for_async(self, name: str, count: int = 1, timeout: float = 10) -> tuple[HookEvent, ...]:
        await asyncio.wait_for(asyncio.to_thread(self.wait_for, name, count, timeout), timeout=timeout + 1)
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=timeout)
        return tuple(event for event in self.events if event.name == name)

    def log_pre_api_call(self, model, messages, kwargs):
        self._record("log_pre_api_call", kwargs)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record("log_success_event", kwargs, response_obj)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record("async_log_success_event", kwargs, response_obj)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._record("log_failure_event", kwargs, response_obj)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._record("async_log_failure_event", kwargs, response_obj)

    def logging_hook(self, kwargs, result, call_type):
        self._record("logging_hook", kwargs, result)
        return kwargs, result


class HookRecorder(RecordingLogger):
    def log_post_api_call(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self._record("log_post_api_call", kwargs, response_obj)

    async def async_logging_hook(
        self, kwargs: dict[str, object], result: object, call_type: str
    ) -> tuple[dict[str, object], object]:
        self._record("async_logging_hook", kwargs, result)
        return kwargs, result

    def redact_standard_logging_payload_from_model_call_details(
        self, model_call_details: dict[str, object]
    ) -> dict[str, object]:
        self._record("redact_standard_logging_payload_from_model_call_details", model_call_details)
        return model_call_details

    async def async_pre_call_deployment_hook(self, kwargs: dict[str, object], call_type: object) -> dict[str, object]:
        self._record("async_pre_call_deployment_hook", kwargs)
        return kwargs

    async def async_post_call_success_deployment_hook(
        self, request_data: dict[str, object], response: object, call_type: object
    ) -> object:
        self._record("async_post_call_success_deployment_hook", request_data, response)
        return response

    async def async_post_call_failure_deployment_hook(
        self,
        request_data: dict[str, object],
        original_exception: Exception,
        call_type: object,
        *,
        fallback_depth: int | None = None,
    ) -> None:
        self._record("async_post_call_failure_deployment_hook", request_data, original_exception)

    async def log_model_group_rate_limit_error(
        self, exception: Exception, original_model_group: str | None, kwargs: dict[str, object]
    ) -> None:
        self._record("log_model_group_rate_limit_error", kwargs, exception)


def request_body(kwargs: dict[str, object]) -> dict[str, object]:
    additional_args = kwargs["additional_args"]
    assert isinstance(additional_args, dict)
    body = additional_args["complete_input_dict"]
    assert isinstance(body, dict)
    return body


def request_headers(kwargs: dict[str, object]) -> dict[str, object]:
    additional_args = kwargs["additional_args"]
    assert isinstance(additional_args, dict)
    headers = additional_args["headers"]
    assert isinstance(headers, dict)
    return headers
