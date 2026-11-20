import asyncio
import atexit
import time
from collections.abc import Mapping
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import Future
from datetime import datetime
from functools import partial
from typing import Final, TypeVar

from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.kafka.config import resolve_params
from litellm.integrations.kafka.producer import (
    KafkaExportStats,
    KafkaSink,
    ProducerFactory,
    interpreter_exiting,
    load_producer_factory,
    prepare_interpreter_exit,
)
from litellm.integrations.kafka.worker import KafkaCallbackWorker
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.litellm_core_utils.redact_messages import (
    redacted_standard_logging_payload,
    should_redact_message_logging,  # pyright: ignore[reportUnknownVariableType]  # legacy helper takes untyped kwargs
)
from litellm.types.integrations.kafka import KafkaInitParams

_MAPPING: Final = TypeAdapter(Mapping[str, object])
_T: Final = TypeVar("_T")


async def _wait_until(future: Future[_T], deadline: float) -> bool:
    if future.cancelled():
        return True
    if future.done():
        try:
            _ = future.result()
        except FutureCancelledError:
            return True
        return True
    try:
        _ = await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), max(0.0, deadline - time.monotonic()))
    except TimeoutError:
        return False
    except asyncio.CancelledError:
        if future.cancelled():
            return True
        raise
    return True


class KafkaLogger(CustomLogger):
    def __init__(
        self,
        params: KafkaInitParams | None = None,
        producer_factory: ProducerFactory | None = None,
        *,
        start_polling: bool = True,
    ) -> None:
        configured: Final = params if params is not None else litellm.kafka_params
        resolved: Final = resolve_params(configured)
        super().__init__(turn_off_message_logging=bool(resolved.turn_off_message_logging))  # pyright: ignore[reportUnknownMemberType]  # legacy base accepts untyped extra kwargs
        self.sink: Final = KafkaSink(
            resolved,
            producer_factory if producer_factory is not None else load_producer_factory(),
            self._callback_failed,
            start_polling=start_polling,
        )
        self._worker: Final = KafkaCallbackWorker(resolved.callback_queue_max_messages, self.sink.drop)
        atexit.register(self._close_on_exit)

    def _callback_failed(self) -> None:
        self.handle_callback_failure(callback_name="KafkaLogger")

    def redacts_messages_itself(self) -> bool:
        return True

    @property
    def statistics(self) -> KafkaExportStats:
        return self.sink.statistics

    def _log(self, kwargs: Mapping[str, object]) -> None:
        try:
            if kwargs.get("standard_logging_object") is None:
                return
            details: Final = _MAPPING.validate_python(
                self.redact_standard_logging_payload_from_model_call_details(dict(kwargs))  # pyright: ignore[reportUnknownMemberType]  # TypeAdapter validates the legacy helper result below
            )
            payload: Final = _MAPPING.validate_python(details.get("standard_logging_object"))
            redacted: Final = (
                redacted_standard_logging_payload(payload)
                if self.sink.params.turn_off_message_logging or should_redact_message_logging(dict(kwargs))
                else payload
            )
            self.sink.publish(redacted)
        except Exception:  # noqa: BLE001  # malformed log data must not change a model response
            self.sink.drop("invalid_payload")

    def log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self._log(kwargs)

    def log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self._log(kwargs)

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await self._async_log(kwargs)

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await self._async_log(kwargs)

    async def _async_log(self, kwargs: Mapping[str, object]) -> None:
        if kwargs.get("standard_logging_object") is None:
            return
        if interpreter_exiting():
            self._log(kwargs)
            return
        finished: Final = self._worker.submit(partial(self._log, kwargs))
        if finished is not None:
            try:
                await asyncio.wait_for(asyncio.wrap_future(finished), self.sink.params.callback_timeout_ms / 1000)
            except TimeoutError:
                self.sink.preparation_timed_out()

    async def flush_queue(self) -> None:
        deadline: Final = time.monotonic() + self.sink.params.flush_timeout
        for future in self._worker.pending():
            if not await _wait_until(future, deadline):
                self._shutdown_timeout()
                return
        if not await _wait_until(self.sink.request_flush(max(0.0, deadline - time.monotonic())), deadline):
            self._shutdown_timeout()

    def _shutdown_timeout(self) -> None:
        self._callback_failed()
        verbose_logger.warning("Kafka logging callback drain or flush reached its deadline")

    async def aclose(self, timeout: float | None = None) -> None:
        deadline: Final = time.monotonic() + self.sink.budget(timeout)
        if not await _wait_until(self._worker.stop(deadline), deadline):
            self._shutdown_timeout()
        self._worker.discard_pending()
        if not await _wait_until(self.sink.request_close(max(0.0, deadline - time.monotonic())), deadline):
            self._shutdown_timeout()
        atexit.unregister(self._close_on_exit)

    def close(self, timeout: float | None = None) -> None:
        deadline: Final = time.monotonic() + self.sink.budget(timeout)
        try:
            self._worker.stop(deadline).result(timeout=max(0.0, deadline - time.monotonic()))
        except TimeoutError:
            self._shutdown_timeout()
        self._worker.discard_pending()
        self.sink.close(max(0.0, deadline - time.monotonic()))
        atexit.unregister(self._close_on_exit)

    def _close_on_exit(self) -> None:
        prepare_interpreter_exit()
        try:
            GLOBAL_LOGGING_WORKER._flush_on_exit()  # pyright: ignore[reportPrivateUsage]  # drain SDK callbacks before closing their destination
        finally:
            self.close()
