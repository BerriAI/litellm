import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
from typing import Final, Protocol, TypeAlias

from litellm._logging import verbose_logger
from litellm.integrations.kafka.config import KafkaConfigValue, producer_config
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.types.integrations.kafka import KafkaInitParams

_EXITING: Final = threading.Event()


def prepare_interpreter_exit() -> None:
    _EXITING.set()


def interpreter_exiting() -> bool:
    return _EXITING.is_set()


class DeliveryError(Protocol):
    def code(self) -> int: ...


DeliveryCallback: TypeAlias = Callable[[DeliveryError | None, object], None]


class KafkaProducer(Protocol):
    def produce(self, topic: str, *, value: bytes, key: bytes | None, on_delivery: DeliveryCallback) -> None: ...

    def poll(self, timeout: float) -> int: ...

    def flush(self, timeout: float) -> int: ...


ProducerFactory: TypeAlias = Callable[[Mapping[str, KafkaConfigValue]], KafkaProducer]
FailureReporter: TypeAlias = Callable[[], None]


def load_producer_factory() -> ProducerFactory:
    try:
        from confluent_kafka import Producer
    except ImportError:
        raise ImportError("Kafka logging requires the optional dependency. Install litellm[kafka].") from None
    return lambda settings: Producer(dict(settings))


@dataclass(frozen=True, slots=True)
class KafkaExportStats:
    accepted: int = 0
    delivered: int = 0
    dropped: int = 0
    delivery_failed: int = 0
    unconfirmed: int = 0
    callback_timeouts: int = 0


@dataclass(frozen=True, slots=True)
class FlushRequest:
    finished: Future[int | None]
    deadline: float


class KafkaSink:
    def __init__(
        self,
        params: KafkaInitParams,
        factory: ProducerFactory,
        report_failure: FailureReporter,
        *,
        start_polling: bool = True,
    ) -> None:
        self.params: Final = params
        self._factory: Final = factory
        self._report_failure: Final = report_failure
        self._start_polling: Final = start_polling
        self._lock: Final = threading.Lock()
        self._stats_lock: Final = threading.Lock()
        self._close_lock: Final = threading.Lock()
        self._finalize_lock: Final = threading.Lock()
        self._flush_lock: Final = threading.Lock()
        self._stop: Final = threading.Event()
        self._closed: Final[Future[None]] = Future()
        self._producer: KafkaProducer | None = None
        self._poll_thread: threading.Thread | None = None
        self._close_deadline: float | None = None
        self._flush_request: FlushRequest | None = None
        self._stats: KafkaExportStats = KafkaExportStats()

    @property
    def statistics(self) -> KafkaExportStats:
        with self._stats_lock:
            return self._stats

    def _update(
        self,
        *,
        accepted: int = 0,
        delivered: int = 0,
        dropped: int = 0,
        delivery_failed: int = 0,
        unconfirmed: int = 0,
        callback_timeouts: int = 0,
    ) -> KafkaExportStats:
        with self._stats_lock:
            updated: Final = replace(
                self._stats,
                accepted=self._stats.accepted + accepted,
                delivered=self._stats.delivered + delivered,
                dropped=self._stats.dropped + dropped,
                delivery_failed=self._stats.delivery_failed + delivery_failed,
                unconfirmed=self._stats.unconfirmed + unconfirmed,
                callback_timeouts=self._stats.callback_timeouts + callback_timeouts,
            )
            self._stats = updated
            return updated

    def drop(self, reason: str) -> None:
        stats: Final = self._update(dropped=1, delivery_failed=int(reason == "delivery_failed"))
        self._report_failure()
        if stats.dropped & (stats.dropped - 1) == 0:
            verbose_logger.warning("Kafka logging dropped an event reason=%s total_dropped=%d", reason, stats.dropped)

    def preparation_timed_out(self) -> None:
        stats: Final = self._update(callback_timeouts=1)
        self._report_failure()
        if stats.callback_timeouts & (stats.callback_timeouts - 1) == 0:
            verbose_logger.warning(
                "Kafka logging preparation deadline reached total_timeouts=%d", stats.callback_timeouts
            )

    def publish(self, payload: Mapping[str, object]) -> None:
        if self._stop.is_set():
            self.drop("closed")
            return
        try:
            body: Final = safe_dumps(dict(payload), non_finite_as_null=True).encode("utf-8")
            if len(body) > self.params.message_max_bytes:
                self.drop("message_too_large")
                return
            identity: Final = payload.get("litellm_call_id") or payload.get("id")
            key: Final = identity.encode("utf-8") if isinstance(identity, str) else None
            reason: Final = self._enqueue(body, key)
            if reason is not None:
                self.drop(reason)
        except BufferError:
            self.drop("queue_full")
        except Exception:  # noqa: BLE001  # exporter failures must not change the model response
            self.drop("producer_error")

    def _enqueue(self, body: bytes, key: bytes | None) -> str | None:
        with self._lock:
            try:
                if self._stop.is_set():
                    return "closed"
                producer: Final = self._producer if self._producer is not None else self._create_producer()
                if self._stop.is_set():
                    return "closed"
                producer.produce(self.params.topic, value=body, key=key, on_delivery=self._delivered)
                _ = self._update(accepted=1)
            finally:
                closing: Final = self._producer
                if self._stop.is_set() and closing is not None:
                    self._finalize(closing)
        return None

    def _create_producer(self) -> KafkaProducer:
        producer: Final = self._factory(producer_config(self.params))
        self._producer = producer
        if self._stop.is_set():
            self._finalize(producer)
            return producer
        if self._start_polling and not interpreter_exiting():
            thread: Final = threading.Thread(
                target=self._poll, args=(producer,), name="litellm-kafka-poll", daemon=True
            )
            try:
                thread.start()
            except RuntimeError:
                _ = self.request_close(0)
                self._finalize(producer)
                raise
            self._poll_thread = thread
        return producer

    def _poll(self, producer: KafkaProducer) -> None:
        try:
            while not self._stop.is_set():
                self._run_flush_request(producer)
                _ = producer.poll(0.1)
        except Exception:  # noqa: BLE001  # stop a failed poller without crashing the application
            self._report_failure()
            verbose_logger.warning("Kafka logging poller stopped after a client error")
            _ = self.request_close(self.params.flush_timeout)
        finally:
            self._finish_close(producer)

    def _delivered(self, error: DeliveryError | None, message: object) -> None:
        if error is not None:
            self.drop("delivery_failed")
            return
        _ = self._update(delivered=1)

    def flush(self, timeout: float | None = None) -> int | None:
        producer: Final = self._producer
        if producer is None:
            return self.statistics.unconfirmed
        return self._flush(producer, self.budget(timeout))

    def _flush(self, producer: KafkaProducer, timeout: float) -> int | None:
        try:
            remaining: Final = producer.flush(timeout)
        except Exception:  # noqa: BLE001  # a bounded shutdown flush must not prevent other cleanup
            self._report_failure()
            verbose_logger.warning("Kafka logging flush failed")
            return None
        if remaining:
            self._report_failure()
            verbose_logger.warning("Kafka logging flush ended with %d undelivered events", remaining)
        return remaining

    def request_flush(self, timeout: float | None = None) -> Future[int | None]:
        future: Final[Future[int | None]] = Future()
        if self._producer is None or self._stop.is_set():
            future.set_result(self.statistics.unconfirmed if self._closed.done() or not self._stop.is_set() else None)
            return future
        if not self._start_polling or interpreter_exiting():
            future.set_result(self.flush(timeout))
            return future
        with self._flush_lock:
            if self._flush_request is not None:
                return self._flush_request.finished
            if self._closed.done():
                future.set_result(self.statistics.unconfirmed)
                return future
            self._flush_request = FlushRequest(future, time.monotonic() + self.budget(timeout))
        return future

    def _run_flush_request(self, producer: KafkaProducer) -> None:
        with self._flush_lock:
            request: Final = self._flush_request
            self._flush_request = None
        if request is not None and request.finished.set_running_or_notify_cancel():
            request.finished.set_result(self._flush(producer, max(0.0, request.deadline - time.monotonic())))

    def budget(self, timeout: float | None) -> float:
        return self.params.flush_timeout if timeout is None else min(max(timeout, 0.0), self.params.flush_timeout)

    def request_close(self, timeout: float | None = None) -> Future[None]:
        with self._close_lock:
            if self._close_deadline is None:
                self._close_deadline = time.monotonic() + self.budget(timeout)
                self._stop.set()
        producer: Final = self._producer
        if producer is None:
            with self._close_lock:
                if not self._closed.done():
                    self._closed.set_result(None)
        elif not self._start_polling or interpreter_exiting():
            self._finish_close(producer)
        return self._closed

    def _finish_close(self, producer: KafkaProducer) -> None:
        deadline: Final = self._close_deadline
        if not self._lock.acquire(timeout=max(0.0, deadline - time.monotonic()) if deadline is not None else 0.0):
            return
        try:
            self._finalize(producer)
        finally:
            self._lock.release()

    def _finalize(self, producer: KafkaProducer) -> None:
        if not self._finalize_lock.acquire(blocking=False):
            return
        try:
            deadline: Final = self._close_deadline or time.monotonic()
            remaining: Final = self._flush(producer, max(0.0, deadline - time.monotonic()))
            stats: Final = self.statistics
            unconfirmed: Final = (
                remaining if remaining is not None else max(0, stats.accepted - stats.delivered - stats.delivery_failed)
            )
            _ = self._update(unconfirmed=unconfirmed)
        finally:
            self._producer = None
            with self._close_lock:
                if not self._closed.done():
                    self._closed.set_result(None)
            with self._flush_lock:
                request: Final = self._flush_request
                self._flush_request = None
            if request is not None and request.finished.set_running_or_notify_cancel():
                request.finished.set_result(self.statistics.unconfirmed)

    def close(self, timeout: float | None = None) -> None:
        budget: Final = self.budget(timeout)
        closed: Final = self.request_close(budget)
        try:
            closed.result(timeout=budget)
        except FutureTimeoutError:
            self._report_failure()
            verbose_logger.warning(
                "Kafka logging shutdown deadline reached; client cleanup continues in the background"
            )
