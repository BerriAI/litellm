import contextvars
import queue
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


class DeliveryError(Protocol):
    def code(self) -> int: ...


DeliveryCallback: TypeAlias = Callable[[DeliveryError | None, object], None]


class KafkaProducer(Protocol):
    def produce(self, topic: str, *, value: bytes, key: bytes | None, on_delivery: DeliveryCallback) -> None: ...

    def poll(self, timeout: float) -> int: ...

    def flush(self, timeout: float) -> int: ...


ProducerFactory: TypeAlias = Callable[[Mapping[str, KafkaConfigValue]], KafkaProducer]


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


@dataclass(frozen=True, slots=True)
class CallbackJob:
    callback: Callable[[], None]
    context: contextvars.Context


@dataclass(frozen=True, slots=True)
class FlushRequest:
    finished: Future[int | None]
    deadline: float
    target: int


class KafkaSink:
    def __init__(self, params: KafkaInitParams, factory: ProducerFactory, report_failure: Callable[[], None]) -> None:
        self.params: Final = params
        self._factory: Final = factory
        self._report_failure: Final = report_failure
        self._queue: Final[queue.Queue[CallbackJob]] = queue.Queue(maxsize=params.callback_queue_max_messages)
        self._lock: Final = threading.Lock()
        self._stats_lock: Final = threading.Lock()
        self._closed: Final[Future[None]] = Future()
        self._shutdown_flush: Final[Future[int | None]] = Future()
        self._producer: KafkaProducer | None = None
        self._thread: threading.Thread | None = None
        self._close_deadline: float | None = None
        self._flush_request: FlushRequest | None = None
        self._submitted: int = 0
        self._completed: int = 0
        self._stats: KafkaExportStats = KafkaExportStats()

    @property
    def statistics(self) -> KafkaExportStats:
        with self._stats_lock:
            return self._stats

    def _update(
        self, *, accepted: int = 0, delivered: int = 0, dropped: int = 0, delivery_failed: int = 0, unconfirmed: int = 0
    ) -> KafkaExportStats:
        with self._stats_lock:
            updated: Final = replace(
                self._stats,
                accepted=self._stats.accepted + accepted,
                delivered=self._stats.delivered + delivered,
                dropped=self._stats.dropped + dropped,
                delivery_failed=self._stats.delivery_failed + delivery_failed,
                unconfirmed=self._stats.unconfirmed + unconfirmed,
            )
            self._stats = updated
            return updated

    def drop(self, reason: str) -> None:
        stats: Final = self._update(dropped=1, delivery_failed=int(reason == "delivery_failed"))
        self._report_failure()
        if stats.dropped & (stats.dropped - 1) == 0:
            verbose_logger.warning("Kafka logging dropped an event reason=%s total_dropped=%d", reason, stats.dropped)

    def submit(self, callback: Callable[[], None]) -> None:
        job: Final = CallbackJob(callback, contextvars.copy_context())
        if _EXITING.is_set() and self._thread is None and self._close_deadline is None:
            self._execute(job)
            return
        reason: Final = self._enqueue(job)
        if reason is not None:
            self.drop(reason)

    def _enqueue(self, job: CallbackJob) -> str | None:
        with self._lock:
            if self._close_deadline is not None:
                return "closed"
            if self._thread is None:
                thread: Final = threading.Thread(target=self._run, name="litellm-kafka", daemon=True)
                try:
                    thread.start()
                except RuntimeError:
                    return "worker_unavailable"
                self._thread = thread
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                return "callback_queue_full"
            self._submitted += 1
        return None

    def publish(self, payload: Mapping[str, object]) -> None:
        if self._expired():
            self.drop("closed")
            return
        try:
            body: Final = safe_dumps(dict(payload), non_finite_as_null=True).encode("utf-8")
            if len(body) > self.params.message_max_bytes:
                self.drop("message_too_large")
                return
            if self._producer is None:
                self._producer = self._factory(producer_config(self.params))
            if self._expired():
                self.drop("closed")
                return
            identity: Final = payload.get("litellm_call_id") or payload.get("id")
            key: Final = identity.encode("utf-8") if isinstance(identity, str) else None
            self._producer.produce(self.params.topic, value=body, key=key, on_delivery=self._delivered)
            _ = self._update(accepted=1)
        except BufferError:
            self.drop("queue_full")
        except Exception:  # noqa: BLE001  # exporter failures must not change the model response
            self.drop("producer_error")

    def _execute(self, job: CallbackJob) -> None:
        try:
            job.context.run(job.callback)
        except Exception:  # noqa: BLE001  # a failed log job must not stop later events
            self.drop("invalid_payload")

    def _run_next(self) -> None:
        try:
            job: Final = self._queue.get(timeout=0.1)
        except queue.Empty:
            return
        if self._expired():
            self.drop("shutdown_callback")
        else:
            self._execute(job)
        with self._lock:
            self._completed += 1

    def _run(self) -> None:
        try:
            while self._close_deadline is None or (not self._queue.empty() and not self._expired()):
                self._run_flush_request()
                self._run_next()
                if self._producer is not None:
                    _ = self._producer.poll(0)
        except Exception:  # noqa: BLE001  # a failed client must not crash the application
            self._report_failure()
            verbose_logger.warning("Kafka logging worker stopped after a client error")
        finally:
            with self._lock:
                if self._close_deadline is None:
                    self._close_deadline = time.monotonic() + self.params.flush_timeout
            self._finish_close()

    def _expired(self) -> bool:
        deadline: Final = self._close_deadline
        return deadline is not None and time.monotonic() >= deadline

    def _delivered(self, error: DeliveryError | None, message: object) -> None:
        if error is not None:
            self.drop("delivery_failed")
            return
        _ = self._update(delivered=1)

    def _flush(self, timeout: float) -> int | None:
        if self._producer is None:
            return self.statistics.unconfirmed
        try:
            remaining: Final = self._producer.flush(timeout)
        except Exception:  # noqa: BLE001  # a bounded flush must not prevent other cleanup
            self._report_failure()
            verbose_logger.warning("Kafka logging flush failed")
            return None
        if remaining:
            self._report_failure()
            verbose_logger.warning("Kafka logging flush ended with %d undelivered events", remaining)
        return remaining

    def request_flush(self, timeout: float | None = None) -> Future[int | None]:
        with self._lock:
            if self._close_deadline is not None:
                return self._shutdown_flush
            if self._thread is None:
                finished: Final[Future[int | None]] = Future()
                finished.set_result(self._flush(self.budget(timeout)))
                return finished
            deadline: Final = time.monotonic() + self.budget(timeout)
            if self._flush_request is not None:
                self._flush_request = replace(
                    self._flush_request, deadline=min(deadline, self._flush_request.deadline), target=self._submitted
                )
            else:
                self._flush_request = FlushRequest(Future(), deadline, self._submitted)
            return self._flush_request.finished

    def _run_flush_request(self) -> None:
        with self._lock:
            request: Final = self._flush_request
            if request is None or self._completed < request.target:
                return
            self._flush_request = None
        request.finished.set_result(self._flush(max(0.0, request.deadline - time.monotonic())))

    def budget(self, timeout: float | None) -> float:
        return self.params.flush_timeout if timeout is None else min(max(timeout, 0.0), self.params.flush_timeout)

    def flush(self, timeout: float | None = None) -> int | None:
        budget: Final = self.budget(timeout)
        try:
            return self.request_flush(budget).result(timeout=budget)
        except FutureTimeoutError:
            return None

    def request_close(self, timeout: float | None = None) -> Future[None]:
        with self._lock:
            if self._close_deadline is not None:
                return self._closed
            self._close_deadline = time.monotonic() + self.budget(timeout)
            inline: Final = self._thread is None
        if inline:
            self._finish_close()
        return self._closed

    def _finish_close(self) -> None:
        while True:
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                break
            self.drop("shutdown_callback")
        remaining: Final = self._flush(max(0.0, (self._close_deadline or time.monotonic()) - time.monotonic()))
        stats: Final = self.statistics
        unconfirmed: Final = (
            remaining if remaining is not None else max(0, stats.accepted - stats.delivered - stats.delivery_failed)
        )
        _ = self._update(unconfirmed=unconfirmed)
        self._producer = None
        with self._lock:
            request: Final = self._flush_request
            self._flush_request = None
        if request is not None:
            request.finished.set_result(remaining)
        self._shutdown_flush.set_result(remaining)
        self._closed.set_result(None)

    def close(self, timeout: float | None = None) -> None:
        budget: Final = self.budget(timeout)
        try:
            self.request_close(budget).result(timeout=budget)
        except FutureTimeoutError:
            self._report_failure()
            verbose_logger.warning("Kafka logging shutdown deadline reached; cleanup continues in the background")
