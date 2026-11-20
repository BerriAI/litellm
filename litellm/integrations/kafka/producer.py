import atexit
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Final, Protocol, TypeAlias

from litellm._logging import verbose_logger
from litellm.integrations.kafka.config import KafkaConfigValue, producer_config
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.types.integrations.kafka import KafkaInitParams


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
        self._stop: Final = threading.Event()
        self._producer: KafkaProducer | None = None
        self._poll_thread: threading.Thread | None = None
        self._closed: bool = False
        self._stats: KafkaExportStats = KafkaExportStats()

    @property
    def statistics(self) -> KafkaExportStats:
        with self._stats_lock:
            return self._stats

    def _update(self, *, accepted: int = 0, delivered: int = 0, dropped: int = 0) -> KafkaExportStats:
        with self._stats_lock:
            updated: Final = replace(
                self._stats,
                accepted=self._stats.accepted + accepted,
                delivered=self._stats.delivered + delivered,
                dropped=self._stats.dropped + dropped,
            )
            self._stats = updated
            return updated

    def drop(self, reason: str) -> None:
        stats: Final = self._update(dropped=1)
        self._report_failure()
        if stats.dropped & (stats.dropped - 1) == 0:
            verbose_logger.warning("Kafka logging dropped an event reason=%s total_dropped=%d", reason, stats.dropped)

    def publish(self, payload: Mapping[str, object]) -> None:
        try:
            body: Final = safe_dumps(dict(payload), non_finite_as_null=True).encode("utf-8")
            if len(body) > self.params.message_max_bytes:
                self.drop("message_too_large")
                return
            identity: Final = payload.get("litellm_call_id") or payload.get("id")
            key: Final = identity.encode("utf-8") if isinstance(identity, str) else None
            with self._lock:
                if self._closed or self._stop.is_set():
                    self.drop("closed")
                    return
                producer: Final = self._producer if self._producer is not None else self._create_producer()
                producer.produce(self.params.topic, value=body, key=key, on_delivery=self._delivered)
                _ = self._update(accepted=1)
        except BufferError:
            self.drop("queue_full")
        except Exception:  # noqa: BLE001  # exporter failures must not change the model response
            self.drop("producer_error")

    def _create_producer(self) -> KafkaProducer:
        producer: Final = self._factory(producer_config(self.params))
        self._producer = producer
        if self._start_polling:
            thread: Final = threading.Thread(
                target=self._poll, args=(producer,), name="litellm-kafka-poll", daemon=True
            )
            self._poll_thread = thread
            thread.start()
        atexit.register(self.close)
        return producer

    def _poll(self, producer: KafkaProducer) -> None:
        try:
            while not self._stop.is_set():
                _ = producer.poll(0.1)
        except Exception:  # noqa: BLE001  # stop a failed poller without crashing the application
            self._stop.set()
            self._report_failure()
            verbose_logger.warning("Kafka logging poller stopped after a client error")

    def _delivered(self, error: DeliveryError | None, message: object) -> None:
        if error is not None:
            self.drop("delivery_failed")
            return
        _ = self._update(delivered=1)

    def flush(self, timeout: float | None = None) -> int | None:
        with self._lock:
            producer: Final = self._producer
        if producer is None:
            return 0
        try:
            remaining: Final = producer.flush(
                self.params.flush_timeout if timeout is None else min(max(timeout, 0.0), self.params.flush_timeout)
            )
        except Exception:  # noqa: BLE001  # a bounded shutdown flush must not prevent other cleanup
            self._report_failure()
            verbose_logger.warning("Kafka logging flush failed")
            return None
        if remaining:
            self._report_failure()
            verbose_logger.warning("Kafka logging flush ended with %d undelivered events", remaining)
        return remaining

    def close(self, timeout: float | None = None) -> None:
        budget: Final = (
            self.params.flush_timeout if timeout is None else min(max(timeout, 0.0), self.params.flush_timeout)
        )
        deadline: Final = time.monotonic() + budget
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        thread: Final = self._poll_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=min(1.0, budget))
        _ = self.flush(max(0.0, deadline - time.monotonic()))
        with self._lock:
            self._producer = None
        atexit.unregister(self.close)
