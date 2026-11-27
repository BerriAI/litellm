import asyncio
import json
import sys
import threading
from collections.abc import AsyncGenerator, Generator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final

import pytest
from pydantic import BaseModel, ConfigDict

import litellm
from litellm.integrations.kafka.config import KafkaConfigValue
from litellm.integrations.kafka.logger import KafkaLogger
from litellm.integrations.kafka.producer import DeliveryCallback, DeliveryError
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.responses.streaming_iterator import drain_logging_tasks
from litellm.types.integrations.kafka import KafkaInitParams
from litellm.types.utils import ModelResponse

NOW: Final = datetime(2025, 1, 1, tzinfo=timezone.utc)


class Payload(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    metadata: Mapping[str, object]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class Message:
    topic: str
    value: bytes
    key: bytes | None
    callback: DeliveryCallback


@dataclass(frozen=True, slots=True)
class Rejected:
    def code(self) -> int:
        return 1


class RecordingProducer:
    def __init__(
        self, *, full: bool = False, rejected: bool = False, fail_flush: bool = False, pending_flush: bool = False
    ) -> None:
        self.full: Final = full
        self.rejected: Final = rejected
        self.fail_flush: Final = fail_flush
        self.pending_flush: Final = pending_flush
        self.messages: tuple[Message, ...] = ()
        self.waiting: tuple[Message, ...] = ()
        self.flush_timeouts: tuple[float, ...] = ()
        self.producing_threads: tuple[int, ...] = ()

    def __len__(self) -> int:
        return len(self.waiting)

    def produce(self, topic: str, *, value: bytes, key: bytes | None, on_delivery: DeliveryCallback) -> None:
        if self.full:
            raise BufferError("queue full")
        message: Final = Message(topic, value, key, on_delivery)
        self.messages = (*self.messages, message)
        self.waiting = (*self.waiting, message)
        self.producing_threads = (*self.producing_threads, threading.get_ident())

    def poll(self, timeout: float) -> int:
        return 0

    def flush(self, timeout: float) -> int:
        self.flush_timeouts = (*self.flush_timeouts, timeout)
        if self.fail_flush:
            raise RuntimeError("a client error containing credentials must not be logged")
        if self.pending_flush:
            return len(self.waiting)
        waiting: Final = self.waiting
        self.waiting = ()
        error: Final[DeliveryError | None] = Rejected() if self.rejected else None
        for message in waiting:
            message.callback(error, message)
        return 0


class RecordingFactory:
    def __init__(self, producer: RecordingProducer) -> None:
        self.producer: Final = producer
        self.configurations: tuple[Mapping[str, KafkaConfigValue], ...] = ()

    def __call__(self, configuration: Mapping[str, KafkaConfigValue]) -> RecordingProducer:
        self.configurations = (*self.configurations, configuration)
        return self.producer


@contextmanager
def configured(
    producer: RecordingProducer, settings: Mapping[str, object] | None = None
) -> Generator[tuple[KafkaLogger, RecordingFactory]]:
    factory: Final = RecordingFactory(producer)
    logger: Final = KafkaLogger(
        KafkaInitParams.model_validate({"bootstrap_servers": "broker:9092", "topic": "events", **(settings or {})}),
        producer_factory=factory,
    )
    try:
        yield logger, factory
    finally:
        logger.close()


def event(status: str = "success", fields: Mapping[str, object] | None = None) -> Mapping[str, object]:
    return {
        "standard_logging_object": {
            "id": "response-id",
            "litellm_call_id": "gateway-id",
            "status": status,
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "total_tokens": 15,
            "metadata": {"user_api_key_alias": "application-key", "user_api_key_hash": "key-hash"},
            "messages": [{"role": "user", "content": "private-prompt"}],
            "response": {
                "choices": [
                    {
                        "message": {"content": "private-answer"},
                        "logprobs": {
                            "content": [{"token": "private-answer", "bytes": [112, 114, 105, 118, 97, 116, 101]}]
                        },
                    }
                ]
            },
            "error_str": "invalid private-prompt",
            "model_parameters": {
                "query": "private-prompt",
                "documents": ["private-answer"],
                "prediction": {"content": "private-answer"},
            },
            "error_information": {
                "error_message": "private-answer",
                "traceback": "private-prompt",
                "error_code": "429",
            },
            **(fields or {}),
        }
    }


def test_configuration_does_not_connect_and_requests_reuse_one_producer() -> None:
    producer: Final = RecordingProducer()
    with configured(producer) as (logger, factory):
        assert factory.configurations == ()
        logger.log_success_event(event(), None, NOW, NOW)
        _ = logger.sink.flush()
        logger.log_failure_event(event("failure"), None, NOW, NOW)
        _ = logger.sink.flush()

        assert len(factory.configurations) == 1
        assert tuple(json.loads(message.value)["status"] for message in producer.messages) == ("success", "failure")


def test_the_standard_event_keeps_identity_usage_and_request_correlation() -> None:
    producer: Final = RecordingProducer()
    original: Final = event()
    with configured(producer) as (logger, _):
        logger.log_success_event(original, None, NOW, NOW)
        assert logger.sink.flush() == 0
        message: Final = producer.messages[0]

        assert message.topic == logger.sink.params.topic
        assert message.key == b"gateway-id"
        assert json.loads(message.value) == original["standard_logging_object"]
        assert logger.statistics.accepted == 1
        assert logger.statistics.delivered == 1


@pytest.mark.parametrize("status", ["success", "failure"])
@pytest.mark.asyncio
async def test_async_export_runs_produce_off_the_event_loop(status: str) -> None:
    producer: Final = RecordingProducer()
    event_loop_thread: Final = threading.get_ident()
    with configured(producer) as (logger, _):
        if status == "success":
            await logger.async_log_success_event(event(status), None, NOW, NOW)
        else:
            await logger.async_log_failure_event(event(status), None, NOW, NOW)

        await logger.flush_queue()
        assert producer.producing_threads
        assert all(thread != event_loop_thread for thread in producer.producing_threads)
        assert logger.statistics.delivered == 1


def test_per_destination_redaction_keeps_attribution_and_does_not_change_other_destinations() -> None:
    producer: Final = RecordingProducer()
    original: Final = event()
    snapshot: Final = json.dumps(original)
    with configured(producer, {"turn_off_message_logging": True}) as (logger, _):
        logger.log_success_event(original, None, NOW, NOW)
        assert logger.sink.flush() == 0
        body: Final = producer.messages[0].value
        payload: Final = Payload.model_validate_json(body)

        assert b"private-prompt" not in body
        assert b"private-answer" not in body
        assert payload.metadata["user_api_key_alias"] == "application-key"
        assert payload.total_tokens == 15
        assert json.dumps(original) == snapshot


def test_global_redaction_applies_to_a_failure_event(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "turn_off_message_logging", True)
    producer: Final = RecordingProducer()
    with configured(producer) as (logger, _):
        logger.log_failure_event(event("failure"), None, NOW, NOW)
        _ = logger.sink.flush()

        assert b"private-prompt" not in producer.messages[0].value
        assert b"private-answer" not in producer.messages[0].value


def test_excluded_fields_are_not_exported_or_removed_from_the_shared_event(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "standard_logging_payload_excluded_fields", ["messages"])
    producer: Final = RecordingProducer()
    original: Final = event()
    with configured(producer) as (logger, _):
        logger.log_success_event(original, None, NOW, NOW)
        assert logger.sink.flush() == 0

        assert "messages" not in json.loads(producer.messages[0].value)
        shared: Final = original["standard_logging_object"]
        assert isinstance(shared, dict)
        assert "messages" in shared


def test_a_full_buffer_drops_the_log_without_failing_the_request() -> None:
    producer: Final = RecordingProducer(full=True)
    with configured(producer) as (logger, _):
        logger.log_success_event(event(), None, NOW, NOW)
        _ = logger.sink.flush()

        assert producer.messages == ()
        assert logger.statistics.accepted == 0
        assert logger.statistics.dropped == 1


def test_a_delivery_error_is_observable_and_is_not_counted_as_delivered() -> None:
    producer: Final = RecordingProducer(rejected=True)
    with configured(producer) as (logger, _):
        logger.log_success_event(event(), None, NOW, NOW)
        _ = logger.sink.flush()

        assert logger.statistics.accepted == 1
        assert logger.statistics.delivered == 0
        assert logger.statistics.dropped == 1


def test_oversized_events_are_rejected_before_allocating_a_producer() -> None:
    producer: Final = RecordingProducer()
    with configured(producer, {"message_max_bytes": 1000}) as (logger, factory):
        logger.log_success_event(event(fields={"messages": "large-prompt" * 1000}), None, NOW, NOW)
        assert logger.sink.flush() == 0

        assert factory.configurations == ()
        assert logger.statistics.dropped == 1


def test_intermediate_events_without_a_standard_payload_are_ignored() -> None:
    producer: Final = RecordingProducer()
    with configured(producer) as (logger, factory):
        logger.log_success_event({"stream": True}, None, NOW, NOW)

        assert factory.configurations == ()
        assert logger.statistics.dropped == 0


def test_nonfinite_usage_values_produce_valid_json() -> None:
    producer: Final = RecordingProducer()
    with configured(producer) as (logger, _):
        logger.log_success_event(event(fields={"response_cost": float("nan")}), None, NOW, NOW)
        assert logger.sink.flush() == 0

        assert json.loads(producer.messages[0].value)["response_cost"] is None
        assert b"NaN" not in producer.messages[0].value


def test_shutdown_flush_is_bounded_idempotent_and_rejects_new_events() -> None:
    producer: Final = RecordingProducer()
    with configured(producer, {"flush_timeout": 2.0}) as (logger, _):
        logger.log_success_event(event(), None, NOW, NOW)
        _ = logger.sink.flush()
        logger.sink.request_close(0).result(5)
        logger.close()
        logger.log_success_event(event(), None, NOW, NOW)
        _ = logger.sink.flush()

        assert producer.flush_timeouts[-1] == 0.0
        assert logger.statistics.delivered == 1
        assert logger.statistics.dropped == 1
        assert len(producer.messages) == 1


def test_a_flush_failure_does_not_expose_client_exception_text(caplog: pytest.LogCaptureFixture) -> None:
    producer: Final = RecordingProducer(fail_flush=True)
    with configured(producer) as (logger, _):
        logger.log_success_event(event(), None, NOW, NOW)

        assert logger.sink.flush() is None
        assert "a client error containing credentials" not in caplog.text


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_sdk_success_and_streams_export_one_final_event(monkeypatch: pytest.MonkeyPatch, stream: bool) -> None:
    producer: Final = RecordingProducer()
    with configured(producer) as (logger, _):
        monkeypatch.setattr(litellm, "callbacks", [logger])
        response: Final = await litellm.acompletion(
            model="openai/gpt-6-luna",
            api_key="unit-test-key",
            messages=[{"role": "user", "content": "sdk-event"}],
            mock_response="sdk-reply",
            metadata={"user_api_key_alias": "sdk-key"},
            stream=stream,
        )
        if stream:
            assert isinstance(response, CustomStreamWrapper)
            async for _ in response:
                pass
        await drain_logging_tasks()
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=5)
        await logger.flush_queue()

        assert len(producer.messages) == 1
        payload: Final = Payload.model_validate_json(producer.messages[0].value)
        assert payload.status == "success"
        assert payload.metadata["user_api_key_alias"] == "sdk-key"
        assert payload.total_tokens == payload.prompt_tokens + payload.completion_tokens
        assert "sdk-reply" in producer.messages[0].value.decode()


@pytest.mark.asyncio
async def test_sdk_failures_export_a_terminal_failure_event(monkeypatch: pytest.MonkeyPatch) -> None:
    producer: Final = RecordingProducer()
    with configured(producer) as (logger, _):
        monkeypatch.setattr(litellm, "callbacks", [logger])
        with pytest.raises(litellm.RateLimitError):
            await litellm.acompletion(
                model="openai/gpt-6-luna",
                api_key="unit-test-key",
                messages=[{"role": "user", "content": "sdk-failure"}],
                mock_response="litellm.RateLimitError",
                num_retries=0,
            )
        await drain_logging_tasks()
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=5)
        await logger.flush_queue()

        assert len(producer.messages) == 1
        assert json.loads(producer.messages[0].value)["status"] == "failure"


@pytest.mark.asyncio
async def test_queue_saturation_does_not_change_a_successful_model_response(monkeypatch: pytest.MonkeyPatch) -> None:
    producer: Final = RecordingProducer(full=True)
    with configured(producer) as (logger, _):
        monkeypatch.setattr(litellm, "callbacks", [logger])
        response: Final = await litellm.acompletion(
            model="openai/gpt-6-luna",
            api_key="unit-test-key",
            messages=[{"role": "user", "content": "queue-full"}],
            mock_response="still-served",
        )
        await drain_logging_tasks()
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=5)
        await logger.flush_queue()

        assert isinstance(response, ModelResponse)
        assert response.choices[0].message.content == "still-served"
        assert logger.statistics.dropped == 1


@pytest.mark.asyncio
async def test_proxy_shutdown_flushes_each_active_destination_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.shutdown.kafka import close_kafka_loggers

    producer: Final = RecordingProducer()
    with configured(producer) as (logger, _):
        logger.log_success_event(event(), None, NOW, NOW)
        _ = logger.sink.flush()
        monkeypatch.setattr(litellm, "callbacks", [logger])
        monkeypatch.setattr(litellm, "success_callback", [logger])

        await close_kafka_loggers(1.0)

        assert len(producer.flush_timeouts) == 2
        assert logger.statistics.delivered == 1


def test_the_native_callback_name_constructs_a_cached_destination(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.litellm_core_utils import litellm_logging
    from litellm.litellm_core_utils.custom_logger_registry import CustomLoggerRegistry

    monkeypatch.setattr(litellm_logging, "_in_memory_loggers", [])
    monkeypatch.setattr(litellm, "kafka_params", {"bootstrap_servers": "broker:9092", "topic": "events"})
    first: Final = litellm_logging.init_custom_logger_compatible_class_at_startup("kafka", None, None)
    second: Final = litellm_logging.init_custom_logger_compatible_class_at_startup("kafka", None, None)

    assert first is second
    assert isinstance(first, KafkaLogger)
    assert CustomLoggerRegistry.get_callback_str_from_class_type(type(first)) == "kafka"


def test_invalid_native_callback_configuration_fails_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic import ValidationError

    from litellm.litellm_core_utils import litellm_logging

    monkeypatch.setattr(litellm_logging, "_in_memory_loggers", [])
    monkeypatch.setattr(litellm, "kafka_params", {"topic": "events"})
    with pytest.raises(ValidationError, match="bootstrap_servers"):
        litellm_logging.init_custom_logger_compatible_class_at_startup("kafka", None, None)


def test_enabling_kafka_without_the_optional_client_explains_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "confluent_kafka", None)
    with pytest.raises(ImportError, match=r"Install litellm\[kafka\]"):
        KafkaLogger(KafkaInitParams(bootstrap_servers="broker:9092", topic="events"))


@asynccontextmanager
async def saturated_default_executor() -> AsyncGenerator[None]:
    loop: Final = asyncio.get_running_loop()
    entered: Final = asyncio.Event()
    release: Final = threading.Event()

    def occupy() -> None:
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(10), "executor test did not release its blocked dependency"

    with ThreadPoolExecutor(max_workers=1) as executor:
        loop.set_default_executor(executor)
        occupied: Final = loop.run_in_executor(None, occupy)
        await asyncio.wait_for(entered.wait(), 5)
        try:
            yield
        finally:
            release.set()
            await occupied


def test_async_logging_does_not_wait_for_the_applications_default_executor() -> None:
    async def run() -> None:
        producer: Final = RecordingProducer()
        with configured(producer) as (logger, _):
            async with saturated_default_executor():
                await asyncio.wait_for(logger.async_log_success_event(event(), None, NOW, NOW), 5)
                await logger.flush_queue()
                assert len(producer.messages) == 1

    asyncio.run(run())


def test_proxy_kafka_shutdown_does_not_wait_for_the_applications_default_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy.shutdown.kafka import close_kafka_loggers

    async def run() -> None:
        producer: Final = RecordingProducer()
        with configured(producer) as (logger, _):
            logger.log_success_event(event(), None, NOW, NOW)
            monkeypatch.setattr(litellm, "callbacks", [logger])
            async with saturated_default_executor():
                await asyncio.wait_for(close_kafka_loggers(1), 5)
                assert logger.statistics.delivered == 1

    asyncio.run(run())


def test_shutdown_does_not_wait_for_or_publish_after_an_inflight_producer_constructor() -> None:
    producer: Final = RecordingProducer()
    entered: Final = threading.Event()
    release: Final = threading.Event()
    closed: Final = threading.Event()

    def factory(configuration: Mapping[str, KafkaConfigValue]) -> RecordingProducer:
        entered.set()
        assert release.wait(10), "constructor test did not release its blocked dependency"
        return producer

    logger: Final = KafkaLogger(KafkaInitParams(bootstrap_servers="broker:9092", topic="events"), factory)

    def close() -> None:
        logger.close(0)
        closed.set()

    publisher: Final = threading.Thread(target=logger.log_success_event, args=(event(), None, NOW, NOW), daemon=True)
    closer: Final = threading.Thread(target=close, daemon=True)
    publisher.start()
    try:
        assert entered.wait(5)
        closer.start()
        assert closed.wait(5), "close waited on the blocked producer constructor"
    finally:
        release.set()
        publisher.join(5)
        closer.join(5)
        logger.sink.request_close(0).result(5)

    assert producer.messages == ()
    assert logger.statistics.accepted == 0
    assert logger.statistics.dropped == 1


def test_close_does_not_report_an_empty_queue_after_an_unconfirmed_flush() -> None:
    producer: Final = RecordingProducer(pending_flush=True)
    with configured(producer) as (logger, _):
        logger.log_success_event(event(), None, NOW, NOW)
        _ = logger.sink.flush()
        logger.sink.request_close(0).result(5)

        assert logger.sink.flush() == 1
        assert logger.statistics.unconfirmed == 1
        assert logger.statistics.delivered == 0
        assert logger.statistics.dropped == 0


def test_a_stalled_kafka_constructor_does_not_hold_the_next_accounting_callback() -> None:
    producer: Final = RecordingProducer()
    entered: Final = threading.Event()
    release: Final = threading.Event()
    accounted: Final = threading.Event()

    def factory(configuration: Mapping[str, KafkaConfigValue]) -> RecordingProducer:
        entered.set()
        assert release.wait(10), "test did not release its blocked Kafka client"
        return producer

    logger: Final = KafkaLogger(KafkaInitParams(bootstrap_servers="broker:9092", topic="events"), factory)
    publisher: Final = threading.Thread(target=logger.log_success_event, args=(event(), None, NOW, NOW), daemon=True)

    async def callback_chain() -> None:
        await logger.async_log_success_event(event(), None, NOW, NOW)
        accounted.set()

    async def before_next_loop_turn() -> None:
        loop: Final = asyncio.get_running_loop()
        checkpoint: Final[asyncio.Future[None]] = loop.create_future()
        callback: Final = asyncio.create_task(callback_chain())
        loop.call_soon(checkpoint.set_result, None)
        try:
            await checkpoint
            assert callback.done(), "Kafka preparation held the accounting callback across an event-loop turn"
            await callback
        finally:
            callback.cancel()
            await asyncio.gather(callback, return_exceptions=True)

    publisher.start()
    try:
        assert entered.wait(5)
        asyncio.run(asyncio.wait_for(before_next_loop_turn(), 5))
        assert accounted.is_set()
        assert not release.is_set()
    finally:
        release.set()
        publisher.join(5)
        logger.close()


def test_shutdown_accounts_for_a_produce_call_that_finishes_after_the_close_deadline() -> None:
    entered: Final = threading.Event()
    release: Final = threading.Event()

    class GatedProducer(RecordingProducer):
        def produce(self, topic: str, *, value: bytes, key: bytes | None, on_delivery: DeliveryCallback) -> None:
            entered.set()
            assert release.wait(5), "test did not release its blocked produce call"
            super().produce(topic, value=value, key=key, on_delivery=on_delivery)

    producer: Final = GatedProducer(pending_flush=True)
    logger: Final = KafkaLogger(
        KafkaInitParams(bootstrap_servers="broker:9092", topic="events"), RecordingFactory(producer)
    )
    publisher: Final = threading.Thread(target=logger.log_success_event, args=(event(), None, NOW, NOW), daemon=True)
    publisher.start()
    try:
        assert entered.wait(5)
        logger.close(0)
    finally:
        release.set()
        publisher.join(5)
        logger.sink.request_close(0).result(5)

    assert logger.statistics.accepted == 1
    assert logger.statistics.delivered == 0
    assert logger.statistics.unconfirmed == 1
    assert logger.sink.flush() == 1
