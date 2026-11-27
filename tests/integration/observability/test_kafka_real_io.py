from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final
from uuid import uuid4

import pytest
from confluent_kafka import OFFSET_BEGINNING, Consumer, Producer, TopicPartition
from confluent_kafka.admin import ClusterMetadata
from pydantic import ConfigDict, TypeAdapter

from litellm.integrations.kafka import KafkaLogger
from litellm.types.integrations.kafka import KafkaInitParams
from tests.test_litellm_rust.support.child_interpreter import run_child_interpreter

METADATA: Final = TypeAdapter(ClusterMetadata, config=ConfigDict(arbitrary_types_allowed=True))
PAYLOAD: Final = TypeAdapter(dict[str, object])
NOW: Final = datetime(2025, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class Broker:
    owner: Producer
    servers: str
    topic: str


@pytest.fixture
def broker() -> Generator[Broker]:
    owner: Final = Producer({"test.mock.num.brokers": 1, "log_level": 0})
    metadata: Final = METADATA.validate_python(owner.list_topics(timeout=10))
    servers: Final = ",".join(f"{node.host}:{node.port}" for node in metadata.brokers.values())
    topic: Final = f"litellm-{uuid4().hex}"
    _ = METADATA.validate_python(owner.list_topics(topic, timeout=10))
    try:
        yield Broker(owner, servers, topic)
    finally:
        _ = owner.close()


@pytest.mark.parametrize("status", ["success", "failure"])
def test_native_client_delivers_standard_json_and_correlation_over_kafka(broker: Broker, status: str) -> None:
    logger: Final = KafkaLogger(
        KafkaInitParams(bootstrap_servers=broker.servers, topic=broker.topic, security_protocol="PLAINTEXT")
    )
    consumer: Final = Consumer(
        {"bootstrap.servers": broker.servers, "group.id": uuid4().hex, "enable.auto.commit": False, "log_level": 0}
    )
    payload: Final = {
        "litellm_call_id": "request-correlation",
        "id": "response-id",
        "status": status,
        "metadata": {"user_api_key_alias": "analytics-key"},
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
    }
    try:
        metadata: Final = METADATA.validate_python(broker.owner.list_topics(broker.topic, timeout=10))
        consumer.assign(
            [
                TopicPartition(broker.topic, partition, OFFSET_BEGINNING)
                for partition in metadata.topics[broker.topic].partitions
            ]
        )
        log: Final = logger.log_failure_event if status == "failure" else logger.log_success_event
        log({"standard_logging_object": payload}, None, NOW, NOW)
        assert logger.sink.flush(5) == 0
        messages: Final = consumer.consume(1, timeout=10)

        assert len(messages) == 1
        message: Final = messages[0]
        assert message.error() is None
        assert message.key() == b"request-correlation"
        assert message.topic() == broker.topic
        value: Final = message.value()
        assert value is not None
        assert PAYLOAD.validate_json(value) == payload
        assert logger.statistics.delivered == 1
        assert logger.statistics.dropped == 0
    finally:
        consumer.close()
        logger.close()


def test_native_delivery_timeout_reports_a_drop_without_raising(broker: Broker) -> None:
    _ = broker.owner.close()
    logger: Final = KafkaLogger(
        KafkaInitParams(
            security_protocol="PLAINTEXT",
            bootstrap_servers=broker.servers,
            topic=broker.topic,
            request_timeout_ms=500,
            delivery_timeout_ms=1000,
            flush_timeout=5,
        )
    )
    try:
        logger.log_success_event({"standard_logging_object": {"id": "unavailable-broker"}}, None, NOW, NOW)
        assert logger.sink.flush() == 0
        assert logger.statistics.accepted == 1
        assert logger.statistics.delivered == 0
        assert logger.statistics.dropped == 1
    finally:
        logger.close()


def test_native_queue_limit_drops_extra_events_during_a_broker_outage(broker: Broker) -> None:
    _ = broker.owner.close()
    logger: Final = KafkaLogger(
        KafkaInitParams(
            security_protocol="PLAINTEXT",
            bootstrap_servers=broker.servers,
            topic=broker.topic,
            queue_max_messages=1,
            delivery_timeout_ms=300_000,
        )
    )
    try:
        logger.log_success_event({"standard_logging_object": {"id": "queued"}}, None, NOW, NOW)
        logger.log_success_event({"standard_logging_object": {"id": "over-capacity"}}, None, NOW, NOW)
        assert logger.sink.request_flush(0).result(5) == 1
        assert logger.statistics.accepted == 1
        assert logger.statistics.delivered == 0
        assert logger.statistics.dropped == 1
    finally:
        logger.close(0)


@pytest.mark.parametrize("prime_producer", [False, True])
def test_interpreter_exit_drains_pending_sdk_callbacks_before_closing_kafka(
    broker: Broker, prime_producer: bool
) -> None:
    metadata: Final = METADATA.validate_python(broker.owner.list_topics(broker.topic, timeout=10))
    consumer: Final = Consumer(
        {"bootstrap.servers": broker.servers, "group.id": uuid4().hex, "enable.auto.commit": False, "log_level": 0}
    )
    source: Final = f"""
import asyncio
from datetime import datetime, timezone
from litellm.integrations.kafka import KafkaLogger
from litellm.types.integrations.kafka import KafkaInitParams
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
logger = KafkaLogger(KafkaInitParams(bootstrap_servers={broker.servers!r}, topic={broker.topic!r}, security_protocol="PLAINTEXT"))
now = datetime(2025, 1, 1, tzinfo=timezone.utc)
if {prime_producer!r}:
    logger.log_success_event({{"standard_logging_object": {{"id": "already-enqueued"}}}}, None, now, now)
async def enqueue():
    GLOBAL_LOGGING_WORKER.start()
    GLOBAL_LOGGING_WORKER.enqueue(logger.async_log_success_event(
        {{"standard_logging_object": {{"id": "pending-at-exit"}}}}, None, now, now))
loop = asyncio.new_event_loop()
loop.run_until_complete(enqueue())
loop.close()
"""
    try:
        consumer.assign(
            [
                TopicPartition(broker.topic, partition, OFFSET_BEGINNING)
                for partition in metadata.topics[broker.topic].partitions
            ]
        )
        result: Final = run_child_interpreter(source, timeout=20)
        assert result.returncode == 0, result.stdout + result.stderr
        messages: Final = consumer.consume(2 if prime_producer else 1, timeout=10)
        assert all(message.error() is None for message in messages)
        events: Final = tuple(PAYLOAD.validate_json(message.value() or b"null") for message in messages)
        assert frozenset(event["id"] for event in events) == (
            frozenset(("already-enqueued", "pending-at-exit")) if prime_producer else frozenset(("pending-at-exit",))
        ), result.stdout + result.stderr
    finally:
        consumer.close()
