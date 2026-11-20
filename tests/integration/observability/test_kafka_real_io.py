import selectors
import socket
import socketserver
import threading
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
from tests.integration._support.ports import allocate_port
from tests.test_litellm_rust.support.child_interpreter import run_child_interpreter

METADATA: Final = TypeAdapter(ClusterMetadata, config=ConfigDict(arbitrary_types_allowed=True))
PAYLOAD: Final = TypeAdapter(dict[str, object])
NOW: Final = datetime(2025, 1, 1, tzinfo=timezone.utc)
SOCKET: Final = TypeAdapter(socket.socket, config=ConfigDict(arbitrary_types_allowed=True))


class RetryingBootstrap(socketserver.ThreadingTCPServer):
    daemon_threads = True

    def __init__(self, target: tuple[str, int]) -> None:
        self.target: Final = target
        self.attempts: int = 0
        self._attempt_lock: Final = threading.Lock()
        super().__init__(("127.0.0.1", allocate_port()), BootstrapHandler)

    def reject_first(self) -> bool:
        with self._attempt_lock:
            self.attempts += 1
            return self.attempts == 1


class BootstrapHandler(socketserver.BaseRequestHandler):
    request: socket.socket

    def handle(self) -> None:
        assert isinstance(self.server, RetryingBootstrap)
        if self.server.reject_first():
            return
        client: Final = SOCKET.validate_python(self.request)
        with (
            socket.create_connection(self.server.target, timeout=5) as upstream,
            selectors.DefaultSelector() as selector,
        ):
            _ = selector.register(client, selectors.EVENT_READ)
            _ = selector.register(upstream, selectors.EVENT_READ)
            while True:
                for key, _ in selector.select(5):
                    if not self._forward(SOCKET.validate_python(key.fileobj), client, upstream):
                        return

    @staticmethod
    def _forward(source: socket.socket, client: socket.socket, upstream: socket.socket) -> bool:
        body: Final = source.recv(65_536)
        if not body:
            return False
        destination: Final = upstream if source is client else client
        destination.sendall(body)
        return True


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
    logger: Final = KafkaLogger(KafkaInitParams(bootstrap_servers=broker.servers, topic=broker.topic))
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
            bootstrap_servers=broker.servers,
            topic=broker.topic,
            queue_max_messages=1,
            delivery_timeout_ms=300_000,
        )
    )
    try:
        logger.log_success_event({"standard_logging_object": {"id": "queued"}}, None, NOW, NOW)
        logger.log_success_event({"standard_logging_object": {"id": "over-capacity"}}, None, NOW, NOW)
        assert logger.statistics.accepted == 1
        assert logger.statistics.delivered == 0
        assert logger.statistics.dropped == 1
    finally:
        logger.close(0)


def test_native_retry_recovers_after_a_dropped_bootstrap_connection_without_resubmitting(broker: Broker) -> None:
    metadata: Final = METADATA.validate_python(broker.owner.list_topics(broker.topic, timeout=10))
    node: Final = next(iter(metadata.brokers.values()))
    assert node.host is not None
    with RetryingBootstrap((node.host, node.port)) as relay:
        server: Final = threading.Thread(target=relay.serve_forever, daemon=True)
        server.start()
        logger: Final = KafkaLogger(
            KafkaInitParams(
                bootstrap_servers=f"127.0.0.1:{relay.server_address[1]}",
                topic=broker.topic,
                request_timeout_ms=1000,
                delivery_timeout_ms=10_000,
                flush_timeout=10,
            )
        )
        consumer: Final = Consumer(
            {"bootstrap.servers": broker.servers, "group.id": uuid4().hex, "enable.auto.commit": False, "log_level": 0}
        )
        try:
            consumer.assign(
                [
                    TopicPartition(broker.topic, partition, OFFSET_BEGINNING)
                    for partition in metadata.topics[broker.topic].partitions
                ]
            )
            logger.log_success_event({"standard_logging_object": {"id": "retried-once-by-the-client"}}, None, NOW, NOW)
            assert logger.sink.flush() == 0
            messages: Final = consumer.consume(2, timeout=3)

            assert len(messages) == 1
            assert messages[0].error() is None
            value: Final = messages[0].value()
            assert value is not None
            assert PAYLOAD.validate_json(value) == {"id": "retried-once-by-the-client"}
            assert relay.attempts >= 2
            assert logger.statistics.accepted == 1
            assert logger.statistics.delivered == 1
            assert logger.statistics.dropped == 0
        finally:
            logger.close()
            consumer.close()
            relay.shutdown()
            server.join(5)


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
logger = KafkaLogger(KafkaInitParams(bootstrap_servers={broker.servers!r}, topic={broker.topic!r}))
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
