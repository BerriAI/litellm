import json
from math import isclose
from pathlib import Path
from typing import Final
from uuid import uuid4

import pytest
import yaml
from confluent_kafka import OFFSET_BEGINNING, Consumer, Producer, TopicPartition
from confluent_kafka.admin import ClusterMetadata
from pydantic import ConfigDict, JsonValue, TypeAdapter

from tests.integration._support.client import Gateway, eventually, object_value, string_value
from tests.integration._support.ports import allocate_port
from tests.integration._support.process import owned_proxy_process
from tests.integration._support.wire import Reply, Request, wire_server

METADATA: Final = TypeAdapter(ClusterMetadata, config=ConfigDict(arbitrary_types_allowed=True))
NUMBER: Final = TypeAdapter(float)
JSON: Final = TypeAdapter(dict[str, JsonValue])


def completion(request: Request) -> Reply:
    assert request.method == "POST"
    assert request.target == "/v1/chat/completions"
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid4().hex}",
                "object": "chat.completion",
                "created": 1735689600,
                "model": "gpt-6-luna",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "warehouse-output"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            }
        ).encode()
    )


@pytest.mark.parametrize("mode", ["healthy", "unavailable", "queue_full"])
def test_kafka_export_preserves_pg_spend_and_virtual_key_counters(gateway: Gateway, tmp_path: Path, mode: str) -> None:
    with Producer({"test.mock.num.brokers": 1, "log_level": 0}) as broker, wire_server(completion) as upstream:
        topic: Final = f"litellm-{uuid4().hex}"
        metadata: Final = METADATA.validate_python(broker.list_topics(topic, timeout=10))
        servers: Final = ",".join(f"{node.host}:{node.port}" for node in metadata.brokers.values())
        config: Final = tmp_path / "kafka.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [],
                    "general_settings": {
                        "master_key": "os.environ/LITELLM_MASTER_KEY",
                        "database_url": "os.environ/DATABASE_URL",
                        "store_model_in_db": True,
                        "disable_model_info_refresh": True,
                        "store_prompts_in_spend_logs": True,
                        "database_connection_pool_limit": 4,
                        "proxy_batch_write_at": 1,
                        "proxy_batch_polling_interval": 1,
                        "coordination_redis": {"host": "os.environ/REDIS_HOST", "port": "os.environ/REDIS_PORT"},
                    },
                    "router_settings": {"redis_host": "os.environ/REDIS_HOST", "redis_port": "os.environ/REDIS_PORT"},
                    "litellm_settings": {
                        "callbacks": ["kafka"],
                        "kafka_params": {
                            "security_protocol": "PLAINTEXT",
                            "bootstrap_servers": servers if mode == "healthy" else f"127.0.0.1:{allocate_port()}",
                            "topic": topic,
                            "turn_off_message_logging": True,
                            "request_timeout_ms": 500,
                            "delivery_timeout_ms": 300_000 if mode == "queue_full" else 1000,
                            "queue_max_messages": 1 if mode == "queue_full" else 10_000,
                            "flush_timeout": 1,
                        },
                    },
                }
            )
        )
        consumer: Final = Consumer(
            {"bootstrap.servers": servers, "group.id": uuid4().hex, "enable.auto.commit": False, "log_level": 0}
        )
        try:
            consumer.assign(
                [TopicPartition(topic, partition, OFFSET_BEGINNING) for partition in metadata.topics[topic].partitions]
            )
            with (
                owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as running,
                running.gateway.scenario() as scenario,
            ):
                candidate: Final = running.gateway
                model: Final = scenario.model(
                    model="openai/gpt-6-luna",
                    api_base=upstream.url + "/v1",
                    input_cost_per_token=0.001,
                    output_cost_per_token=0.002,
                )
                alias: Final = f"warehouse-{uuid4().hex}"
                key: Final = scenario.key(key_alias=alias, models=[model])
                responses: Final = tuple(
                    candidate.request(
                        "POST",
                        "/v1/chat/completions",
                        {"model": model, "messages": [{"role": "user", "content": "warehouse-input"}]},
                        key=key,
                    )
                    for _ in range(3 if mode == "queue_full" else 1)
                )
                assert all(response.status_code == 200 for response in responses), tuple(
                    response.text for response in responses
                )
                identities: Final = frozenset(
                    string_value(JSON.validate_json(response.content)["id"]) for response in responses
                )
                tokens: Final = candidate.rows(
                    'SELECT token FROM "LiteLLM_VerificationToken" WHERE key_alias=%s', (alias,)
                )
                digest: Final = string_value(tokens[0]["token"])
                rows: Final = eventually(
                    lambda: candidate.rows(
                        "SELECT request_id, spend, prompt_tokens, completion_tokens, total_tokens, proxy_server_request, response "
                        'FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
                        (digest,),
                    ),
                    lambda values: len(values) == len(responses),
                    seconds=70,
                )
                cost: Final = 10 * 0.001 + 4 * 0.002
                total_cost: Final = cost * len(responses)
                assert frozenset(string_value(row["request_id"]) for row in rows) == identities
                assert tuple(
                    (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) for row in rows
                ) == tuple((10, 4, 14) for _ in responses)
                assert all(isclose(NUMBER.validate_python(row["spend"]), cost, abs_tol=1e-12) for row in rows)
                assert all("warehouse-input" in json.dumps(row["proxy_server_request"]) for row in rows)
                assert all("warehouse-output" in json.dumps(row["response"]) for row in rows)
                spending: Final = eventually(
                    lambda: candidate.rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest,)),
                    lambda values: (
                        len(values) == 1
                        and isclose(NUMBER.validate_python(values[0]["spend"]), total_cost, abs_tol=1e-12)
                    ),
                    seconds=70,
                )
                assert isclose(NUMBER.validate_python(spending[0]["spend"]), total_cost, abs_tol=1e-12)
                if mode != "healthy":
                    reason: Final = "queue_full" if mode == "queue_full" else "delivery_failed"
                    observed: Final = eventually(
                        lambda: running.log.read_text(), lambda body: f"reason={reason}" in body
                    )
                    assert f"reason={reason}" in observed
                if mode == "healthy":
                    messages: Final = consumer.consume(1, timeout=10)
                    assert len(messages) == 1
                    raw: Final = messages[0].value()
                    assert raw is not None
                    event: Final = JSON.validate_json(raw)
                    assert event["id"] in identities
                    assert object_value(event["metadata"])["user_api_key_alias"] == alias
                    assert event["total_tokens"] == 14
                    assert isclose(NUMBER.validate_python(event["response_cost"]), cost, abs_tol=1e-12)
                    assert b"warehouse-input" not in raw and b"warehouse-output" not in raw
        finally:
            consumer.close()


def test_named_kafka_callback_exports_without_application_database(gateway: Gateway, tmp_path: Path) -> None:
    from tests.integration._support.redis_process import owned_redis

    with (
        Producer({"test.mock.num.brokers": 1, "log_level": 0}) as broker,
        wire_server(completion) as upstream,
        owned_redis(tmp_path / "redis") as redis,
    ):
        topic: Final = "litellm-" + uuid4().hex
        metadata: Final = METADATA.validate_python(broker.list_topics(topic, timeout=10))
        servers: Final = ",".join(f"{node.host}:{node.port}" for node in metadata.brokers.values())
        config: Final = tmp_path / "kafka.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [
                        {
                            "model_name": "analytics",
                            "litellm_params": {
                                "model": "openai/gpt-6-luna",
                                "api_key": "synthetic",
                                "api_base": upstream.url + "/v1",
                            },
                        }
                    ],
                    "litellm_settings": {
                        "callbacks": ["kafka"],
                        "kafka_params": {
                            "bootstrap_servers": servers,
                            "topic": topic,
                            "security_protocol": "PLAINTEXT",
                            "turn_off_message_logging": True,
                        },
                    },
                    "router_settings": {"redis_host": redis.host, "redis_port": redis.port},
                    "general_settings": {
                        "master_key": "os.environ/LITELLM_MASTER_KEY",
                        "disable_model_info_refresh": True,
                        "coordination_redis": {"host": redis.host, "port": redis.port},
                    },
                }
            )
        )
        consumer: Final = Consumer(
            {"bootstrap.servers": servers, "group.id": uuid4().hex, "enable.auto.commit": False, "log_level": 0}
        )
        try:
            consumer.assign(
                [TopicPartition(topic, partition, OFFSET_BEGINNING) for partition in metadata.topics[topic].partitions]
            )
            with owned_proxy_process(
                gateway,
                tmp_path,
                {"STORE_MODEL_IN_DB": "False"},
                config=config,
                workers=2,
                database_setup=(),
                remove_environment=("DATABASE_URL", "DATABASE_URL_READ_REPLICA"),
            ) as running:
                assert len(running.worker_pids()) == 2
                response: Final = running.gateway.request(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": "analytics",
                        "messages": [{"role": "user", "content": "warehouse-input"}],
                    },
                )
                assert response.status_code == 200, response.text
            messages: Final = consumer.consume(1, timeout=10)
            assert len(messages) == 1 and messages[0].error() is None
            raw: Final = messages[0].value()
            assert raw is not None
            event: Final = JSON.validate_json(raw)
            assert event["id"] == JSON.validate_json(response.content)["id"]
            assert event["status"] == "success" and event["total_tokens"] == 14
            assert b"warehouse-input" not in raw and b"warehouse-output" not in raw
        finally:
            consumer.close()
