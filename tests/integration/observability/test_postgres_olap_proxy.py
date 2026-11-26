import json
import os
from pathlib import Path
from typing import Final
from uuid import uuid4

import psycopg
import yaml
from psycopg import sql

from tests.integration._support.client import gateway_from_environment
from tests.integration._support.process import owned_proxy
from tests.integration._support.redis_process import owned_redis
from tests.integration._support.wire import Reply, Request, wire_server


def _provider(request: Request) -> Reply:
    body: Final = json.loads(request.body)
    if body["messages"][0]["content"] == "failure":
        return Reply(status=429, body=b'{"error":{"message":"synthetic limit","type":"rate_limit_error"}}')
    usage: Final = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    common: Final = {"id": "chatcmpl-" + uuid4().hex, "created": 1, "model": body["model"]}
    if body.get("stream"):
        chunk: Final = {**common, "object": "chat.completion.chunk"}
        return Reply(
            content_type="text/event-stream",
            chunks=(
                (
                    "data: "
                    + json.dumps(
                        {
                            **chunk,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": "private response"},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                    + "\n\n"
                ).encode(),
                (
                    "data: "
                    + json.dumps(
                        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage}
                    )
                    + "\n\n"
                ).encode(),
                b"data: [DONE]\n\n",
            ),
        )
    return Reply(
        body=json.dumps(
            {
                **common,
                "object": "chat.completion",
                "usage": usage,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "private response"},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode()
    )


def test_proxy_drains_success_failure_and_stream_logs_on_shutdown(tmp_path: Path) -> None:
    table: Final = "olap_proxy_" + uuid4().hex
    dsn: Final = os.environ["DATABASE_URL"]
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(
            sql.SQL("""CREATE TABLE {} (
            event_id TEXT NOT NULL, request_id TEXT, status TEXT, model TEXT,
            start_time TIMESTAMPTZ, end_time TIMESTAMPTZ, prompt_tokens BIGINT,
            completion_tokens BIGINT, total_tokens BIGINT, response_cost DOUBLE PRECISION, payload TEXT
        )""").format(sql.Identifier(table))
        )
        try:
            with (
                wire_server(_provider) as provider,
                owned_redis(tmp_path / "redis") as cache,
                gateway_from_environment() as rig,
            ):
                (tmp_path / "olap_callback.py").write_text(
                    "import os\nfrom pydantic import SecretStr\n"
                    "from litellm.integrations.postgres_olap import PostgresOlapConfig, PostgresOlapLogger\n"
                    "logger = PostgresOlapLogger(PostgresOlapConfig(\n"
                    "    dsn=SecretStr(os.environ['POSTGRES_OLAP_DSN']),\n"
                    "    table_name=os.environ['POSTGRES_OLAP_TABLE'], flush_interval=3600,\n"
                    "))\n"
                )
                config: Final = tmp_path / "proxy.yaml"
                config.write_text(
                    yaml.safe_dump(
                        {
                            "model_list": [
                                {
                                    "model_name": "analytics",
                                    "litellm_params": {
                                        "model": "openai/olap-test",
                                        "api_key": "synthetic",
                                        "api_base": provider.url + "/v1",
                                    },
                                }
                            ],
                            "litellm_settings": {"callbacks": ["olap_callback.logger"]},
                            "router_settings": {"num_retries": 0, "redis_host": cache.host, "redis_port": cache.port},
                            "general_settings": {
                                "master_key": "os.environ/LITELLM_MASTER_KEY",
                                "disable_model_info_refresh": True,
                                "coordination_redis": {"host": cache.host, "port": cache.port},
                            },
                        }
                    )
                )
                with owned_proxy(
                    rig,
                    tmp_path,
                    {"POSTGRES_OLAP_DSN": dsn, "POSTGRES_OLAP_TABLE": table, "STORE_MODEL_IN_DB": "False"},
                    config=config,
                    workers=2,
                    database_setup=(),
                    remove_environment=("DATABASE_URL", "DATABASE_URL_READ_REPLICA"),
                ) as proxy:
                    success: Final = proxy.request(
                        "POST",
                        "/v1/chat/completions",
                        {
                            "model": "analytics",
                            "messages": [{"role": "user", "content": "private prompt"}],
                        },
                    )
                    assert success.status_code == 200, success.text
                    failure: Final = proxy.request(
                        "POST",
                        "/v1/chat/completions",
                        {
                            "model": "analytics",
                            "messages": [{"role": "user", "content": "failure"}],
                        },
                    )
                    assert failure.status_code == 429, failure.text
                    streaming: Final = proxy.request(
                        "POST",
                        "/v1/chat/completions",
                        {
                            "model": "analytics",
                            "messages": [{"role": "user", "content": "private prompt"}],
                            "stream": True,
                            "stream_options": {"include_usage": True},
                        },
                    )
                    assert streaming.status_code == 200 and "[DONE]" in streaming.text, streaming.text
                    assert connection.execute(
                        sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
                    ).fetchone() == (0,)
                rows: Final = connection.execute(
                    sql.SQL("SELECT event_id,status,total_tokens,payload FROM {}").format(sql.Identifier(table))
                ).fetchall()
                assert len(rows) == 3, rows
                assert len({row[0] for row in rows}) == 3
                assert sorted(row[1] for row in rows) == ["failure", "success", "success"]
                assert [row[2] for row in rows if row[1] == "success"] == [7, 7]
                assert all("private prompt" not in row[3] and "private response" not in row[3] for row in rows)
        finally:
            connection.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(table)))
