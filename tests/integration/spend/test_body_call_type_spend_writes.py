import json
from hashlib import sha256
from pathlib import Path
from typing import Final
from uuid import uuid4

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.openai_wire import answering_model_discovery, chat_reply
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.shares_worker_state

UPSTREAM_MODEL: Final = "body-call-type-upstream"


def _batch_reply(batch_id: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": batch_id,
                "object": "batch",
                "endpoint": "/v1/chat/completions",
                "errors": None,
                "input_file_id": "file-body-call-type",
                "completion_window": "24h",
                "status": "validating",
                "output_file_id": None,
                "error_file_id": None,
                "created_at": 1,
                "expires_at": 1,
                "request_counts": {"total": 0, "completed": 0, "failed": 0},
                "metadata": None,
            }
        ).encode()
    )


def _proxy_config(directory: Path, alias: str, wire_url: str) -> Path:
    deployment: Final[dict[str, JsonValue]] = {
        "model_name": alias,
        "litellm_params": {
            "model": f"openai/{UPSTREAM_MODEL}",
            "api_base": f"{wire_url}/v1",
            "api_key": "integration-provider-key",
            "input_cost_per_token": 0.001,
            "output_cost_per_token": 0.002,
        },
    }
    config: Final = directory / "body_call_type_config.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": [deployment],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "disable_spend_logs": False,
                    "proxy_batch_write_at": 1,
                },
                "router_settings": {"disable_cooldowns": True},
            }
        )
    )
    return config


@pytest.mark.timeout(240)
def test_a_non_string_body_call_type_keeps_the_proxy_call_type_and_never_blocks_later_spend_rows(
    gateway: Gateway, tmp_path: Path
) -> None:
    alias: Final = f"body-call-type-{uuid4().hex}"
    upstream_batch_id: Final = f"batch_{uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.target.endswith("/batches"):
            return _batch_reply(upstream_batch_id)
        assert request.target.endswith("/chat/completions"), request.target
        return chat_reply(f"chatcmpl-{uuid4().hex}", UPSTREAM_MODEL, "ok", stream=False)

    with wire_server(answering_model_discovery(respond)) as wire:
        config: Final = _proxy_config(tmp_path, alias, wire.url)
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
            key: Final = scenario.key(user_id=f"body-call-type-{uuid4().hex}")
            batch: Final = candidate.request(
                "POST",
                "/v1/batches",
                {
                    "model": alias,
                    "input_file_id": "file-body-call-type",
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                    "call_type": 5,
                },
                key=key,
            )
            assert batch.status_code == 200, batch.text
            chat: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": alias, "messages": [{"role": "user", "content": "after the batch"}]},
                key=key,
            )
            assert chat.status_code == 200, chat.text

            rows: Final = eventually(
                lambda: read_rows(
                    'SELECT request_id, call_type FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
                    (sha256(key.encode()).hexdigest(),),
                ),
                lambda values: len(values) == 2,
                seconds=70,
            )
            assert {str(row["request_id"]): row["call_type"] for row in rows} == {
                batch.json()["id"]: "acreate_batch",
                chat.json()["id"]: "acompletion",
            }
