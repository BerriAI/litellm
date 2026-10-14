import json
from base64 import b64decode
from collections.abc import Callable
from hashlib import sha256
from typing import Final
from uuid import uuid4

import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

PROMPT_TOKENS: Final = 30
COMPLETION_TOKENS: Final = 5
INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002
EXPECTED_SPEND: Final = PROMPT_TOKENS * INPUT_RATE + COMPLETION_TOKENS * OUTPUT_RATE


def _responses_reply(prompt: str, response_id: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(
                body=json.dumps({"object": "list", "data": [{"id": "gpt-6.1-sol", "object": "model"}]}).encode()
            )
        assert request.method == "POST", request.method
        assert request.target == "/v1/responses", request.target
        body: Final = object_value(json.loads(request.body))
        encoded_input: Final = json.dumps(body["input"])
        assert prompt in encoded_input, encoded_input
        assert body.get("stream") is not True, body
        response: Final = {
            "id": response_id,
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": body["model"],
            "output": [
                {
                    "type": "message",
                    "id": f"msg_{uuid4().hex}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": PROMPT_TOKENS,
                "output_tokens": COMPLETION_TOKENS,
                "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
            },
        }
        return Reply(body=json.dumps(response).encode())

    return respond


def _rows_for_key(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT litellm_call_id, request_id, spend FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
        (sha256(key.encode()).hexdigest(),),
    )


@pytest.mark.timeout(120)
def test_bridged_chat_completion_above_log_offload_threshold_logs_and_charges_once(gateway: Gateway) -> None:
    prompt_prefix: Final = f"spend once {uuid4().hex[:8]} "
    long_prompt: Final = (prompt_prefix + "lorem ipsum " * 30_000)[:300_000]
    upstream_id: Final = f"resp_{uuid4().hex}"
    with (
        wire_server(_responses_reply(long_prompt, upstream_id)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="openai/responses/gpt-6.1-sol",
            api_base=f"{wire.url}/v1",
            input_cost_per_token=INPUT_RATE,
            output_cost_per_token=OUTPUT_RATE,
        )
        key: Final = scenario.key(models=[model])
        digest: Final = sha256(key.encode()).hexdigest()

        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": long_prompt}], "stream": False},
            key=key,
        )
        assert response.status_code == 200, response.text
        assert "ok" in response.text, response.text
        assert response.json()["id"] == upstream_id, response.text
        call_id: Final = response.headers["x-litellm-call-id"]

        rows: Final = eventually(
            lambda: _rows_for_key(key),
            lambda values: any(row["litellm_call_id"] == call_id for row in values),
            seconds=70,
        )
        call_rows: Final = [row for row in rows if row["litellm_call_id"] == call_id]
        assert len(call_rows) == 1, rows
        assert float(str(call_rows[0]["spend"])) == pytest.approx(EXPECTED_SPEND), rows
        logged_id: Final = b64decode(str(call_rows[0]["request_id"]).removeprefix("resp_")).decode()
        assert logged_id.startswith("litellm:") and f";response_id:{upstream_id}" in logged_id, rows

        key_spend: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest,)),
            lambda values: len(values) == 1 and float(str(values[0]["spend"])) >= EXPECTED_SPEND,
            seconds=70,
        )
        assert float(str(key_spend[0]["spend"])) == pytest.approx(EXPECTED_SPEND), key_spend
