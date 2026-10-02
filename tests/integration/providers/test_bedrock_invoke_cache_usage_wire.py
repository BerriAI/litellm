import json
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server

MODEL_ID: Final = "us.amazon.nova-pro-v1:0"
TOKEN: Final = "synthetic-bedrock-bearer"
PROMPT: Final = "summarize the cached policy"
INPUT_TOKENS: Final = 11
OUTPUT_TOKENS: Final = 4
CACHE_READ_TOKENS: Final = 900
CACHE_WRITE_TOKENS: Final = 300
INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002
CACHE_READ_RATE: Final = 0.0001
CACHE_WRITE_RATE: Final = 0.0015
RESPONSE: Final = json.dumps(
    {
        "output": {"message": {"role": "assistant", "content": [{"text": "cached policy summary"}]}},
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": INPUT_TOKENS,
            "outputTokens": OUTPUT_TOKENS,
            "totalTokens": INPUT_TOKENS + OUTPUT_TOKENS,
            "cacheReadInputTokenCount": CACHE_READ_TOKENS,
            "cacheWriteInputTokenCount": CACHE_WRITE_TOKENS,
        },
    }
).encode()


def nova_invoke_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == f"/model/{MODEL_ID}/invoke", request.target
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    body: Final = json.loads(request.body)
    assert body["messages"] == [{"role": "user", "content": [{"text": PROMPT}]}], body
    return Reply(body=RESPONSE)


@pytest.mark.covers("providers.bedrock_invoke.count_suffixed_cache_usage_fields_are_reported_and_charged")
def test_nova_invoke_count_suffixed_cache_usage_fields_are_reported_and_charged(gateway: Gateway) -> None:
    with wire_server(nova_invoke_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/invoke/{MODEL_ID}",
            api_key=TOKEN,
            aws_region_name="us-east-1",
            api_base=wire.url,
            input_cost_per_token=INPUT_RATE,
            output_cost_per_token=OUTPUT_RATE,
            cache_read_input_token_cost=CACHE_READ_RATE,
            cache_creation_input_token_cost=CACHE_WRITE_RATE,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "max_tokens": 32, "messages": [{"role": "user", "content": PROMPT}]},
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["choices"][0]["message"]["content"] == "cached policy summary", response.text
        usage: Final = body["usage"]
        assert usage["prompt_tokens"] == INPUT_TOKENS + CACHE_READ_TOKENS + CACHE_WRITE_TOKENS, response.text
        assert usage["completion_tokens"] == OUTPUT_TOKENS, response.text
        assert usage["prompt_tokens_details"]["cached_tokens"] == CACHE_READ_TOKENS, response.text
        assert usage["cache_read_input_tokens"] == CACHE_READ_TOKENS, response.text
        assert usage["cache_creation_input_tokens"] == CACHE_WRITE_TOKENS, response.text
        expected_cost: Final = (
            INPUT_TOKENS * INPUT_RATE
            + CACHE_READ_TOKENS * CACHE_READ_RATE
            + CACHE_WRITE_TOKENS * CACHE_WRITE_RATE
            + OUTPUT_TOKENS * OUTPUT_RATE
        )
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected_cost), response.text
        assert len(wire.drain()) == 1
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT spend, prompt_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (body["id"],)
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert float(rows[0]["spend"]) == pytest.approx(expected_cost), rows
        assert rows[0]["prompt_tokens"] == INPUT_TOKENS + CACHE_READ_TOKENS + CACHE_WRITE_TOKENS, rows
