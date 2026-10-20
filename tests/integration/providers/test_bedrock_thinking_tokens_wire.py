import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

MODEL: Final = "bedrock/converse/global.anthropic.claude-opus-4-8"
TOKEN: Final = "synthetic-bedrock-bearer"
PROMPT: Final = "How many prime numbers are less than 30? Think it through, then answer with just the number."
RESPONSES_PROMPT: Final = "How many prime numbers are less than 30? Answer with just the number."
REDACTED_DATA: Final = "RWRhY3RlZC1ieS1CZWRyb2Nr"
INPUT_TOKENS: Final = 31
OUTPUT_TOKENS: Final = 257
RESPONSE: Final = json.dumps(
    {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"reasoningContent": {"redactedContent": REDACTED_DATA}}, {"text": "10"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": INPUT_TOKENS,
            "outputTokens": OUTPUT_TOKENS,
            "totalTokens": INPUT_TOKENS + OUTPUT_TOKENS,
        },
        "metrics": {"latencyMs": 1},
    }
).encode()
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_JSON_LIST: Final = TypeAdapter(list[dict[str, JsonValue]])


def redacted_thinking_peer(request: Request, prompts: tuple[str, str]) -> Reply:
    assert request.method == "POST" and request.target == "/model/global.anthropic.claude-opus-4-8/converse"
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    body: Final = json.loads(request.body)
    assert body["messages"] in (
        [{"role": "user", "content": [{"text": prompts[0]}]}],
        [{"role": "user", "content": [{"text": prompts[1]}]}],
    ), body
    assert body["additionalModelRequestFields"]["thinking"]["type"] == "adaptive", body
    return Reply(body=RESPONSE)


@pytest.mark.covers("other.provider_wire.bedrock.hidden_thinking_tokens_are_not_reported_as_text")
def test_bedrock_redacted_thinking_is_not_reported_as_zero_reasoning_tokens(gateway: Gateway) -> None:
    identity: Final = " " + uuid.uuid4().hex
    prompts: Final = (PROMPT + identity, RESPONSES_PROMPT + identity)
    with wire_server(lambda request: redacted_thinking_peer(request, prompts)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=MODEL, api_key=TOKEN, aws_region_name="us-east-1", aws_bedrock_runtime_endpoint=wire.url
        )
        chat: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompts[0]}],
                "max_tokens": 4000,
                "reasoning_effort": "max",
            },
        )
        assert chat.status_code == 200, chat.text
        chat_body: Final = _JSON_OBJECT.validate_json(chat.content)
        message: Final = _JSON_OBJECT.validate_python(_JSON_LIST.validate_python(chat_body["choices"])[0]["message"])
        assert message["content"] == "10", chat.text
        assert message["thinking_blocks"] == [{"type": "redacted_thinking", "data": REDACTED_DATA}], chat.text
        usage: Final = _JSON_OBJECT.validate_python(chat_body["usage"])
        assert usage["completion_tokens"] == OUTPUT_TOKENS, chat.text
        details: Final = _JSON_OBJECT.validate_python(usage["completion_tokens_details"])
        assert details == {}, chat.text
        assert len(wire.drain()) == 1

        responses: Final = gateway.request(
            "POST",
            "/v1/responses",
            {"model": model, "input": prompts[1], "max_output_tokens": 4000, "reasoning": {"effort": "max"}},
        )
        assert responses.status_code == 200, responses.text
        responses_body: Final = _JSON_OBJECT.validate_json(responses.content)
        output: Final = _JSON_LIST.validate_python(responses_body["output"])
        reasoning_items: Final = tuple(item for item in output if item["type"] == "reasoning")
        assert len(reasoning_items) == 1, responses.text
        assert reasoning_items[0]["encrypted_content"] == json.dumps(
            [{"type": "redacted_thinking", "data": REDACTED_DATA}], separators=(",", ":")
        ), responses.text
        responses_usage: Final = _JSON_OBJECT.validate_python(responses_body["usage"])
        assert responses_usage["output_tokens"] == OUTPUT_TOKENS, responses.text
        assert _JSON_OBJECT.validate_python(responses_usage["output_tokens_details"])["reasoning_tokens"] == 0, (
            responses.text
        )
        assert len(wire.drain()) == 1
