import base64
import json
from typing import Final
from urllib.parse import unquote

import anthropic
import httpx
import pytest
from pydantic import JsonValue

from integration._support.client import Gateway
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, wire_server

_PROMPT: Final = "Respond with exactly the following text and nothing else:\nHello from LiteLLM!"
_HEAD: Final = "Hello from "
_TAIL: Final = "LiteLLM!"
_TOKEN: Final = "synthetic-bedrock-bearer"


def _frame(event_type: str, payload: dict[str, JsonValue]) -> bytes:
    return _aws_event_frame(event_type, payload, "sc", "u")


def _invoke_chunk(event: dict[str, JsonValue]) -> bytes:
    return _frame("chunk", {"bytes": base64.b64encode(json.dumps(event).encode()).decode()})


_INVOKE_FRAMES: Final = (
    _invoke_chunk(
        {
            "type": "message_start",
            "message": {
                "id": "msg_bedrock_invoke",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-sonnet-4-5-20250929",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 21, "output_tokens": 1},
            },
        }
    ),
    _invoke_chunk({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    _invoke_chunk({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": _HEAD}}),
    _invoke_chunk({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": _TAIL}}),
    _invoke_chunk({"type": "content_block_stop", "index": 0}),
    _invoke_chunk({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}),
    _invoke_chunk({"type": "message_stop"}),
)
_CONVERSE_FRAMES: Final = (
    _frame("messageStart", {"role": "assistant"}),
    _frame("contentBlockDelta", {"delta": {"text": _HEAD}, "contentBlockIndex": 0}),
    _frame("contentBlockDelta", {"delta": {"text": _TAIL}, "contentBlockIndex": 0}),
    _frame("contentBlockStop", {"contentBlockIndex": 0}),
    _frame("messageStop", {"stopReason": "end_turn"}),
    _frame("metadata", {"usage": {"inputTokens": 21, "outputTokens": 5, "totalTokens": 26}, "metrics": {"latencyMs": 1}}),
)


def _bedrock(request: Request) -> Reply:
    frames: Final = _INVOKE_FRAMES if request.target.endswith("/invoke-with-response-stream") else _CONVERSE_FRAMES
    return Reply(chunks=frames, content_type="application/vnd.amazon.eventstream")


@pytest.mark.parametrize(
    ("model", "target"),
    [
        (
            "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "/model/us.anthropic.claude-sonnet-4-5-20250929-v1:0/invoke-with-response-stream",
        ),
        (
            "bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "/model/us.anthropic.claude-sonnet-4-5-20250929-v1:0/converse-stream",
        ),
        ("bedrock/us.amazon.nova-pro-v1:0", "/model/us.amazon.nova-pro-v1:0/converse-stream"),
    ],
    ids=["invoke", "converse", "nova"],
)
def test_messages_stream_from_each_bedrock_api_reaches_the_anthropic_sdk(
    gateway: Gateway, model: str, target: str
) -> None:
    with (
        wire_server(_bedrock) as provider,
        gateway.scenario() as scenario,
        httpx.Client(timeout=15, trust_env=False) as transport,
    ):
        name: Final = scenario.model(model=model, api_key=_TOKEN, aws_region_name="us-east-1", api_base=provider.url)
        client: Final = anthropic.Anthropic(
            api_key=gateway.key, base_url=str(gateway.client.base_url), max_retries=0, http_client=transport
        )
        with client.messages.stream(
            model=name,
            max_tokens=256,
            system="You are a helpful AI assistant. Always follow the user's instructions exactly.",
            messages=[{"role": "user", "content": _PROMPT}],
        ) as stream:
            deltas: Final = tuple(stream.text_stream)
            final: Final = stream.get_final_message()
        sent: Final = provider.drain()

    assert deltas == (_HEAD, _TAIL)
    assert [(block.type, block.text) for block in final.content] == [("text", "Hello from LiteLLM!")]
    assert final.stop_reason == "end_turn"
    assert [(request.method, unquote(request.target)) for request in sent] == [("POST", target)]
    assert sent[0].headers["authorization"] == f"Bearer {_TOKEN}"
    outbound: Final = json.loads(sent[0].body)
    expected_content: Final = _PROMPT if target.endswith("/invoke-with-response-stream") else [{"text": _PROMPT}]
    assert outbound["messages"] == [{"role": "user", "content": expected_content}]
