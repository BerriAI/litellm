import json
from pathlib import Path
from typing import Final

import anthropic
import httpx
import pytest

from tests.integration._support.client import Gateway
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server

_EVENTS: Final = (
    {
        "type": "message_start",
        "message": {
            "id": "msg_thinking_stream",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "claude-haiku-4-5",
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 1},
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Two plus two "}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "is four."}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig-thinking"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "The answer "}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "is 4."}},
    {"type": "content_block_stop", "index": 1},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 9}},
    {"type": "message_stop"},
)
_THINKING: Final = {"type": "enabled", "budget_tokens": 1024}


def _stream(request: Request) -> Reply:
    return Reply(
        chunks=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in _EVENTS),
        content_type="text/event-stream",
    )


@pytest.mark.parametrize("route", ["/anthropic", ""], ids=["anthropic_pass_through", "v1_messages"])
def test_messages_stream_relays_thinking_and_text_deltas(gateway: Gateway, tmp_path: Path, route: str) -> None:
    with wire_server(_stream) as provider, owned_proxy(
        gateway,
        tmp_path,
        {"ANTHROPIC_API_BASE": provider.url, "ANTHROPIC_API_KEY": "synthetic-anthropic-key"},
    ) as proxy, proxy.scenario() as scenario, httpx.Client(timeout=15, trust_env=False) as transport:
        model: Final = scenario.model(
            model="anthropic/claude-haiku-4-5", api_base=provider.url, api_key="synthetic-anthropic-key"
        )
        client: Final = anthropic.Anthropic(
            api_key=proxy.key,
            base_url=f"{str(proxy.client.base_url).rstrip('/')}{route}",
            max_retries=0,
            http_client=transport,
        )
        with client.messages.stream(
            model=model if route == "" else "claude-haiku-4-5",
            max_tokens=2048,
            thinking=_THINKING,
            messages=[{"role": "user", "content": "What is 2+2?"}],
        ) as stream:
            deltas: Final = tuple(
                (event.delta.type, getattr(event.delta, "thinking", None) or getattr(event.delta, "text", None))
                for event in stream
                if event.type == "content_block_delta" and event.delta.type in ("thinking_delta", "text_delta")
            )
            final: Final = stream.get_final_message()
        sent: Final = tuple(request for request in provider.drain() if request.target.endswith("/v1/messages"))

    assert deltas == (
        ("thinking_delta", "Two plus two "),
        ("thinking_delta", "is four."),
        ("text_delta", "The answer "),
        ("text_delta", "is 4."),
    )
    assert [block.type for block in final.content] == ["thinking", "text"]
    assert (final.content[0].thinking, final.content[0].signature) == ("Two plus two is four.", "sig-thinking")
    assert final.content[1].text == "The answer is 4."
    assert len(sent) == 1
    outbound: Final = json.loads(sent[0].body)
    assert (outbound["thinking"], outbound["stream"], outbound["max_tokens"]) == (_THINKING, True, 2048)
    assert outbound["messages"] == [{"role": "user", "content": "What is 2+2?"}]
