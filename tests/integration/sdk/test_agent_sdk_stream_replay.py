import json
from pathlib import Path
from typing import Final

import anthropic

from tests.integration._support.client import Gateway
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server


def test_messages_stream_reaches_anthropic_sdk(gateway: Gateway, tmp_path: Path) -> None:
    def upstream(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        assert body["messages"] == [{"role": "user", "content": "Say hello"}]
        return Reply(
            body=(
                b'event: message_start\n'
                b'data: {"type":"message_start","message":{"id":"msg_agent_sdk","type":"message","role":"assistant","content":[],"model":"claude-haiku-4-5","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":2,"output_tokens":0}}}\n\n'
                b'event: content_block_start\n'
                b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
                b'event: content_block_delta\n'
                b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello from LiteLLM!"}}\n\n'
                b'event: content_block_stop\n'
                b'data: {"type":"content_block_stop","index":0}\n\n'
                b'event: message_delta\n'
                b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":4}}\n\n'
                b'event: message_stop\n'
                b'data: {"type":"message_stop"}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    with wire_server(upstream) as provider:
        with owned_proxy(gateway, tmp_path, {}) as proxy:
            with proxy.scenario() as scenario:
                model: Final = scenario.model(
                    model="anthropic/claude-haiku-4-5",
                    api_base=provider.url,
                    api_key="synthetic-anthropic-key",
                )
                with anthropic.Anthropic(api_key=proxy.key, base_url=str(proxy.client.base_url)) as client:
                    with client.messages.stream(
                        model=model,
                        max_tokens=32,
                        messages=[{"role": "user", "content": "Say hello"}],
                    ) as stream:
                        text: Final = "".join(stream.text_stream)

    assert text == "Hello from LiteLLM!"
    assert len(provider.drain()) == 1
