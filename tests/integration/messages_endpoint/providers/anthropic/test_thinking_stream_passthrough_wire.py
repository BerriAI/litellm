import json
from typing import Final

from tests.integration._support.client import Gateway
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server


def test_messages_stream_preserves_thinking_and_text_deltas(gateway: Gateway, tmp_path) -> None:
    def upstream(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}
        return Reply(
            body=(
                b'event: message_start\n'
                b'data: {"type":"message_start","message":{"id":"msg_migration","type":"message","role":"assistant","content":[],"model":"claude-haiku-4-5","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
                b'event: content_block_start\n'
                b'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}\n\n'
                b'event: content_block_delta\n'
                b'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"reasoning"}}\n\n'
                b'event: content_block_stop\n'
                b'data: {"type":"content_block_stop","index":0}\n\n'
                b'event: content_block_start\n'
                b'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}\n\n'
                b'event: content_block_delta\n'
                b'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"hello"}}\n\n'
                b'event: content_block_stop\n'
                b'data: {"type":"content_block_stop","index":1}\n\n'
                b'event: message_delta\n'
                b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}\n\n'
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
                response: Final = proxy.request(
                    "POST",
                    "/v1/messages",
                    {
                        "model": model,
                        "max_tokens": 20000,
                        "thinking": {"type": "enabled", "budget_tokens": 1024},
                        "messages": [{"role": "user", "content": "thinking migration request"}],
                        "stream": True,
                    },
                )

        assert response.status_code == 200, response.text
        assert '"thinking_delta","thinking":"reasoning"' in response.text
        assert '"text_delta","text":"hello"' in response.text
        assert len(provider.drain()) == 1
