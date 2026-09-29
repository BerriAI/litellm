import uuid
from typing import Final

from integration._support import claude_code as cc
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server


def test_client_disconnect_mid_stream_still_bills_the_message(gateway: Gateway) -> None:
    identity: Final = f"msg_dc_{uuid.uuid4().hex}"
    request_body: Final = cc.claude_code_request(f"cache-bust-{uuid.uuid4().hex}")

    def respond(request: Request) -> Reply:
        return Reply(
            content_type="text/event-stream",
            chunks=(
                cc.sse_frame(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": identity,
                            "type": "message",
                            "role": "assistant",
                            "model": cc.SONNET,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 12, "output_tokens": 1},
                        },
                    },
                ),
                cc.sse_frame(
                    "content_block_start",
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                ),
                cc.sse_frame(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "PONG"}},
                ),
                cc.sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
                cc.sse_frame(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": 4},
                    },
                ),
                cc.sse_frame("message_stop", {"type": "message_stop"}),
            ),
            pause_between_chunks=3.0,
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{cc.SONNET}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        with gateway.client.stream(
            "POST",
            "/v1/messages",
            params={"beta": "true"},
            json={**request_body, "model": model},
            headers={
                **cc.cli_headers(gateway.key),
                "authorization": f"Bearer {gateway.key}",
            },
        ) as response:
            assert response.status_code == 200, response.status_code
            first: Final = next(response.iter_text())
            assert "message_start" in first, first
        assert len(wire.drain()) == 1
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT prompt_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (identity,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
            return_last_on_timeout=True,
        )
        assert rows and rows[0]["prompt_tokens"] == 12, rows
        assert wire.disconnected.empty(), (
            "closing the client stream must not abort the upstream call before it finishes; "
            f"wire recorded a disconnect on {wire.disconnected.get_nowait()}"
        )
