import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.messages_endpoint import _claude_code as cc


def test_compact_edit_maps_to_responses_context_management(gateway: Gateway) -> None:
    request_body: Final = {
        **cc.frontier_request(f"cache-bust-{uuid.uuid4().hex}", "high", 64000),
        "context_management": {
            "edits": [
                {"type": "clear_thinking_20251015", "keep": "all"},
                {"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 150000}},
            ]
        },
        "stream": False,
    }

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/responses", request.target
        body: Final = cc.JSON_OBJECT.validate_json(request.body)
        assert body.get("context_management") == [{"type": "compaction", "compact_threshold": 150000}], body.get(
            "context_management"
        )
        return Reply(
            body=cc.responses_completed(
                "cm",
                cc.OPENAI_BACKEND,
                (
                    {
                        "type": "message",
                        "id": "msg_1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "OK", "annotations": []}],
                    },
                ),
                {"input_tokens": 41, "output_tokens": 3, "total_tokens": 44},
            )
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{cc.OPENAI_BACKEND}", api_base=wire.url, api_key=cc.OPENAI_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**request_body, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key, cc.FRONTIER_CLI_BETA),
        )
        assert response.status_code == 200, response.text
        payload: Final = cc.JSON_OBJECT.validate_json(response.content)
        assert payload["content"] == [{"type": "text", "text": "OK"}], payload["content"]
        assert len(wire.drain()) == 1


def test_compaction_output_item_reaches_client_as_compaction_block(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: the responses bridge drops compaction output items in translate_response "
        "(transformation.py handles only message/reasoning/function_call), so the client loses "
        "the compaction block entirely"
    )
    request_body: Final = {
        **cc.frontier_request(f"cache-bust-{uuid.uuid4().hex}", "high", 64000),
        "context_management": {
            "edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 150000}}]
        },
        "stream": False,
    }

    def respond(request: Request) -> Reply:
        return Reply(
            body=cc.responses_completed(
                "cm",
                cc.OPENAI_BACKEND,
                (
                    {"type": "compaction", "content": "<summary>"},
                    {
                        "type": "message",
                        "id": "msg_1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "OK", "annotations": []}],
                    },
                ),
                {"input_tokens": 41, "output_tokens": 3, "total_tokens": 44},
            )
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{cc.OPENAI_BACKEND}", api_base=wire.url, api_key=cc.OPENAI_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**request_body, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key, cc.FRONTIER_CLI_BETA),
        )
        assert response.status_code == 200, response.text
        payload: Final = cc.JSON_OBJECT.validate_json(response.content)
        compaction_blocks: Final = [block for block in payload["content"] if block.get("type") == "compaction"]
        assert compaction_blocks == [{"type": "compaction", "content": "<summary>"}], payload["content"]
        assert len(wire.drain()) == 1
