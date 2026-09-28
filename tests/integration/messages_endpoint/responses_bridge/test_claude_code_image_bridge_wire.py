import uuid
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.messages_endpoint import _claude_code as cc
from pydantic import JsonValue

_PNG_B64: Final = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAEElEQVR4nGP4z8AARwzEcQCukw/x0F8jngAAAABJRU5ErkJggg=="
_IMAGE_BLOCK: Final = {
    "type": "image",
    "source": {"type": "base64", "data": _PNG_B64, "media_type": "image/png"},
}
_DATA_URL: Final = f"data:image/png;base64,{_PNG_B64}"


def _respond_ok(request: Request) -> Reply:
    return Reply(
        body=cc.responses_completed(
            "img",
            cc.OPENAI_BACKEND,
            (
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "RED", "annotations": []}],
                },
            ),
            {"input_tokens": 41, "output_tokens": 3, "total_tokens": 44},
        )
    )


def test_tool_result_image_maps_to_input_image_on_bridge(gateway: Gateway) -> None:
    turn1: Final = {
        **cc.frontier_request(
            f"cache-bust-{uuid.uuid4().hex}",
            "high",
            64000,
            prompt_text="Read /tmp/cc_probe/dot.png and say what colour it is",
        ),
        "stream": False,
    }
    turn2: Final = cc.tool_loop_turn2(
        turn1,
        ({"type": "tool_use", "id": "call_img", "name": "Read", "input": {"file_path": "/tmp/cc_probe/dot.png"}},),
        (("call_img", [dict(_IMAGE_BLOCK)]),),
    )

    def respond(request: Request) -> Reply:
        body: Final = cc.JSON_OBJECT.validate_json(request.body)
        outputs: Final = [
            item for item in body["input"] if isinstance(item, dict) and item.get("type") == "function_call_output"
        ]
        assert len(outputs) == 1 and outputs[0]["call_id"] == "call_img", outputs
        image_messages: Final = [
            item
            for item in body["input"]
            if isinstance(item, dict)
            and item.get("type") == "message"
            and item.get("role") == "user"
            and any(isinstance(part, dict) and part.get("type") == "input_image" for part in item.get("content", ()))
        ]
        assert image_messages, body["input"]
        image_parts: Final = [
            part
            for part in image_messages[0]["content"]
            if isinstance(part, dict) and part.get("type") == "input_image"
        ]
        assert image_parts[0]["image_url"] == _DATA_URL, image_parts
        return _respond_ok(request)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{cc.OPENAI_BACKEND}", api_base=wire.url, api_key=cc.OPENAI_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**turn2, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key, cc.FRONTIER_CLI_BETA),
        )
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1


def test_pasted_image_maps_to_input_image_on_bridge(gateway: Gateway) -> None:
    request_body: Final = {**cc.frontier_request(f"cache-bust-{uuid.uuid4().hex}", "high", 64000), "stream": False}
    pasted_text: Final = f"What colour is this? {uuid.uuid4().hex}"
    request_body["messages"] = [
        {"role": "user", "content": [dict(_IMAGE_BLOCK), {"type": "text", "text": pasted_text}]}
    ]

    def respond(request: Request) -> Reply:
        body: Final = cc.JSON_OBJECT.validate_json(request.body)
        user_msg: Final = body["input"][0]
        assert user_msg["content"][0] == {"type": "input_image", "image_url": _DATA_URL}, user_msg
        assert user_msg["content"][1] == {"type": "input_text", "text": pasted_text}, user_msg
        return _respond_ok(request)

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
        assert len(wire.drain()) == 1
