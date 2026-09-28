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


def test_tool_result_image_block_and_pasted_image_reach_anthropic_identical(gateway: Gateway) -> None:
    turn1: Final = cc.frontier_request(
        f"cache-bust-{uuid.uuid4().hex}",
        "high",
        64000,
        prompt_text="Read /tmp/cc_probe/dot.png and say what colour it is",
    )
    with_image_result: Final = cc.tool_loop_turn2(
        turn1,
        ({"type": "tool_use", "id": "toolu_img", "name": "Read", "input": {"file_path": "/tmp/cc_probe/dot.png"}},),
        (("toolu_img", [dict(_IMAGE_BLOCK)]),),
    )
    pasted: Final = cc.frontier_request(f"cache-bust-{uuid.uuid4().hex}", "high", 64000)
    pasted["messages"] = [
        {
            "role": "user",
            "content": [
                dict(_IMAGE_BLOCK),
                {"type": "text", "text": f"What colour is this? {uuid.uuid4().hex}"},
            ],
        }
    ]
    seen: list[dict[str, JsonValue]] = []

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/messages", request.target
        body: Final = cc.JSON_OBJECT.validate_json(request.body)
        seen.append(body)
        expected: Final = {**(pasted if len(seen) == 2 else with_image_result), "model": cc.FABLE}
        assert body == expected, {
            key: (expected.get(key), body.get(key))
            for key in expected.keys() | body.keys()
            if expected.get(key) != body.get(key)
        }
        return Reply(
            content_type="text/event-stream",
            chunks=cc.text_stream(
                f"msg_img_{uuid.uuid4().hex}", cc.FABLE, "RED", {"input_tokens": 20, "output_tokens": 2}
            ),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{cc.FABLE}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        headers: Final = cc.cli_headers(gateway.key, cc.FRONTIER_CLI_BETA)
        response1: Final = gateway.request(
            "POST", "/v1/messages", {**with_image_result, "model": model}, params={"beta": "true"}, headers=headers
        )
        assert response1.status_code == 200, response1.text
        response2: Final = gateway.request(
            "POST", "/v1/messages", {**pasted, "model": model}, params={"beta": "true"}, headers=headers
        )
        assert response2.status_code == 200, response2.text
        assert len(wire.drain()) == 2
