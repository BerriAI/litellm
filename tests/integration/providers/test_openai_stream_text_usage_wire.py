import json
from collections.abc import Mapping
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])

_IDENTITY: Final = "chatcmpl-stream-usage"


def _frame(delta: Mapping[str, JsonValue], finish: str | None = None) -> bytes:
    return (
        b"data: "
        + json.dumps(
            {
                "id": _IDENTITY,
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        ).encode()
        + b"\n\n"
    )


def test_streaming_chat_assembles_text_and_final_usage(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.target == "/chat/completions"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["stream"] is True, body
        assert body["stream_options"]["include_usage"] is True, body
        usage: Final = json.dumps(
            {
                "id": _IDENTITY,
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
            }
        )
        return Reply(
            content_type="text/event-stream",
            chunks=[
                _frame({"role": "assistant", "content": "Hello "}),
                _frame({"content": "world"}),
                _frame({}, finish="stop"),
                b"data: " + usage.encode() + b"\n\n",
                b"data: [DONE]\n\n",
            ],
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=wire.url)
        response: Final = gateway.client.post(
            "/chat/completions",
            json={
                "model": model,
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"Authorization": f"Bearer {gateway.key}"},
            timeout=30,
        )
        assert response.status_code == 200, response.text
        chunks: Final = tuple(
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        )
        text: Final = "".join(choice["delta"].get("content", "") for chunk in chunks for choice in chunk["choices"])
        assert text == "Hello world"
        usages: Final = tuple(chunk["usage"] for chunk in chunks if chunk.get("usage"))
        assert len(usages) == 1
        assert usages[0]["prompt_tokens"] == 11 and usages[0]["completion_tokens"] == 4
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]
