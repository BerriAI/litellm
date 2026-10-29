import json
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server

_CONTENTS: Final = ({"role": "user", "parts": [{"text": "adapter spend probe"}]},)
_USAGE: Final = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}

_CHAT_BODY: Final = {
    "id": "chatcmpl-adapter",
    "object": "chat.completion",
    "created": 1,
    "model": "adapter-upstream",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": _USAGE,
}

_CHAT_FRAMES: Final = tuple(
    f"data: {json.dumps(frame)}\n\n".encode()
    for frame in (
        {
            "id": "chatcmpl-adapter",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "adapter-upstream",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-adapter",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "adapter-upstream",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl-adapter",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "adapter-upstream",
            "choices": [],
            "usage": _USAGE,
        },
    )
) + (b"data: [DONE]\n\n",)

_CALL_TYPE: Final = {False: "agenerate_content", True: "agenerate_content_stream"}
_SPEND_ROWS: Final = 'SELECT litellm_call_id, call_type, model_group, spend FROM "LiteLLM_SpendLogs" WHERE api_key=%s'


@pytest.mark.parametrize("stream", [False, True])
def test_generate_content_adapter_writes_one_row_under_the_served_call_id(gateway: Gateway, stream: bool) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/chat/completions"
        if stream:
            return Reply(content_type="text/event-stream", chunks=_CHAT_FRAMES)
        return Reply(body=json.dumps(_CHAT_BODY).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/adapter-upstream",
            api_base=wire.url,
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )
        key: Final = scenario.key(models=[model], max_budget=10)
        query: Final = "?alt=sse" if stream else ""
        action: Final = "streamGenerateContent" if stream else "generateContent"
        response: Final = gateway.request(
            "POST",
            f"/v1beta/models/{model}:{action}{query}",
            {"contents": list(_CONTENTS)},
            key=key,
        )
        assert response.status_code == 200, response.text
        digest: Final = sha256(key.encode()).hexdigest()
        rows: Final = eventually(
            lambda: read_rows(_SPEND_ROWS, (digest,)),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert [(row["litellm_call_id"], row["call_type"], row["model_group"]) for row in rows] == [
            (response.headers["x-litellm-call-id"], _CALL_TYPE[stream], model)
        ], rows
        assert float(rows[0]["spend"]) == pytest.approx(11 * 0.001 + 7 * 0.002), rows
        info: Final = eventually(
            lambda: object_value(gateway.get("/key/info", {"key": digest})["info"]),
            lambda values: float(values.get("spend", 0)) > 0,
            seconds=70,
        )
        assert float(info["spend"]) == pytest.approx(11 * 0.001 + 7 * 0.002), info
        assert read_rows(_SPEND_ROWS, (digest,)) == rows
        assert len(wire.drain()) == 1
