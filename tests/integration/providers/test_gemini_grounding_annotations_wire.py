import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server


def _grounded_reply(request: Request) -> Reply:
    body: Final = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "The passage"}]},
                "finishReason": "STOP",
                "groundingMetadata": {
                    "groundingChunks": [
                        {"retrievedContext": {"uri": "gs://documents/source"}},
                        {"web": {"uri": "https://example.com/first", "title": "First"}},
                        {"web": {"uri": "https://example.com/second", "title": "Second"}},
                    ],
                    "groundingSupports": [
                        {"segment": {"startIndex": 4, "endIndex": 11}, "groundingChunkIndices": [0, 99, 2, 1]}
                    ],
                },
            }
        ],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "totalTokenCount": 7},
    }
    assert json.loads(request.body)["contents"] == [{"role": "user", "parts": [{"text": "Find sources"}]}]
    if ":streamGenerateContent" in request.target:
        return Reply(content_type="text/event-stream", chunks=(f"data: {json.dumps(body)}\n\n".encode(),))
    return Reply(body=json.dumps(body).encode())


@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
def test_chat_grounding_annotations_include_every_web_source(gateway: Gateway, stream: bool) -> None:
    with wire_server(_grounded_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="gemini/gemini-3.8-flash", api_base=wire.url, api_key="synthetic-key")
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "Find sources"}], "stream": stream},
        )
        assert response.status_code == 200, response.text
        payloads: Final = (
            [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
            if stream
            else [response.json()]
        )
        messages: Final = tuple(payload["choices"][0]["delta" if stream else "message"] for payload in payloads)
        annotations: Final = tuple(message["annotations"] for message in messages if message.get("annotations"))
        assert annotations == (
            [
                {
                    "type": "url_citation",
                    "url_citation": {
                        "start_index": 4,
                        "end_index": 11,
                        "url": f"https://example.com/{name.lower()}",
                        "title": name,
                    },
                }
                for name in ("Second", "First")
            ],
        ), response.text
