import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

MODEL: Final = "gpt-audio-1.5"
TOKEN: Final = "scripted-openai-audio-key"
ENCODED_AUDIO: Final = "UklGRg=="
AUDIO_RESPONSE: Final = json.dumps(
    {
        "id": "chatcmpl-audio",
        "object": "chat.completion",
        "created": 1700000000,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "audio": {
                        "id": "audio_123",
                        "data": ENCODED_AUDIO,
                        "expires_at": 1700000060,
                        "transcript": "yes",
                    },
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }
).encode()
AUDIO_STREAM_RESPONSE: Final = b"".join(
    (
        b'data: {"id":"chatcmpl-audio","object":"chat.completion.chunk","created":1700000000,'
        b'"model":"gpt-audio-1.5","choices":[{"index":0,"delta":{"role":"assistant",'
        b'"audio":{"id":"audio_123","expires_at":1700000060,"data":"UklGRg=="}},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-audio","object":"chat.completion.chunk","created":1700000000,'
        b'"model":"gpt-audio-1.5","choices":[{"index":0,"delta":{"audio":{"transcript":"yes"}},'
        b'"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-audio","object":"chat.completion.chunk","created":1700000000,'
        b'"model":"gpt-audio-1.5","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    )
)


def test_openai_audio_input_body_reaches_provider(gateway: Gateway) -> None:
    request_body: Final = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this recording?"},
                    {"type": "input_audio", "input_audio": {"data": ENCODED_AUDIO, "format": "wav"}},
                ],
            }
        ],
        "modalities": ["text", "audio"],
        "audio": {"voice": "alloy", "format": "wav"},
    }

    def audio_peer(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/chat/completions"
        assert json.loads(request.body) == request_body
        return Reply(body=AUDIO_RESPONSE)

    with wire_server(audio_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{MODEL}",
            api_key=TOKEN,
            api_base=wire.url,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {**request_body, "model": model},
        )
        received: Final = wire.drain()

    assert response.status_code == 200, response.text
    assert len(received) == 1
    assert json.loads(response.content)["choices"][0]["message"]["audio"]["transcript"] == "yes"


@pytest.mark.parametrize("stream", [False, True])
def test_openai_audio_response_preserves_audio_fields(gateway: Gateway, stream: bool) -> None:
    request_body: Final = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Respond in one word."}],
        "modalities": ["text", "audio"],
        "audio": {"voice": "alloy", "format": "pcm16"},
        **({"stream_options": {"include_usage": True}} if stream else {}),
        **({"stream": True} if stream else {}),
    }

    def audio_peer(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/chat/completions"
        assert json.loads(request.body) == request_body
        return (
            Reply(content_type="text/event-stream", body=AUDIO_STREAM_RESPONSE)
            if stream
            else Reply(body=AUDIO_RESPONSE)
        )

    with wire_server(audio_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{MODEL}",
            api_key=TOKEN,
            api_base=wire.url,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {**request_body, "model": model},
        )
        received: Final = wire.drain()

    assert response.status_code == 200, response.text
    assert len(received) == 1
    if stream:
        audio_chunks: Final = tuple(
            json.loads(line.removeprefix("data: ")) for line in response.text.splitlines() if line.startswith("data: {")
        )
        audio_delta: Final = audio_chunks[0]["choices"][0]["delta"]["audio"]
        transcript_delta: Final = audio_chunks[1]["choices"][0]["delta"]["audio"]
        assert audio_delta == {"id": "audio_123", "expires_at": 1700000060, "data": ENCODED_AUDIO}
        assert transcript_delta == {"transcript": "yes"}
    else:
        audio: Final = json.loads(response.content)["choices"][0]["message"]["audio"]
        assert audio == {
            "id": "audio_123",
            "data": ENCODED_AUDIO,
            "expires_at": 1700000060,
            "transcript": "yes",
        }
