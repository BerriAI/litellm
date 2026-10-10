import json
from pathlib import Path
from typing import Final

from tests.integration._support.client import Gateway
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server


def test_chat_stream_forwards_logprobs_and_preserves_them(gateway: Gateway, tmp_path: Path) -> None:
    def upstream(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        assert body["logprobs"] is True
        assert body["top_logprobs"] == 2
        assert body["stream"] is True
        return Reply(
            body=(
                b'data: {"id":"chat-migration","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"hello"},"logprobs":{"content":[{"token":"hello","logprob":-0.1,"bytes":[104,101,108,108,111],"top_logprobs":[]}]}}]}\n\n'
                b'data: {"id":"chat-migration","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                b'data: [DONE]\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    with wire_server(upstream) as provider:
        with owned_proxy(gateway, tmp_path, {}) as proxy:
            with proxy.scenario() as scenario:
                model: Final = scenario.model(
                    model="openai/gpt-4o-mini",
                    api_base=f"{provider.url}/v1",
                    api_key="synthetic-openai-key",
                )
                response: Final = proxy.request(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": model,
                        "messages": [{"role": "user", "content": "logprobs migration"}],
                        "logprobs": True,
                        "top_logprobs": 2,
                        "stream": True,
                    },
                )

    assert response.status_code == 200, response.text
    assert '"logprob":-0.1' in response.text
    assert '"content":"hello"' in response.text
    assert len(provider.drain()) == 1


def test_completion_stream_includes_exact_usage(gateway: Gateway, tmp_path: Path) -> None:
    def upstream(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        assert body["stream_options"] == {"include_usage": True}
        assert body["prompt"] == "completion usage migration"
        return Reply(
            body=(
                b'data: {"id":"completion-migration","object":"text_completion","created":1,"model":"gpt-3.5-turbo-instruct","choices":[{"text":"done","index":0,"logprobs":null,"finish_reason":null}]}\n\n'
                b'data: {"id":"completion-migration","object":"text_completion","created":1,"model":"gpt-3.5-turbo-instruct","choices":[{"text":"","index":0,"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}\n\n'
                b'data: [DONE]\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    with wire_server(upstream) as provider:
        with owned_proxy(gateway, tmp_path, {}) as proxy:
            with proxy.scenario() as scenario:
                model: Final = scenario.model(
                    model="openai/gpt-3.5-turbo-instruct",
                    api_base=f"{provider.url}/v1",
                    api_key="synthetic-openai-key",
                )
                response: Final = proxy.request(
                    "POST",
                    "/v1/completions",
                    {
                        "model": model,
                        "prompt": "completion usage migration",
                        "stream": True,
                        "stream_options": {"include_usage": True},
                        "max_tokens": 4,
                    },
                )

    assert response.status_code == 200, response.text
    assert '"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}' in response.text
    assert len(provider.drain()) == 1
