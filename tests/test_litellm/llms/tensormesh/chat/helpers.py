from typing import Any


class _FakeOpenAITextCompletionResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self) -> dict[str, Any]:
        return self._payload


class _FakeOpenAIRawResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def parse(self) -> _FakeOpenAITextCompletionResponse:
        return _FakeOpenAITextCompletionResponse(self._payload)


class _FakeOpenAIStreamRawResponse:
    headers: dict[str, str] = {}

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def parse(self):
        return iter(self._chunks)


class _FakeBaseURL:
    def __init__(self, url: str) -> None:
        self._uri_reference = url


def _chat_completion_payload(
    response_id: str,
    model: str,
    content: str = "ok",
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    }


def _make_fake_openai_chat_client(
    captured: dict[str, Any],
    response_id: str,
):
    class FakeChatCompletions:
        def __init__(self) -> None:
            self.with_raw_response = self

        def create(self, **data: Any) -> _FakeOpenAIRawResponse:
            captured["body"] = {k: v for k, v in data.items() if k != "timeout"}
            return _FakeOpenAIRawResponse(
                _chat_completion_payload(response_id, captured["body"]["model"])
            )

    class FakeChatNamespace:
        def __init__(self) -> None:
            self.completions = FakeChatCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured["api_key"] = kwargs["api_key"]
            captured["base_url"] = kwargs["base_url"]
            self.api_key = kwargs["api_key"]
            self._base_url = _FakeBaseURL(kwargs["base_url"])
            self.chat = FakeChatNamespace()

    return FakeOpenAI


def _make_fake_openai_streaming_chat_client(
    captured: dict[str, Any],
    chunks: list[Any],
):
    class FakeChatCompletions:
        def __init__(self) -> None:
            self.with_raw_response = self

        def create(self, **data: Any) -> _FakeOpenAIStreamRawResponse:
            captured["body"] = {k: v for k, v in data.items() if k != "timeout"}
            return _FakeOpenAIStreamRawResponse(chunks)

    class FakeChatNamespace:
        def __init__(self) -> None:
            self.completions = FakeChatCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured["api_key"] = kwargs["api_key"]
            captured["base_url"] = kwargs["base_url"]
            self.api_key = kwargs["api_key"]
            self._base_url = _FakeBaseURL(kwargs["base_url"])
            self.chat = FakeChatNamespace()

    return FakeOpenAI


def _openai_stream_chunk(delta: dict[str, Any], finish_reason: str | None = None):
    from openai.types.chat.chat_completion_chunk import (
        ChatCompletionChunk,
        Choice,
        ChoiceDelta,
    )

    return ChatCompletionChunk(
        id="chatcmpl-tensormesh-stream-test",
        created=1,
        model="MiniMaxAI/MiniMax-M2.7",
        object="chat.completion.chunk",
        choices=[
            Choice(
                delta=ChoiceDelta(**delta),
                finish_reason=finish_reason,
                index=0,
            )
        ],
    )
