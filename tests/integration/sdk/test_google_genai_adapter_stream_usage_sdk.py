import asyncio
import json
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Final

import pytest
from integration._support.client import eventually, held
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.google_genai import agenerate_content, agenerate_content_stream, generate_content
from litellm.integrations.custom_logger import CustomLogger

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_PROVIDER_KEY: Final = "synthetic-provider-key"
_REPLY_TEXT: Final = "sdk reply"
_PROMPT_TOKENS: Final = 13
_COMPLETION_TOKENS: Final = 4
_CACHED_TOKENS: Final = 5
_USAGE: Final = {
    "prompt_tokens": _PROMPT_TOKENS,
    "completion_tokens": _COMPLETION_TOKENS,
    "total_tokens": _PROMPT_TOKENS + _COMPLETION_TOKENS,
    "prompt_tokens_details": {"cached_tokens": _CACHED_TOKENS},
}
_GEMINI_USAGE: Final = {
    "promptTokenCount": _PROMPT_TOKENS,
    "candidatesTokenCount": _COMPLETION_TOKENS,
    "totalTokenCount": _PROMPT_TOKENS + _COMPLETION_TOKENS,
}
_REJECTION: Final = "scripted key rejected"
_PROJECT: Final = "scripted-project"
_TOKEN: Final = json.dumps({"access_token": "scripted-token", "token_type": "Bearer", "expires_in": 3600}).encode()


def _frame(chunk: Mapping[str, JsonValue]) -> bytes:
    body: Final = {
        "id": "chatcmpl-sdk",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "m",
        **chunk,
    }
    return f"data: {json.dumps(body)}\n\n".encode()


def _frames(include_usage: bool) -> tuple[bytes, ...]:
    usage: Final = (_frame({"choices": [], "usage": _USAGE}),) if include_usage else ()
    return (
        _frame({"choices": [{"index": 0, "delta": {"role": "assistant", "content": _REPLY_TEXT}}]}),
        b"\n\n",
        _frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        b"\n\n\n\n",
        *usage,
        b"data: [DONE]\n\n",
    )


def _asks_for_usage(body: Mapping[str, JsonValue]) -> bool:
    options: Final = body.get("stream_options")
    return isinstance(options, dict) and options.get("include_usage") is True


def _respond(status: int) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.target.endswith("/_oauth/token"):
            return Reply(body=_TOKEN)
        if status != 200:
            return Reply(status=status, body=json.dumps({"error": {"message": _REJECTION, "code": status}}).encode())
        body: Final = _JSON_OBJECT.validate_json(request.body)
        return Reply(
            content_type="text/event-stream", chunks=_frames(_asks_for_usage(body) or "/ml/v1/" in request.target)
        )

    return respond


@pytest.fixture
def wire() -> Iterator[Wire]:
    with wire_server(_respond(200)) as served:
        yield served


class _FailureCounter(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self._lock: Final = threading.Lock()
        self._failures: Final[list[str]] = []

    def log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        with self._lock:
            self._failures.append("sync")

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        with self._lock:
            self._failures.append("async")

    def failures(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._failures)


def _logged_once(counter: _FailureCounter, kind: str) -> tuple[str, ...]:
    eventually(counter.failures, lambda seen: kind in seen, seconds=10)
    return held(counter.failures, lambda seen: seen.count(kind) == 1, holding=2)


def _contents() -> list[JsonValue]:
    return [{"role": "user", "parts": [{"text": "sdk adapter probe"}]}]


def _stop_usage(chunks: Sequence[Mapping[str, JsonValue]]) -> list[JsonValue]:
    return [
        chunk.get("usageMetadata")
        for chunk in chunks
        if any(isinstance(c, dict) and c.get("finishReason") for c in chunk.get("candidates") or [])
    ]


def _model_requests(wire: Wire) -> list[Request]:
    return [request for request in wire.drain() if not request.target.endswith("/_oauth/token")]


def _upstream_bodies(wire: Wire) -> list[dict[str, JsonValue]]:
    return [_JSON_OBJECT.validate_json(request.body) for request in _model_requests(wire)]


def _member(name: str, wire: Wire) -> dict[str, JsonValue]:
    match name:
        case "vertex_model_garden":
            return {
                "model": "vertex_ai/xai/grok-4.20-non-reasoning",
                "api_base": wire.url,
                "vertex_project": _PROJECT,
                "vertex_location": "global",
                "vertex_credentials": service_account_json(_PROJECT, wire.url),
            }
        case "watsonx":
            return {
                "model": "watsonx/ibm/granite-3-8b-instruct",
                "api_base": wire.url,
                "zen_api_key": "synthetic-zen-key",
                "project_id": _PROJECT,
            }
        case "maritalk":
            return {"model": "maritalk/sabia-3", "api_base": wire.url, "api_key": _PROVIDER_KEY}
        case _:
            return {"model": "amazon_nova/nova-pro-v1", "api_base": wire.url, "api_key": _PROVIDER_KEY}


def _chat_usage(chunks: Sequence[litellm.ModelResponseStream]) -> list[litellm.Usage]:
    return [usage for chunk in chunks if isinstance(usage := getattr(chunk, "usage", None), litellm.Usage)]


@pytest.mark.parametrize("member", ["vertex_model_garden", "watsonx", "maritalk", "amazon_nova"])
def test_sync_completion_stream_keeps_usage_sent_after_finish(wire: Wire, member: str) -> None:
    stream: Final = litellm.completion(
        messages=[{"role": "user", "content": "sync iterator"}],
        stream=True,
        stream_options={"include_usage": True},
        num_retries=0,
        **_member(member, wire),
    )
    chunks: Final = tuple(stream)
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == _REPLY_TEXT
    usage: Final = _chat_usage(chunks)
    assert [(u.prompt_tokens, u.completion_tokens) for u in usage] == [(_PROMPT_TOKENS, _COMPLETION_TOKENS)], chunks
    assert len(_model_requests(wire)) == 1


@pytest.mark.parametrize("member", ["vertex_model_garden", "watsonx"])
async def test_async_completion_stream_keeps_usage_sent_after_finish(wire: Wire, member: str) -> None:
    stream: Final = await litellm.acompletion(
        messages=[{"role": "user", "content": "async iterator"}],
        stream=True,
        stream_options={"include_usage": True},
        num_retries=0,
        **_member(member, wire),
    )
    chunks: Final = tuple([chunk async for chunk in stream])
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == _REPLY_TEXT
    usage: Final = _chat_usage(chunks)
    assert [(u.prompt_tokens, u.completion_tokens) for u in usage] == [(_PROMPT_TOKENS, _COMPLETION_TOKENS)], chunks
    assert len(_model_requests(wire)) == 1


def _sse_chunks(frames: Sequence[bytes]) -> list[dict[str, JsonValue]]:
    parts: Final = [part for part in b"".join(frames).decode().split("\n\n") if part.strip()]
    assert all(part.startswith("data: ") for part in parts), parts
    return [_JSON_OBJECT.validate_json(part.removeprefix("data: ")) for part in parts]


@pytest.mark.parametrize(
    ("stream_options", "asks"),
    [
        pytest.param(None, True, id="caller_sends_none"),
        pytest.param({"include_usage": True}, True, id="caller_opts_in"),
        pytest.param({"include_usage": False}, False, id="caller_opts_out"),
    ],
)
async def test_async_generate_content_stream_adapter_usage_follows_the_caller(
    wire: Wire, stream_options: JsonValue, asks: bool
) -> None:
    options: Final = {} if stream_options is None else {"stream_options": stream_options}
    stream: Final = await agenerate_content_stream(
        model="openai/gpt-5.4-mini", contents=_contents(), api_base=wire.url, api_key=_PROVIDER_KEY, **options
    )
    frames: Final = tuple([frame async for frame in stream])
    chunks: Final = _sse_chunks(frames)
    zero: Final = {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0}
    assert _stop_usage(chunks) == [_GEMINI_USAGE if asks else zero], chunks
    bodies: Final = _upstream_bodies(wire)
    assert len(bodies) == 1 and bodies[0].get("stream_options") == ({"include_usage": True} if asks else None), bodies


def test_sync_generate_content_adapter_error_keeps_the_provider_status_and_logs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter: Final = _FailureCounter()
    monkeypatch.setattr(litellm, "callbacks", [counter])
    with wire_server(_respond(401)) as rejecting:
        with pytest.raises(litellm.AuthenticationError) as raised:
            generate_content(
                model="openai/gpt-5.4-mini", contents=_contents(), api_base=rejecting.url, api_key=_PROVIDER_KEY
            )
        assert raised.value.status_code == 401
        assert _REJECTION in str(raised.value)
        assert len(rejecting.drain()) == 1
    assert _logged_once(counter, "sync").count("sync") == 1


async def test_async_generate_content_adapter_error_keeps_the_provider_status_and_logs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter: Final = _FailureCounter()
    monkeypatch.setattr(litellm, "callbacks", [counter])
    with wire_server(_respond(401)) as rejecting:
        with pytest.raises(litellm.AuthenticationError) as raised:
            await agenerate_content(
                model="openai/gpt-5.4-mini", contents=_contents(), api_base=rejecting.url, api_key=_PROVIDER_KEY
            )
        assert raised.value.status_code == 401
        assert _REJECTION in str(raised.value)
        assert len(rejecting.drain()) == 1
    logged: Final = await asyncio.to_thread(_logged_once, counter, "async")
    assert logged.count("async") == 1, logged
