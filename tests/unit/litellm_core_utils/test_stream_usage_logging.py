import asyncio
import json
from typing import Final, TypedDict

from typing_extensions import ReadOnly

import httpx
import pytest
import respx
from pydantic import TypeAdapter

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.redact_messages import REDACTED_BY_LITELLM
from litellm.types.utils import Usage

_OPENAI_URL: Final = "https://api.openai.com/v1/chat/completions"
_BODY: Final = TypeAdapter(dict[str, object])
_PROMPT_TOKENS: Final = 607
_COMPLETION_TOKENS: Final = 23


def _sse(chunks: tuple[dict[str, object], ...]) -> str:
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


def _chunk(delta: dict[str, str], finish_reason: str | None) -> dict[str, object]:
    return {
        "id": "chatcmpl-usage",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-5.5",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


_STREAM: Final = _sse(
    (
        _chunk({"role": "assistant", "content": "I am"}, None),
        _chunk({"content": " well"}, "stop"),
        {
            "id": "chatcmpl-usage",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-5.5",
            "choices": [],
            "usage": {
                "prompt_tokens": _PROMPT_TOKENS,
                "completion_tokens": _COMPLETION_TOKENS,
                "total_tokens": _PROMPT_TOKENS + _COMPLETION_TOKENS,
            },
        },
    )
)


class _LoggedUsage(TypedDict):
    prompt_tokens: ReadOnly[int]
    completion_tokens: ReadOnly[int]
    total_tokens: ReadOnly[int]
    messages: ReadOnly[object]


class _UsageRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.payloads: Final[list[_LoggedUsage]] = []
        self.logged: Final = asyncio.Event()

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.payloads.append(TypeAdapter(_LoggedUsage).validate_python(kwargs["standard_logging_object"]))
        self.logged.set()


async def _stream_and_record(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, include_usage: bool
) -> tuple[Usage, _LoggedUsage, dict[str, object]]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    recorder: Final = _UsageRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    route: Final = respx_mock.post(_OPENAI_URL).mock(
        return_value=httpx.Response(200, text=_STREAM, headers={"content-type": "text/event-stream"})
    )
    stream_options: Final = {"stream_options": {"include_usage": True}} if include_usage else {}
    response: Final = await litellm.acompletion(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "Hello, how are you?" * 100}],
        stream=True,
        api_key="sk-unit-test",
        **stream_options,
    )
    usages: Final = tuple([chunk.usage async for chunk in response if getattr(chunk, "usage", None) is not None])
    await asyncio.wait_for(recorder.logged.wait(), timeout=10)
    return usages[-1], recorder.payloads[-1], _BODY.validate_json(route.calls.last.request.content)


def _assert_logged_usage_matches(client_usage: Usage, payload: _LoggedUsage) -> None:
    assert client_usage.prompt_tokens == _PROMPT_TOKENS
    assert client_usage.completion_tokens == _COMPLETION_TOKENS
    assert (payload["prompt_tokens"], payload["completion_tokens"], payload["total_tokens"]) == (
        client_usage.prompt_tokens,
        client_usage.completion_tokens,
        client_usage.total_tokens,
    )


@pytest.mark.asyncio
async def test_logged_stream_usage_equals_the_final_chunk_usage_with_include_usage(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    client_usage, payload, body = await _stream_and_record(monkeypatch, respx_mock, include_usage=True)

    assert body["stream_options"] == {"include_usage": True}
    _assert_logged_usage_matches(client_usage, payload)


@pytest.mark.asyncio
async def test_logged_stream_usage_equals_the_usage_chunk_without_stream_options(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    client_usage, payload, body = await _stream_and_record(monkeypatch, respx_mock, include_usage=False)

    assert body["stream_options"] == {"include_usage": True}
    _assert_logged_usage_matches(client_usage, payload)


@pytest.mark.asyncio
async def test_logged_stream_usage_survives_message_redaction(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setattr(litellm, "turn_off_message_logging", True)

    client_usage, payload, _ = await _stream_and_record(monkeypatch, respx_mock, include_usage=False)

    _assert_logged_usage_matches(client_usage, payload)
    assert payload["messages"] == [{"role": "user", "content": REDACTED_BY_LITELLM}]
