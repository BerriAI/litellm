import json
from typing import Final

import httpx
import pytest
import respx

import litellm

CODESTRAL_FIM_URL: Final = "https://codestral.mistral.ai/v1/fim/completions"
_LINE_SEPARATED_TEXT: Final = "before between after"

COMPLETION_BODY: Final = {
    "id": "cmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "codestral-latest",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "return 1"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
}


def _fim_stream_body(text: str) -> bytes:
    chunks: Final = (
        {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "codestral-latest",
         "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]},
        {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "codestral-latest",
         "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}]},
    )
    events: Final = "".join(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in chunks)
    return f"{events}data: [DONE]\n\n".encode()


async def _streamed_texts(sync_stream: bool) -> list[str | None]:
    request: Final = {
        "model": "text-completion-codestral/codestral-latest",
        "prompt": "def f(",
        "stream": True,
        "api_key": "codestral-key",
        "max_retries": 0,
    }
    if sync_stream:
        return [chunk.choices[0].text for chunk in litellm.text_completion(**request)]
    return [chunk.choices[0].text async for chunk in await litellm.atext_completion(**request)]


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_stream", [True, False])
async def test_fim_stream_keeps_unicode_line_separators_inside_a_delta(
    sync_stream: bool, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post(CODESTRAL_FIM_URL).mock(
        return_value=httpx.Response(200, content=_fim_stream_body(_LINE_SEPARATED_TEXT))
    )

    texts: Final = await _streamed_texts(sync_stream)

    assert texts == [_LINE_SEPARATED_TEXT, None]
    assert json.loads(route.calls.last.request.content)["stream"] is True


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_codestral_text_completion_sends_the_callers_timeout(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, sync_mode: bool, stream: bool
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    response: Final = (
        httpx.Response(200, content=_fim_stream_body("return 1"))
        if stream
        else httpx.Response(200, json=COMPLETION_BODY)
    )
    route: Final = respx_mock.post(CODESTRAL_FIM_URL).mock(return_value=response)

    kwargs: Final = {
        "model": "text-completion-codestral/codestral-latest",
        "prompt": "def one():",
        "api_key": "test-codestral-key",
        "timeout": 7.0,
        "stream": stream,
    }
    if sync_mode:
        result = litellm.text_completion(**kwargs)
        text = "".join(chunk.choices[0].text or "" for chunk in result) if stream else result.choices[0].text
    else:
        result = await litellm.atext_completion(**kwargs)
        text = "".join([chunk.choices[0].text or "" async for chunk in result]) if stream else result.choices[0].text

    assert text == "return 1"
    assert route.calls[0].request.extensions["timeout"] == {"connect": 7.0, "read": 7.0, "write": 7.0, "pool": 7.0}
