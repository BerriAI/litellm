import json
from typing import Final

import httpx
import pytest
import respx

import litellm

_LINE_SEPARATED_TEXT: Final = "before\u2028between\u2029after"


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
    route: Final = respx_mock.post("https://codestral.mistral.ai/v1/fim/completions").mock(
        return_value=httpx.Response(200, content=_fim_stream_body(_LINE_SEPARATED_TEXT))
    )

    texts: Final = await _streamed_texts(sync_stream)

    assert texts == [_LINE_SEPARATED_TEXT, None]
    assert json.loads(route.calls.last.request.content)["stream"] is True
