import json
from typing import Final

import httpx
import pytest
import respx

import litellm

CODESTRAL_FIM_URL: Final = "https://codestral.mistral.ai/v1/fim/completions"

COMPLETION_BODY: Final = {
    "id": "cmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "codestral-latest",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "return 1"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
}

STREAM_CHUNK: Final = {
    "id": "cmpl-1",
    "object": "chat.completion.chunk",
    "created": 1,
    "model": "codestral-latest",
    "choices": [{"index": 0, "delta": {"content": "return 1"}, "finish_reason": "stop"}],
}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
@pytest.mark.respx(assert_all_called=True)
async def test_codestral_text_completion_sends_the_callers_timeout(
    respx_mock: respx.Router, monkeypatch: pytest.MonkeyPatch, sync_mode: bool, stream: bool
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    response: Final = (
        httpx.Response(200, content=f"data: {json.dumps(STREAM_CHUNK)}\n\ndata: [DONE]\n\n")
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
    assert json.loads(route.calls[0].request.content).get("stream", False) is stream
    assert route.calls[0].request.extensions["timeout"] == {"connect": 7.0, "read": 7.0, "write": 7.0, "pool": 7.0}
