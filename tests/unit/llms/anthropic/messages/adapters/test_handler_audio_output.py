from collections.abc import Mapping
from typing import Final
from unittest.mock import patch

import httpx
import pytest
import respx
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm.anthropic_interface import messages as anthropic_messages_api
from litellm.router import Router
from tests.unit.proxy.conftest import httpx_transport

pytestmark: Final = pytest.mark.usefixtures(httpx_transport.__name__)

_ANY_GEMINI_URL: Final = r"https://generativelanguage\.googleapis\.com/.*"
_VOICE: Final = {"voice": "Kore", "format": "pcm16"}
_MESSAGES: Final = [{"role": "user", "content": "Say hi"}]
_TEXT_REPLY: Final = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}, "finishReason": "STOP", "index": 0}],
    "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1, "totalTokenCount": 3},
}
_REFUSED_REQUESTS: Final = pytest.mark.parametrize(
    ("model", "request_options"),
    [
        ("gemini/gemini-3.8-flash-tts", {}),
        ("gemini/gemini-3.8-flash-tts", {"drop_params": True}),
        ("gemini/gemini-3.8-flash", {"audio": _VOICE}),
        ("gemini/gemini-3.8-flash", {"modalities": ["text", "audio"]}),
    ],
    ids=["tts", "tts-under-drop-params", "audio-param", "audio-modality"],
)


class _GenerateContentBody(TypedDict):
    generationConfig: ReadOnly[Mapping[str, object]]


_GENERATE_CONTENT_BODY: Final = TypeAdapter(_GenerateContentBody)


def _router(model: str) -> Router:
    return Router(model_list=[{"model_name": "speaker", "litellm_params": {"model": model, "api_key": "test-key"}}])


def _sent_generation_config(route: respx.Route) -> Mapping[str, object]:
    return _GENERATE_CONTENT_BODY.validate_json(route.calls.last.request.content)["generationConfig"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@_REFUSED_REQUESTS
async def test_an_audio_output_request_is_refused_before_gemini_is_called(
    stream: bool, model: str, request_options: Mapping[str, object]
) -> None:
    with patch.object(litellm, "drop_params", False), respx.mock(assert_all_called=False) as mock_router:
        route: Final = mock_router.post(url__regex=_ANY_GEMINI_URL).mock(
            return_value=httpx.Response(status_code=200, json=_TEXT_REPLY)
        )
        with pytest.raises(litellm.UnsupportedParamsError) as error:
            await _router(model).aanthropic_messages(
                model="speaker", max_tokens=64, messages=_MESSAGES, stream=stream, **request_options
            )

    assert error.value.status_code == 400
    assert "/v1/messages" in error.value.message
    assert not route.called


@_REFUSED_REQUESTS
def test_a_sync_audio_output_request_is_refused_before_gemini_is_called(
    model: str, request_options: Mapping[str, object]
) -> None:
    with patch.object(litellm, "drop_params", False), respx.mock(assert_all_called=False) as mock_router:
        route: Final = mock_router.post(url__regex=_ANY_GEMINI_URL).mock(
            return_value=httpx.Response(status_code=200, json=_TEXT_REPLY)
        )
        with pytest.raises(litellm.UnsupportedParamsError) as error:
            anthropic_messages_api.create(
                model=model, max_tokens=64, messages=_MESSAGES, api_key="test-key", **request_options
            )

    assert error.value.status_code == 400
    assert "/v1/messages" in error.value.message
    assert not route.called


@pytest.mark.asyncio
async def test_drop_params_sends_a_text_gemini_model_no_audio_output_ask() -> None:
    with respx.mock() as mock_router:
        route: Final = mock_router.post(url__regex=_ANY_GEMINI_URL).mock(
            return_value=httpx.Response(status_code=200, json=_TEXT_REPLY)
        )
        response: Final = await _router("gemini/gemini-3.8-flash").aanthropic_messages(
            model="speaker",
            max_tokens=64,
            messages=_MESSAGES,
            audio=_VOICE,
            modalities=["text", "audio"],
            drop_params=True,
        )

    assert isinstance(response, dict)
    assert response["content"] == [{"type": "text", "text": "hi"}]
    generation_config: Final = _sent_generation_config(route)
    assert "speechConfig" not in generation_config
    assert generation_config["responseModalities"] == ["TEXT"]


def test_sync_drop_params_sends_a_text_gemini_model_no_audio_output_ask() -> None:
    with respx.mock() as mock_router:
        route: Final = mock_router.post(url__regex=_ANY_GEMINI_URL).mock(
            return_value=httpx.Response(status_code=200, json=_TEXT_REPLY)
        )
        response: Final = anthropic_messages_api.create(
            model="gemini/gemini-3.8-flash",
            max_tokens=64,
            messages=_MESSAGES,
            api_key="test-key",
            audio=_VOICE,
            modalities=["text", "audio"],
            drop_params=True,
        )

    assert isinstance(response, dict)
    assert response["content"] == [{"type": "text", "text": "hi"}]
    generation_config: Final = _sent_generation_config(route)
    assert "speechConfig" not in generation_config
    assert generation_config["responseModalities"] == ["TEXT"]
