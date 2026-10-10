import json
from datetime import datetime, timezone
from typing import Final

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.alibaba_token_plan.common_utils import SPEECH_PATH
from litellm.llms.alibaba_token_plan.text_to_speech.transformation import AlibabaTokenPlanTextToSpeechConfig
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx import http_handler as http_handler_module
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler, blocked_cookie_jar

MODEL: Final = "qwen-audio-3.0-tts-plus"
GATEWAY: Final = "https://gateway.example/token-plan"
SIGNED_AUDIO_URL: Final = "https://audio.example/speech.wav?OSSAccessKeyId=temporary&Signature=secret-value"


def _injected_async_handler(client: httpx.AsyncClient) -> AsyncHTTPHandler:
    handler: Final = AsyncHTTPHandler()
    handler.client = client
    return handler


def _use_download_clients(
    monkeypatch: pytest.MonkeyPatch, sync_client: httpx.Client, async_client: httpx.AsyncClient
) -> None:
    monkeypatch.setattr(litellm, "module_level_client", HTTPHandler(client=sync_client))
    monkeypatch.setattr(
        http_handler_module,
        "get_async_httpx_client",
        lambda llm_provider, params=None, shared_session=None: _injected_async_handler(async_client),
    )


def _speech_provider(download_status: int) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            assert request.url.host in ("audio.example", "93.184.216.34")
            assert "authorization" not in request.headers
            return httpx.Response(download_status, content=b"RIFF-audio", headers={"Content-Type": "audio/wav"})
        assert str(request.url) == f"{GATEWAY}/{SPEECH_PATH}"
        assert request.headers["Authorization"] == "Bearer explicit-key"
        assert json.loads(request.content) == {
            "model": MODEL,
            "input": {"text": "Hello", "voice": "longanhuan_v3.6", "format": "wav", "sample_rate": 16000},
        }
        return httpx.Response(200, json={"output": {"audio": {"url": SIGNED_AUDIO_URL}}})

    return httpx.MockTransport(respond)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_speech_downloads_the_audio_url_without_provider_credentials(
    use_async: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport: Final = _speech_provider(200)
    params: Final = {
        "model": f"alibaba_token_plan/{MODEL}",
        "input": "Hello",
        "voice": "alloy",
        "response_format": "wav",
        "sample_rate": 16000,
        "api_key": "explicit-key",
        "api_base": f"{GATEWAY}/compatible-mode/v1",
        "max_retries": 0,
    }
    async with httpx.AsyncClient(transport=transport, cookies=blocked_cookie_jar()) as async_download:
        with (
            httpx.Client(transport=transport, cookies=blocked_cookie_jar()) as sync_download,
            httpx.Client(transport=transport) as client,
        ):
            _use_download_clients(monkeypatch, sync_download, async_download)
            async with httpx.AsyncClient(transport=transport) as async_client:
                response: Final = (
                    await litellm.aspeech(**params, client=_injected_async_handler(async_client))
                    if use_async
                    else litellm.speech(**params, client=HTTPHandler(client=client))
                )
    assert response.content == b"RIFF-audio"
    assert response.response.headers["Content-Type"] == "audio/wav"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_audio_download_failure_does_not_expose_the_signed_url(
    use_async: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport: Final = _speech_provider(403)
    params: Final = {
        "model": f"alibaba_token_plan/{MODEL}",
        "input": "Hello",
        "response_format": "wav",
        "sample_rate": 16000,
        "api_key": "explicit-key",
        "api_base": f"{GATEWAY}/compatible-mode/v1",
        "max_retries": 0,
    }
    async with httpx.AsyncClient(transport=transport, cookies=blocked_cookie_jar()) as async_download:
        with (
            httpx.Client(transport=transport, cookies=blocked_cookie_jar()) as sync_download,
            httpx.Client(transport=transport) as client,
        ):
            _use_download_clients(monkeypatch, sync_download, async_download)
            async with httpx.AsyncClient(transport=transport) as async_client:

                async def request_speech() -> object:
                    if use_async:
                        return await litellm.aspeech(**params, client=_injected_async_handler(async_client))
                    return litellm.speech(**params, client=HTTPHandler(client=client))

                with pytest.raises(Exception, match="audio download failed") as error:
                    await request_speech()
    assert "secret-value" not in str(error.value)
    assert "Signature" not in str(error.value)


def test_voice_defaults_and_unsupported_parameters() -> None:
    config: Final = AlibabaTokenPlanTextToSpeechConfig()
    assert config.transform_text_to_speech_request(MODEL, "Hi", None, {}, {}, {}) == {
        "dict_body": {"model": MODEL, "input": {"text": "Hi", "voice": "longanhuan_v3.6", "format": "mp3"}}
    }
    assert config.map_openai_params(MODEL, {}, voice=None) == ("longanhuan_v3.6", {})
    assert config.map_openai_params(MODEL, {}, voice="longanlingxin") == ("longanlingxin", {})
    with pytest.raises(litellm.UnsupportedParamsError, match="speed"):
        config.map_openai_params(MODEL, {"speed": 1.2}, voice="alloy")
    assert config.map_openai_params(MODEL, {"speed": 1.2}, voice="alloy", drop_params=True) == ("longanhuan_v3.6", {})


@pytest.mark.parametrize(
    ("status_code", "expected_status"),
    [(401, 401), (200, 502)],
)
def test_error_bodies_are_never_returned_as_audio(status_code: int, expected_status: int) -> None:
    logging_obj: Final = Logging(
        model=MODEL,
        messages=[],
        stream=False,
        call_type="speech",
        start_time=datetime(2026, 10, 2, tzinfo=timezone.utc),
        litellm_call_id="test-call",
        function_id="test-function",
    )
    with pytest.raises(BaseLLMException, match="Invalid voice") as error:
        AlibabaTokenPlanTextToSpeechConfig().transform_text_to_speech_response(
            MODEL,
            httpx.Response(status_code, json={"code": "InvalidParameter", "message": "Invalid voice"}),
            logging_obj,
        )
    assert error.value.status_code == expected_status
