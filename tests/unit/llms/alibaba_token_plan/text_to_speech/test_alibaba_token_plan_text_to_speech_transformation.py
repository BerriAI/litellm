import base64
import json
from datetime import datetime, timezone
from functools import partial
from typing import Final

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.url_utils import SSRFError
from litellm.llms.alibaba_token_plan.common_utils import SPEECH_ENDPOINT
from litellm.llms.alibaba_token_plan.text_to_speech.transformation import AlibabaTokenPlanTextToSpeechConfig
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx import http_handler as http_handler_module
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler, blocked_cookie_jar
from litellm.types.llms.openai import HttpxBinaryResponseContent


def _logging_obj() -> Logging:
    return Logging(
        model="qwen-audio-3.0-tts-plus",
        messages=[],
        stream=False,
        call_type="speech",
        start_time=datetime(2026, 10, 2, tzinfo=timezone.utc),
        litellm_call_id="test-call",
        function_id="test-function",
    )


class _InjectedAsyncHTTPHandler(AsyncHTTPHandler):
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client


def _set_sync_download_client(monkeypatch: pytest.MonkeyPatch, client: httpx.Client) -> None:
    handler: Final = HTTPHandler(client=client)
    monkeypatch.setattr(litellm, "module_level_client", handler)


def _set_async_download_client(monkeypatch: pytest.MonkeyPatch, client: httpx.AsyncClient) -> None:
    handler: Final = _InjectedAsyncHTTPHandler(client)
    monkeypatch.setattr(
        http_handler_module,
        "get_async_httpx_client",
        lambda llm_provider, params=None, shared_session=None: handler,
    )


def _speech_response(request: httpx.Request, response_mode: str) -> httpx.Response:
    if request.method == "GET":
        assert str(request.url) == "http://93.184.216.34/speech.wav?signature=secret"
        assert request.headers["Host"] == "audio.example"
        assert "authorization" not in request.headers
        assert "x-api-key" not in request.headers
        return httpx.Response(200, content=b"RIFF-audio", headers={"Content-Type": "audio/wav"})
    assert request.method == "POST"
    assert str(request.url) == f"https://gateway.example/token-plan/{SPEECH_ENDPOINT}"
    assert request.headers["Authorization"] == "Bearer explicit-key"
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(request.content) == {
        "model": "qwen-audio-3.0-tts-plus",
        "input": {"text": "Hello", "voice": "longanhuan_v3.6", "format": "wav", "sample_rate": 16000},
    }
    if response_mode == "url":
        return httpx.Response(
            200,
            json={"output": {"audio": {"data": "", "url": "http://audio.example/speech.wav?signature=secret"}}},
        )
    if response_mode == "base64":
        return httpx.Response(200, json={"output": {"audio": {"data": base64.b64encode(b"RIFF-audio").decode()}}})
    return httpx.Response(200, content=b"RIFF-audio", headers={"Content-Type": "audio/wav"})


@pytest.mark.parametrize("response_mode", ["binary", "url", "base64"])
@pytest.mark.parametrize("path", [SPEECH_ENDPOINT, "compatible-mode/v1", "apps/anthropic"])
def test_sync_speech_request_and_binary_response_through_public_api(
    response_mode: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport: Final = httpx.MockTransport(partial(_speech_response, response_mode=response_mode))
    with (
        httpx.Client(transport=transport) as client,
        httpx.Client(transport=transport, cookies=blocked_cookie_jar()) as download_client,
    ):
        _set_sync_download_client(monkeypatch, download_client)
        response: Final = litellm.speech(
            model="alibaba_token_plan/qwen-audio-3.0-tts-plus",
            input="Hello",
            response_format="wav",
            sample_rate=16000,
            api_key="explicit-key",
            api_base=f"https://gateway.example/token-plan/{path}",
            timeout=10,
            max_retries=0,
            client=HTTPHandler(client=client),
        )
    assert isinstance(response, HttpxBinaryResponseContent)
    assert response.content == b"RIFF-audio"
    assert response.response.headers["Content-Type"] == (
        "application/octet-stream" if response_mode == "base64" else "audio/wav"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("response_mode", ["binary", "url", "base64"])
async def test_async_speech_request_and_binary_response_through_public_api(
    response_mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport: Final = httpx.MockTransport(partial(_speech_response, response_mode=response_mode))
    async with (
        httpx.AsyncClient(transport=transport) as client,
        httpx.AsyncClient(transport=transport, cookies=blocked_cookie_jar()) as download_client,
    ):
        _set_async_download_client(monkeypatch, download_client)
        response: Final = await litellm.aspeech(
            model="alibaba_token_plan/qwen-audio-3.0-tts-plus",
            input="Hello",
            response_format="wav",
            sample_rate=16000,
            api_key="explicit-key",
            api_base=f"https://gateway.example/token-plan/{SPEECH_ENDPOINT}",
            timeout=10,
            max_retries=0,
            client=_InjectedAsyncHTTPHandler(client),
        )
    assert response.content == b"RIFF-audio"
    assert response.response.headers["Content-Type"] == (
        "application/octet-stream" if response_mode == "base64" else "audio/wav"
    )


def _credential_isolation_response(request: httpx.Request) -> httpx.Response:
    if request.method == "POST":
        assert request.headers["Authorization"]
        assert request.headers["X-Api-Key"] == "header-api-key"
        assert request.headers["Cookie"] == "header-cookie=secret"
        assert request.url.params["api_key"] == "query-secret"
        return httpx.Response(200, json={"output": {"audio": {"url": "https://audio.example/start?signature=signed"}}})
    assert request.method == "GET"
    assert "authorization" not in request.headers
    assert "x-api-key" not in request.headers
    assert "x-custom-secret" not in request.headers
    assert "cookie" not in request.headers
    assert "api_key" not in request.url.params
    assert request.url.params["signature"] == "signed"
    if request.url.path == "/start":
        return httpx.Response(
            302,
            headers={
                "Location": "https://redirected.audio.example/final?signature=signed",
                "Set-Cookie": "redirect-cookie=secret; Domain=.audio.example; Path=/",
            },
        )
    assert request.url.host == "redirected.audio.example"
    assert request.url.path == "/final"
    return httpx.Response(200, content=b"private-client-public-audio", headers={"Content-Type": "audio/mpeg"})


@pytest.mark.parametrize("use_auth", [False, True])
@pytest.mark.parametrize("validate_urls", [False, True])
def test_sync_audio_download_does_not_inherit_client_credentials_on_redirects(
    use_auth: bool, validate_urls: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", validate_urls)
    transport: Final = httpx.MockTransport(_credential_isolation_response)
    with (
        httpx.Client(
            transport=transport,
            headers={
                "Authorization": "Bearer header-secret",
                "X-Api-Key": "header-api-key",
                "X-Custom-Secret": "custom-header-secret",
                "Cookie": "header-cookie=secret",
            },
            params={"api_key": "query-secret"},
            cookies={"jar-cookie": "cookie-secret"},
            auth=httpx.BasicAuth("client-user", "client-password") if use_auth else None,
            follow_redirects=True,
        ) as client,
        httpx.Client(transport=transport, cookies=blocked_cookie_jar()) as download_client,
    ):
        _set_sync_download_client(monkeypatch, download_client)
        response: Final = litellm.speech(
            model="alibaba_token_plan/qwen-audio-3.0-tts-plus",
            input="Hello",
            api_key="explicit-key",
            client=HTTPHandler(client=client),
            timeout=7,
            max_retries=0,
        )
        assert response.content == b"private-client-public-audio"
        assert client.headers["Authorization"] == "Bearer header-secret"
        assert client.cookies["jar-cookie"] == "cookie-secret"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_auth", [False, True])
@pytest.mark.parametrize("validate_urls", [False, True])
async def test_async_audio_download_does_not_inherit_client_credentials_on_redirects(
    use_auth: bool, validate_urls: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", validate_urls)
    transport: Final = httpx.MockTransport(_credential_isolation_response)
    async with (
        httpx.AsyncClient(
            transport=transport,
            headers={
                "Authorization": "Bearer header-secret",
                "X-Api-Key": "header-api-key",
                "X-Custom-Secret": "custom-header-secret",
                "Cookie": "header-cookie=secret",
            },
            params={"api_key": "query-secret"},
            cookies={"jar-cookie": "cookie-secret"},
            auth=httpx.BasicAuth("client-user", "client-password") if use_auth else None,
            follow_redirects=True,
        ) as client,
        httpx.AsyncClient(transport=transport, cookies=blocked_cookie_jar()) as download_client,
    ):
        _set_async_download_client(monkeypatch, download_client)
        response: Final = await litellm.aspeech(
            model="alibaba_token_plan/qwen-audio-3.0-tts-plus",
            input="Hello",
            api_key="explicit-key",
            client=_InjectedAsyncHTTPHandler(client),
            timeout=7,
            max_retries=0,
        )
        assert response.content == b"private-client-public-audio"
        assert client.headers["Authorization"] == "Bearer header-secret"
        assert client.cookies["jar-cookie"] == "cookie-secret"


def test_audio_download_stops_at_safe_redirect_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(302, headers={"Location": "https://audio.example/loop"})
        return httpx.Response(200, json={"output": {"audio": {"url": "https://audio.example/loop"}}})

    transport: Final = httpx.MockTransport(respond)
    with (
        httpx.Client(transport=transport) as client,
        httpx.Client(transport=transport, cookies=blocked_cookie_jar()) as download_client,
    ):
        _set_sync_download_client(monkeypatch, download_client)
        with pytest.raises(SSRFError, match="Too many redirects"):
            litellm.speech(
                model="alibaba_token_plan/qwen-audio-3.0-tts-plus",
                input="Hello",
                api_key="explicit-key",
                client=HTTPHandler(client=client),
                max_retries=0,
            )


@pytest.mark.parametrize(
    "body", ["not json", "{}", '{"output":{"audio":{}}}', '{"output":{"audio":{"data":"invalid"}}}']
)
def test_malformed_speech_response_is_rejected(body: str) -> None:
    with pytest.raises(BaseLLMException) as error:
        AlibabaTokenPlanTextToSpeechConfig().transform_text_to_speech_response(
            "qwen-audio-3.0-tts-plus",
            httpx.Response(200, content=body, headers={"Content-Type": "application/json"}),
            _logging_obj(),
        )
    assert error.value.status_code == 502


def test_malformed_speech_url_is_not_exposed() -> None:
    secret: Final = "sensitive-presigned-value"
    with pytest.raises(BaseLLMException, match="Invalid Alibaba Token Plan speech response") as error:
        AlibabaTokenPlanTextToSpeechConfig().transform_text_to_speech_response(
            "qwen-audio-3.0-tts-plus",
            httpx.Response(
                200,
                json={"output": {"audio": {"url": f"not-a-url?Signature={secret}"}}},
            ),
            _logging_obj(),
        )
    assert secret not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_audio_download_failure_does_not_expose_signed_url(
    use_async: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(403, text="Expired audio URL")
        return httpx.Response(
            200,
            json={
                "output": {
                    "audio": {"url": "https://audio.example/expired?OSSAccessKeyId=temporary&Signature=secret-value"}
                }
            },
        )

    transport: Final = httpx.MockTransport(respond)
    if use_async:
        async with (
            httpx.AsyncClient(transport=transport) as client,
            httpx.AsyncClient(transport=transport, cookies=blocked_cookie_jar()) as download_client,
        ):
            _set_async_download_client(monkeypatch, download_client)
            with pytest.raises(litellm.APIError, match="audio download failed") as error:
                await litellm.aspeech(
                    model="alibaba_token_plan/qwen-audio-3.0-tts-plus",
                    input="Hello",
                    api_key="explicit-key",
                    client=_InjectedAsyncHTTPHandler(client),
                    max_retries=0,
                )
        assert "secret-value" not in str(error.value)
        assert "Signature" not in str(error.value)
        return
    with (
        httpx.Client(transport=transport) as client,
        httpx.Client(transport=transport, cookies=blocked_cookie_jar()) as download_client,
    ):
        _set_sync_download_client(monkeypatch, download_client)
        with pytest.raises(BaseLLMException, match="audio download failed") as error:
            litellm.speech(
                model="alibaba_token_plan/qwen-audio-3.0-tts-plus",
                input="Hello",
                api_key="explicit-key",
                client=HTTPHandler(client=client),
                max_retries=0,
            )
    assert "secret-value" not in str(error.value)
    assert "Signature" not in str(error.value)


def test_native_voice_and_speech_defaults() -> None:
    config: Final = AlibabaTokenPlanTextToSpeechConfig()
    voice, params = config.map_openai_params("qwen-audio-3.0-tts-plus", {}, voice="longanhuan_v3.6")
    default_voice, _ = config.map_openai_params("qwen-audio-3.0-tts-plus", {}, voice="alloy")
    assert default_voice == "longanhuan_v3.6"
    assert config.transform_text_to_speech_request("qwen-audio-3.0-tts-plus", "Hello", voice, params, {}, {}) == {
        "dict_body": {
            "model": "qwen-audio-3.0-tts-plus",
            "input": {"text": "Hello", "voice": "longanhuan_v3.6", "format": "mp3", "sample_rate": 24000},
        }
    }


@pytest.mark.parametrize("parameter", ["speed", "instructions"])
def test_unsupported_speech_parameters_require_explicit_drop(parameter: str) -> None:
    config: Final = AlibabaTokenPlanTextToSpeechConfig()
    with pytest.raises(litellm.UnsupportedParamsError, match=parameter):
        config.map_openai_params("qwen-audio-3.0-tts-plus", {parameter: 1})
    assert config.map_openai_params("qwen-audio-3.0-tts-plus", {parameter: 1}, drop_params=True) == (None, {})


@pytest.mark.parametrize("status_code", [401, 429, 200])
def test_error_json_cannot_be_returned_as_audio(status_code: int) -> None:
    with pytest.raises(BaseLLMException, match="Invalid voice") as error:
        AlibabaTokenPlanTextToSpeechConfig().transform_text_to_speech_response(
            "qwen-audio-3.0-tts-plus",
            httpx.Response(status_code, json={"code": "InvalidParameter", "message": "Invalid voice"}),
            _logging_obj(),
        )
    assert error.value.status_code == (502 if status_code == 200 else status_code)


def _blocked_download_response(request: httpx.Request, target: str, redirect: bool) -> httpx.Response:
    if request.method == "POST":
        return httpx.Response(
            200, json={"output": {"audio": {"url": "https://audio.example/start" if redirect else target}}}
        )
    assert redirect and request.url.host == "audio.example", "Blocked destination reached the transport"
    return httpx.Response(302, headers={"Location": target})


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("redirect", [False, True])
@pytest.mark.parametrize("target", ["http://127.0.0.1/private", "http://169.254.169.254/metadata", "http://[::1]/"])
async def test_audio_download_blocks_private_destinations_before_transport(
    use_async: bool, redirect: bool, target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport: Final = httpx.MockTransport(partial(_blocked_download_response, target=target, redirect=redirect))
    if use_async:
        async with (
            httpx.AsyncClient(transport=transport) as async_client,
            httpx.AsyncClient(transport=transport, cookies=blocked_cookie_jar()) as download_client,
        ):
            _set_async_download_client(monkeypatch, download_client)
            with pytest.raises((SSRFError, litellm.APIConnectionError), match="blocked address"):
                await litellm.aspeech(
                    model="alibaba_token_plan/qwen-audio-3.0-tts-plus",
                    input="Hello",
                    api_key="explicit-key",
                    client=_InjectedAsyncHTTPHandler(async_client),
                    max_retries=0,
                )
        return
    with (
        httpx.Client(transport=transport) as client,
        httpx.Client(transport=transport, cookies=blocked_cookie_jar()) as download_client,
    ):
        _set_sync_download_client(monkeypatch, download_client)
        with pytest.raises(SSRFError, match="blocked address"):
            litellm.speech(
                model="alibaba_token_plan/qwen-audio-3.0-tts-plus",
                input="Hello",
                api_key="explicit-key",
                client=HTTPHandler(client=client),
                max_retries=0,
            )


def _relative_download_response(request: httpx.Request, hostname: str, address: str) -> httpx.Response:
    if request.method == "POST":
        return httpx.Response(200, json={"output": {"audio": {"url": f"http://{hostname}/start?signature=one"}}})
    assert request.url.host == address
    assert request.headers["Host"] == hostname
    assert "authorization" not in request.headers
    assert "cookie" not in request.headers
    if request.url.path == "/start":
        return httpx.Response(302, headers={"Location": "/final?signature=two"})
    assert request.url.path == "/final"
    assert request.url.params == httpx.QueryParams("signature=two")
    return httpx.Response(200, content=b"audio", headers={"Content-Type": "audio/mpeg"})


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("mode", ["public", "allowlisted", "disabled"])
async def test_audio_download_preserves_relative_redirect_origin_and_url_policy(
    use_async: bool, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    hostname: Final = "audio.example" if mode == "public" else "127.0.0.1"
    address: Final = "93.184.216.34" if mode == "public" else "127.0.0.1"
    monkeypatch.setattr(litellm, "user_url_validation", mode != "disabled")
    monkeypatch.setattr(litellm, "user_url_allowed_hosts", [hostname] if mode == "allowlisted" else [])
    transport: Final = httpx.MockTransport(partial(_relative_download_response, hostname=hostname, address=address))
    if use_async:
        async with (
            httpx.AsyncClient(transport=transport) as async_client,
            httpx.AsyncClient(transport=transport, cookies=blocked_cookie_jar()) as download_client,
        ):
            _set_async_download_client(monkeypatch, download_client)
            async_result: Final = await litellm.aspeech(
                model="alibaba_token_plan/qwen-audio-3.0-tts-plus",
                input="Hello",
                api_key="explicit-key",
                client=_InjectedAsyncHTTPHandler(async_client),
                max_retries=0,
            )
            assert async_result.content == b"audio"
        return
    with (
        httpx.Client(transport=transport) as client,
        httpx.Client(transport=transport, cookies=blocked_cookie_jar()) as download_client,
    ):
        _set_sync_download_client(monkeypatch, download_client)
        result: Final = litellm.speech(
            model="alibaba_token_plan/qwen-audio-3.0-tts-plus",
            input="Hello",
            api_key="explicit-key",
            client=HTTPHandler(client=client),
            max_retries=0,
        )
        assert result.content == b"audio"
