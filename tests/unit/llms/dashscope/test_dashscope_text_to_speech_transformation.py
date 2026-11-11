import datetime
import json
from collections.abc import Callable
from typing import Final

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.dashscope.common_utils import DashScopeError
from litellm.llms.dashscope.text_to_speech.transformation import DashScopeTextToSpeechConfig
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.utils import Usage

INTL: Final = "https://dashscope-intl.aliyuncs.com"
MULTIMODAL_PATH: Final = "/api/v1/services/aigc/multimodal-generation/generation"
SYNTHESIZER_PATH: Final = "/api/v1/services/audio/tts/SpeechSynthesizer"
AUDIO_URL: Final = (
    "http://93.184.216.34/prod/qwen-tts/a1.mp3?Expires=1793316671&OSSAccessKeyId=LTAI&Signature=ab%2Bc%3D"
)
AUDIO: Final = b"ID3\x04\x00synthetic-mp3-frames"
SYNTHESIZED: Final = {
    "output": {"audio": {"url": AUDIO_URL, "data": "", "id": "audio_1"}},
    "usage": {"characters": 6, "input_tokens": 8, "output_tokens": 30, "total_tokens": 38},
}

API_REQUEST: Final = httpx.Request("POST", f"{INTL}{MULTIMODAL_PATH}")

Reply = Callable[[httpx.Request], httpx.Response]


def _api(reply: dict[str, object]) -> Reply:  # mutable-ok: JSON body fixture
    return lambda _: httpx.Response(200, json=reply)


def _audio(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=AUDIO, headers={"content-type": "audio/mpeg"})


def _recording(seen: list[httpx.Request], reply: Reply) -> httpx.MockTransport:  # mutable-ok: request log
    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return reply(request)

    return httpx.MockTransport(respond)


def _refusing_blocking_download(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"blocking client used for {request.url} on the event loop")


def _logging() -> Logging:
    return Logging(
        model="dashscope-tts",
        messages=[],
        stream=False,
        call_type="aspeech",
        start_time=datetime.datetime(2026, 10, 28, tzinfo=datetime.timezone.utc),
        litellm_call_id="call-tts",
        function_id="fn-tts",
    )


@pytest.fixture(autouse=True)
def _no_ambient_dashscope_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_BASE", raising=False)


@pytest.mark.parametrize(
    ("model", "optional_params", "path", "provider_input"),
    (
        (
            "qwen3-tts-flash",
            {"response_format": "wav"},
            MULTIMODAL_PATH,
            {"text": "Hello.", "voice": "Cherry"},
        ),
        (
            "qwen-audio-3.1-tts-flash",
            {"response_format": "mp3", "instructions": "Speak slowly."},
            SYNTHESIZER_PATH,
            {"text": "Hello.", "voice": "Cherry", "format": "mp3", "instruction": "Speak slowly."},
        ),
        (
            "qwen-audio-3.0-tts-plus",
            {},
            SYNTHESIZER_PATH,
            {"text": "Hello.", "voice": "Cherry"},
        ),
    ),
)
async def test_each_family_posts_to_its_endpoint_and_the_async_handler_returns_the_downloaded_audio(
    model: str,
    optional_params: dict[str, object],  # mutable-ok: parametrized fixture
    path: str,
    provider_input: dict[str, object],  # mutable-ok: parametrized fixture
) -> None:
    api_seen: Final[list[httpx.Request]] = []  # mutable-ok: request log
    download_seen: Final[list[httpx.Request]] = []  # mutable-ok: request log
    config: Final = DashScopeTextToSpeechConfig(
        sync_client=lambda: HTTPHandler(
            client=httpx.Client(transport=httpx.MockTransport(_refusing_blocking_download))
        ),
        async_client=lambda: AsyncHTTPHandler(transport=_recording(download_seen, _audio)),
    )
    voice, mapped = config.map_openai_params(model=model, optional_params=optional_params, voice="Cherry")

    speech: Final = await BaseLLMHTTPHandler().async_text_to_speech_handler(
        model=model,
        input="Hello.",
        voice=voice,
        text_to_speech_provider_config=config,
        text_to_speech_optional_params=mapped,
        custom_llm_provider="dashscope",
        litellm_params={"api_key": "sk-dashscope-test", "api_base": None},
        logging_obj=_logging(),
        timeout=30,
        client=AsyncHTTPHandler(transport=_recording(api_seen, _api(SYNTHESIZED))),
    )

    assert speech.content == AUDIO
    (api_request,) = api_seen
    assert str(api_request.url) == f"{INTL}{path}"
    assert api_request.headers["authorization"] == "Bearer sk-dashscope-test"
    assert json.loads(api_request.content) == {"model": model, "input": provider_input}
    (download,) = download_seen
    assert str(download.url) == AUDIO_URL
    assert "authorization" not in download.headers
    assert download.extensions["timeout"] == api_request.extensions["timeout"]


def test_the_sync_handler_downloads_through_the_injected_blocking_client_within_the_call_timeout() -> None:
    api_seen: Final[list[httpx.Request]] = []  # mutable-ok: request log
    download_seen: Final[list[httpx.Request]] = []  # mutable-ok: request log
    config: Final = DashScopeTextToSpeechConfig(
        sync_client=lambda: HTTPHandler(client=httpx.Client(transport=_recording(download_seen, _audio)))
    )

    speech: Final = BaseLLMHTTPHandler().text_to_speech_handler(
        model="qwen3-tts-flash",
        input="Hello.",
        voice="Cherry",
        text_to_speech_provider_config=config,
        text_to_speech_optional_params={},
        custom_llm_provider="dashscope",
        litellm_params={"api_key": "k", "api_base": None},
        logging_obj=_logging(),
        timeout=30,
        client=HTTPHandler(client=httpx.Client(transport=_recording(api_seen, _api(SYNTHESIZED)))),
    )

    assert isinstance(speech, HttpxBinaryResponseContent)
    assert speech.content == AUDIO
    (api_request,) = api_seen
    (download,) = download_seen
    assert str(download.url) == AUDIO_URL
    assert download.extensions["timeout"] == api_request.extensions["timeout"]


@pytest.mark.parametrize(
    ("audio_url", "download_url"),
    (
        (
            "http://dashscope-result-sh.oss-cn-shanghai.aliyuncs.com/a1.wav?Expires=1&Signature=ab%2Bc%3D",
            "https://dashscope-result-sh.oss-cn-shanghai.aliyuncs.com/a1.wav?Expires=1&Signature=ab%2Bc%3D",
        ),
        (
            "https://dashscope-result-sh.oss-cn-shanghai.aliyuncs.com/a1.wav",
            "https://dashscope-result-sh.oss-cn-shanghai.aliyuncs.com/a1.wav",
        ),
        ("http://audio.notaliyuncs.com/a1.wav", "http://audio.notaliyuncs.com/a1.wav"),
        (AUDIO_URL, AUDIO_URL),
    ),
)
def test_an_alibaba_cloud_audio_url_is_downloaded_over_https(
    monkeypatch: pytest.MonkeyPatch, audio_url: str, download_url: str
) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", False)
    download_seen: Final[list[httpx.Request]] = []  # mutable-ok: request log
    config: Final = DashScopeTextToSpeechConfig(
        sync_client=lambda: HTTPHandler(client=httpx.Client(transport=_recording(download_seen, _audio)))
    )

    speech: Final = BaseLLMHTTPHandler().text_to_speech_handler(
        model="qwen3-tts-flash",
        input="Hello.",
        voice="Cherry",
        text_to_speech_provider_config=config,
        text_to_speech_optional_params={},
        custom_llm_provider="dashscope",
        litellm_params={"api_key": "k", "api_base": None},
        logging_obj=_logging(),
        timeout=30,
        client=HTTPHandler(
            client=httpx.Client(
                transport=_recording([], _api({**SYNTHESIZED, "output": {"audio": {"url": audio_url}}}))
            )
        ),
    )

    assert speech.content == AUDIO
    (download,) = download_seen
    assert str(download.url) == download_url


@pytest.mark.parametrize(
    ("api_base", "environment", "model", "url"),
    (
        (None, None, "qwen3-tts-flash", f"{INTL}{MULTIMODAL_PATH}"),
        (None, None, "qwen-audio-3.1-tts-plus", f"{INTL}{SYNTHESIZER_PATH}"),
        (
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            None,
            "qwen-audio-3.1-tts-plus",
            f"https://dashscope.aliyuncs.com{SYNTHESIZER_PATH}",
        ),
        (f"{INTL}{SYNTHESIZER_PATH}", None, "qwen3-tts-flash", f"{INTL}{MULTIMODAL_PATH}"),
        (
            "https://dashscope.aliyuncs.com/apps/anthropic",
            None,
            "qwen3-tts-flash",
            f"https://dashscope.aliyuncs.com{MULTIMODAL_PATH}",
        ),
        (
            "https://relay.example.com/dashscope/",
            None,
            "qwen3-tts-flash",
            f"https://relay.example.com/dashscope{MULTIMODAL_PATH}",
        ),
        (
            None,
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "qwen3-tts-flash",
            f"https://dashscope.aliyuncs.com{MULTIMODAL_PATH}",
        ),
    ),
)
def test_the_endpoint_follows_the_model_family_on_the_configured_host(
    monkeypatch: pytest.MonkeyPatch, api_base: str | None, environment: str | None, model: str, url: str
) -> None:
    if environment is not None:
        monkeypatch.setenv("DASHSCOPE_API_BASE", environment)

    assert DashScopeTextToSpeechConfig().get_complete_url(model=model, api_base=api_base, litellm_params={}) == url


@pytest.mark.parametrize(
    ("model", "optional_params", "voice", "refused"),
    (
        ("qwen3-tts-flash", {"response_format": "mp3"}, "Cherry", "response_format='mp3'"),
        ("qwen3-tts-flash", {"instructions": "Whisper."}, "Cherry", "instructions='Whisper.'"),
        ("qwen-audio-3.1-tts-flash", {"response_format": "aac"}, "Cherry", "response_format='aac'"),
        ("qwen-audio-3.1-tts-flash", {"speed": 1.5}, "Cherry", "speed=1.5"),
        ("qwen-audio-3.1-tts-flash", {}, {"voice_id": "Cherry"}, "voice as an object"),
    ),
)
def test_an_unsupported_param_answers_400_and_is_dropped_under_drop_params(
    model: str,
    optional_params: dict[str, object],  # mutable-ok: parametrized fixture
    voice: str | dict[str, object],  # mutable-ok: parametrized fixture
    refused: str,
) -> None:
    config: Final = DashScopeTextToSpeechConfig()

    with pytest.raises(litellm.UnsupportedParamsError) as rejected:
        config.map_openai_params(model=model, optional_params=optional_params, voice=voice)
    dropped_voice, dropped = config.map_openai_params(
        model=model, optional_params=optional_params, voice=voice, kwargs={"drop_params": True}
    )

    assert rejected.value.status_code == 400
    assert refused in str(rejected.value)
    assert dropped == {}
    assert dropped_voice == (voice if isinstance(voice, str) else None)


async def test_aspeech_uses_the_native_endpoint_on_the_caller_api_base_and_refuses_a_private_audio_url() -> None:
    api_seen: Final[list[httpx.Request]] = []  # mutable-ok: request log
    private_audio: Final = {"output": {"audio": {"url": "http://169.254.169.254/latest/meta-data"}}}

    with pytest.raises(litellm.BadGatewayError) as refused:
        await litellm.aspeech(
            model="dashscope/qwen-audio-3.1-tts-flash",
            input="Hello.",
            voice="longanhuan_v3.1",
            response_format="opus",
            api_key="sk-dashscope-test",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            client=AsyncHTTPHandler(transport=_recording(api_seen, _api(private_audio))),
        )

    assert refused.value.status_code == 502
    assert "audio URL that was refused" in str(refused.value)
    (api_request,) = api_seen
    assert str(api_request.url) == f"https://dashscope.aliyuncs.com{SYNTHESIZER_PATH}"
    assert json.loads(api_request.content) == {
        "model": "qwen-audio-3.1-tts-flash",
        "input": {"text": "Hello.", "voice": "longanhuan_v3.1", "format": "opus"},
    }


@pytest.mark.parametrize(
    ("api_reply", "audio_reply", "status", "message"),
    (
        (
            httpx.Response(200, json={"code": "InvalidParameter", "message": "The voice property is required."}),
            _audio,
            400,
            "The voice property is required.",
        ),
        (
            httpx.Response(200, content=b"<html>gateway</html>"),
            _audio,
            502,
            "Unexpected DashScope text-to-speech response",
        ),
        (
            httpx.Response(200, json=SYNTHESIZED, request=API_REQUEST),
            lambda _: httpx.Response(403),
            502,
            "audio download failed with HTTP 403",
        ),
    ),
)
async def test_a_failed_synthesis_or_download_raises_a_dashscope_error(
    api_reply: httpx.Response, audio_reply: Reply, status: int, message: str
) -> None:
    config: Final = DashScopeTextToSpeechConfig(
        async_client=lambda: AsyncHTTPHandler(transport=httpx.MockTransport(audio_reply))
    )

    with pytest.raises(DashScopeError) as failed:
        await config.async_transform_text_to_speech_response(
            model="qwen3-tts-flash", raw_response=api_reply, logging_obj=_logging()
        )

    assert failed.value.status_code == status
    assert message in failed.value.message


async def test_provider_token_usage_rides_on_the_audio_so_token_priced_speech_is_billed() -> None:
    config: Final = DashScopeTextToSpeechConfig(
        async_client=lambda: AsyncHTTPHandler(transport=httpx.MockTransport(_audio))
    )

    speech: Final = await config.async_transform_text_to_speech_response(
        model="qwen-audio-3.1-tts-flash",
        raw_response=httpx.Response(200, json=SYNTHESIZED, request=API_REQUEST),
        logging_obj=_logging(),
    )

    assert speech.usage == Usage(prompt_tokens=8, completion_tokens=30, total_tokens=38)


@pytest.mark.usefixtures("local_model_cost_map")
async def test_character_priced_speech_bills_the_characters_dashscope_reports() -> None:
    text: Final = "你好 世界。"
    reported: Final = {**SYNTHESIZED, "usage": {"characters": 10}}
    config: Final = DashScopeTextToSpeechConfig(
        async_client=lambda: AsyncHTTPHandler(transport=httpx.MockTransport(_audio))
    )
    speech: Final = await config.async_transform_text_to_speech_response(
        model="qwen3-tts-flash",
        raw_response=httpx.Response(200, json=reported, request=API_REQUEST),
        logging_obj=_logging(),
    )

    cost: Final = litellm.completion_cost(
        completion_response=speech,
        model="dashscope/qwen3-tts-flash",
        call_type="aspeech",
        prompt=text,
        custom_llm_provider="dashscope",
    )

    assert litellm.utils.count_characters(text) != 10
    assert cost == pytest.approx(10 * litellm.model_cost["dashscope/qwen3-tts-flash"]["input_cost_per_character"])


@pytest.mark.parametrize(
    "model", ("qwen3-tts-flash", "qwen-audio-3.1-tts-plus", "qwen-audio-3.0-tts-plus", "qwen-audio-3.0-tts-flash")
)
@pytest.mark.usefixtures("local_model_cost_map")
def test_character_priced_speech_bills_each_input_character_at_the_model_rate(model: str) -> None:
    cost: Final = litellm.completion_cost(
        completion_response=HttpxBinaryResponseContent(httpx.Response(200, content=AUDIO)),
        model=f"dashscope/{model}",
        call_type="aspeech",
        prompt="Hello.",
        custom_llm_provider="dashscope",
    )

    assert cost == pytest.approx(6 * litellm.model_cost[f"dashscope/{model}"]["input_cost_per_character"])
    assert cost > 0


@pytest.mark.usefixtures("local_model_cost_map")
def test_token_priced_speech_bills_the_provider_reported_input_and_output_tokens() -> None:
    speech: Final = HttpxBinaryResponseContent(httpx.Response(200, content=AUDIO))
    speech.usage = Usage(prompt_tokens=8, completion_tokens=30, total_tokens=38)
    row: Final = litellm.model_cost["dashscope/qwen-audio-3.1-tts-flash"]

    cost: Final = litellm.completion_cost(
        completion_response=speech,
        model="dashscope/qwen-audio-3.1-tts-flash",
        call_type="aspeech",
        prompt="Hello.",
        custom_llm_provider="dashscope",
    )

    assert cost == pytest.approx(8 * row["input_cost_per_token"] + 30 * row["output_cost_per_token"])
    assert row["output_cost_per_token"] > row["input_cost_per_token"] > 0
