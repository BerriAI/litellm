import base64
import json
from datetime import datetime
from functools import partial
from typing import Final

import httpx
import pytest

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.alibaba_token_plan.audio_transcription.transformation import AlibabaTokenPlanAudioTranscriptionConfig
from litellm.llms.alibaba_token_plan.common_utils import IMAGE_ENDPOINT
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.utils import TranscriptionResponse

MODEL: Final = "qwen-audio-3.0-asr-flash"
BASE_URL: Final = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
ENDPOINT: Final = (
    "https://token-plan.ap-southeast-1.maas.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
)


def make_logging() -> Logging:
    return Logging(
        model=MODEL,
        messages=[],
        stream=False,
        call_type="transcription",
        start_time=datetime(2020, 1, 1),
        litellm_call_id="test-transcription",
        function_id="test-transcription",
    )


def transcription_transport(request: httpx.Request, expected_url: str = ENDPOINT) -> httpx.Response:
    assert str(request.url) == expected_url
    assert request.headers["authorization"] == "Bearer test-token-plan-key"
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == {
        "model": MODEL,
        "input": {
            "messages": [
                {"role": "user", "content": [{"type": "input_text", "text": "Qwen and Alibaba"}]},
                {
                    "role": "user",
                    "content": [{"type": "input_audio", "input_audio": {"data": "data:audio/wav;base64,YXVkaW8="}}],
                },
            ]
        },
        "parameters": {"format": "wav", "language_hints": ["en"]},
    }
    return httpx.Response(
        200,
        json={
            "output": {"text": "First sentence. Second sentence.", "sentence": {"text": "Second sentence."}},
            "usage": {"duration": 4},
        },
    )


def test_sync_handler_sends_json_audio_and_preserves_full_transcript() -> None:
    config: Final = AlibabaTokenPlanAudioTranscriptionConfig()
    with httpx.Client(transport=httpx.MockTransport(transcription_transport)) as http_client:
        response: Final = BaseLLMHTTPHandler().audio_transcriptions(
            model=MODEL,
            audio_file=("recording.wav", b"audio"),
            optional_params={"model": MODEL, "language": "en", "prompt": "Qwen and Alibaba", "response_format": "json"},
            litellm_params={},
            model_response=TranscriptionResponse(),
            timeout=10.0,
            max_retries=0,
            logging_obj=make_logging(),
            api_key="test-token-plan-key",
            api_base=BASE_URL,
            custom_llm_provider="alibaba_token_plan",
            client=HTTPHandler(client=http_client),
            provider_config=config,
        )
    assert isinstance(response, TranscriptionResponse)
    assert response.text == "First sentence. Second sentence."
    assert response.usage is not None
    assert response.usage.model_dump() == {"type": "duration", "seconds": 4.0}
    assert response["duration"] == 4


@pytest.mark.parametrize(
    "api_base,expected_url",
    [
        (BASE_URL, ENDPOINT),
        ("https://gateway.example/plan/compatible-mode/v1", f"https://gateway.example/plan/{IMAGE_ENDPOINT}"),
        ("https://gateway.example/plan/apps/anthropic", f"https://gateway.example/plan/{IMAGE_ENDPOINT}"),
    ],
)
def test_transcription_routes_openai_compatible_provider_to_native_json_endpoint(
    api_base: str, expected_url: str
) -> None:
    transport: Final = httpx.MockTransport(partial(transcription_transport, expected_url=expected_url))
    with httpx.Client(transport=transport) as http_client:
        response: Final = litellm.transcription(
            model=f"alibaba_token_plan/{MODEL}",
            file=("recording.wav", b"audio"),
            language="en",
            prompt="Qwen and Alibaba",
            response_format="json",
            api_key="test-token-plan-key",
            api_base=api_base,
            client=HTTPHandler(client=http_client),
        )
    assert isinstance(response, TranscriptionResponse)
    assert response.text == "First sentence. Second sentence."


@pytest.mark.asyncio
async def test_async_handler_sends_json_audio() -> None:
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(transcription_transport))
    async with client.client:
        response: Final = await BaseLLMHTTPHandler().async_audio_transcriptions(
            model=MODEL,
            audio_file=("recording.wav", b"audio"),
            optional_params={"language": "en", "prompt": "Qwen and Alibaba"},
            litellm_params={},
            model_response=TranscriptionResponse(),
            timeout=10.0,
            max_retries=0,
            logging_obj=make_logging(),
            api_key="test-token-plan-key",
            api_base=BASE_URL,
            custom_llm_provider="alibaba_token_plan",
            client=client,
            provider_config=AlibabaTokenPlanAudioTranscriptionConfig(),
        )
    assert response.text == "First sentence. Second sentence."


@pytest.mark.parametrize("extension,mime", [("mp3", "audio/mpeg"), ("webm", "video/webm"), ("wav", "audio/wav")])
def test_request_preserves_audio_format_without_inventing_language(extension: str, mime: str) -> None:
    request: Final = AlibabaTokenPlanAudioTranscriptionConfig().transform_audio_transcription_request(
        model=f"alibaba_token_plan/{MODEL}",
        audio_file=(f"recording.{extension}", b"recorded audio"),
        optional_params={},
        litellm_params={},
    )
    assert request.files is None
    assert request.data == {
        "model": MODEL,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"data:{mime};base64,{base64.b64encode(b'recorded audio').decode()}"
                            },
                        }
                    ],
                }
            ]
        },
        "parameters": {"format": extension},
    }


@pytest.mark.parametrize("response_format", ["text", "verbose_json", "srt", "vtt"])
def test_unsupported_response_format_is_rejected_or_explicitly_dropped(response_format: str) -> None:
    config: Final = AlibabaTokenPlanAudioTranscriptionConfig()
    with pytest.raises(UnsupportedParamsError, match="response_format"):
        config.map_openai_params({"response_format": response_format}, {}, MODEL, False)
    assert config.map_openai_params({"response_format": response_format, "language": "en"}, {}, MODEL, True) == {
        "language": "en"
    }


@pytest.mark.parametrize("parameter,value", [("temperature", 1), ("timestamp_granularities", ["word"])])
def test_unsupported_openai_parameters_raise(parameter: str, value: object) -> None:
    with pytest.raises(UnsupportedParamsError, match=parameter):
        AlibabaTokenPlanAudioTranscriptionConfig().map_openai_params({parameter: value}, {}, MODEL, False)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"code": "InvalidParameter", "message": "Invalid audio"}),
        httpx.Response(200, json={"output": {"sentence": {"text": "Only the final sentence"}}}),
        httpx.Response(200, text="not JSON"),
    ],
)
def test_provider_errors_and_malformed_responses_do_not_become_empty_successes(response: httpx.Response) -> None:
    with pytest.raises(BaseLLMException) as error:
        AlibabaTokenPlanAudioTranscriptionConfig().transform_audio_transcription_response(response)
    assert error.value.status_code == 502


def test_malformed_transcription_output_is_not_exposed() -> None:
    secret: Final = "sensitive-provider-value"
    with pytest.raises(BaseLLMException, match="Invalid Alibaba Token Plan transcription response") as error:
        AlibabaTokenPlanAudioTranscriptionConfig().transform_audio_transcription_response(
            httpx.Response(200, json={"output": {"text": "Transcript"}, "usage": {"duration": secret}})
        )
    assert secret not in str(error.value)


def test_http_authentication_error_is_preserved() -> None:
    with pytest.raises(BaseLLMException) as error:
        AlibabaTokenPlanAudioTranscriptionConfig().transform_audio_transcription_response(
            httpx.Response(401, json={"message": "Invalid API key"})
        )
    assert error.value.status_code == 401


def test_non_audio_file_is_rejected() -> None:
    with pytest.raises(BaseLLMException, match="audio format"):
        AlibabaTokenPlanAudioTranscriptionConfig().transform_audio_transcription_request(
            model=MODEL,
            audio_file=("image.png", b"image"),
            optional_params={},
            litellm_params={},
        )
