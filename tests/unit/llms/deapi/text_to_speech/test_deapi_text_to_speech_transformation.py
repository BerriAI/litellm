"""deAPI `/v1/audio/speech`: the answer is raw audio, so the charge travels in the `X-DeAPI-Cost` header."""

import json

import httpx
import pytest

import litellm
from litellm.llms.deapi.common_utils import DeapiException
from litellm.llms.deapi.text_to_speech.transformation import DeapiTextToSpeechConfig
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

DEAPI_SPEECH_URL = "https://oai.deapi.ai/v1/audio/speech"
MODEL = "deapi/Kokoro"
SELLER_MODEL = "Kokoro"
REPORTED_COST = 0.000041
AUDIO = b"ID3\x04\x00fake-mp3-bytes"


def _audio(cost: float | None = REPORTED_COST) -> httpx.Response:
    headers = {"content-type": "audio/mpeg"}
    return httpx.Response(
        200, content=AUDIO, headers=headers if cost is None else {**headers, "X-DeAPI-Cost": f"{cost:.8f}"}
    )


class TestRegistration:
    def test_deapi_is_a_native_text_to_speech_provider(self):
        config = ProviderConfigManager.get_provider_text_to_speech_config(
            model=SELLER_MODEL, provider=LlmProviders.DEAPI
        )

        assert isinstance(config, DeapiTextToSpeechConfig)

    def test_missing_key_is_an_authentication_error_before_any_request(self, no_deapi_key, respx_mock):
        with pytest.raises(litellm.AuthenticationError, match="DEAPI_API_KEY"):
            litellm.speech(model=MODEL, input="hello", voice="af_bella")
        assert not respx_mock.calls


class TestSpeech:
    def test_posts_the_openai_request_with_the_bearer_key_and_returns_the_audio(self, deapi_key, respx_mock):
        respx_mock.post(DEAPI_SPEECH_URL).mock(return_value=_audio())

        response = litellm.speech(model=MODEL, input="hello there", voice="af_bella", response_format="mp3", speed=1.2)

        assert isinstance(response, HttpxBinaryResponseContent)
        assert response.content == AUDIO
        assert respx_mock.calls.last.request.headers["Authorization"] == f"Bearer {deapi_key}"
        assert json.loads(respx_mock.calls.last.request.content) == {
            "model": SELLER_MODEL,
            "input": "hello there",
            "voice": "af_bella",
            "response_format": "mp3",
            "speed": 1.2,
        }

    def test_the_cost_header_becomes_the_response_cost(self, deapi_key, respx_mock):
        respx_mock.post(DEAPI_SPEECH_URL).mock(return_value=_audio())

        response = litellm.speech(model=MODEL, input="hello there", voice="af_bella")

        assert response._hidden_params["response_cost"] == REPORTED_COST

    def test_an_answer_without_the_cost_header_leaves_pricing_to_the_price_map(self):
        response = DeapiTextToSpeechConfig().transform_text_to_speech_response(
            model=SELLER_MODEL, raw_response=_audio(cost=None), logging_obj=None
        )

        assert "response_cost" not in response._hidden_params

    @pytest.mark.asyncio
    async def test_async_call_logs_the_header_cost_as_spend(
        self, deapi_key, httpx_transport, spend_capture, respx_mock
    ):
        respx_mock.post(DEAPI_SPEECH_URL).mock(return_value=_audio())

        response = await litellm.aspeech(
            model=MODEL, input="hello there", voice="af_bella", litellm_call_id=spend_capture.call_id
        )
        await spend_capture.settle()

        assert response.content == AUDIO
        assert spend_capture.costs == [REPORTED_COST]


class TestErrors:
    def test_unsupported_format_surfaces_as_a_deapi_error_with_the_status_code(self, deapi_key, respx_mock):
        envelope = {"error": {"message": "Only mp3 is supported.", "code": "format_not_supported"}}
        respx_mock.post(DEAPI_SPEECH_URL).mock(return_value=httpx.Response(400, json=envelope))

        with pytest.raises(DeapiException, match="Only mp3") as excinfo:
            litellm.speech(model=MODEL, input="hello", voice="af_bella", response_format="wav")
        assert excinfo.value.status_code == 400
