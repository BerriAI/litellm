"""Eden AI `/v3/audio/speech`: OpenAI's text-to-speech API served by Eden's gateway. The answer is
raw audio, so Eden reports the real per-request cost in the `x-edenai-cost` response header."""

import asyncio
import json
import uuid

import httpx
import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.edenai.common_utils import EdenAIException
from litellm.llms.edenai.text_to_speech.transformation import EdenAITextToSpeechConfig
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

EDEN_BASE = "https://api.edenai.run/v3"
EDEN_SPEECH_URL = f"{EDEN_BASE}/audio/speech"
EDEN_REPORTED_COST = 0.00015
MODEL = "edenai/openai/tts-1"
SELLER_MODEL = "openai/tts-1"
AUDIO = b"ID3\x04\x00fake-mp3-bytes"


def _eden_audio(cost: float | None = EDEN_REPORTED_COST) -> httpx.Response:
    """Live `/v3/audio/speech` answer: audio bytes, with the cost and provider in `x-edenai-*` headers."""
    headers = {"content-type": "audio/mpeg", "x-edenai-provider": "openai"}
    return httpx.Response(
        200, content=AUDIO, headers=headers if cost is None else {**headers, "x-edenai-cost": str(cost)}
    )


def _request_body(respx_mock) -> dict:
    return json.loads(respx_mock.calls.last.request.content)


class TestRegistration:
    def test_eden_is_a_native_text_to_speech_provider(self):
        config = ProviderConfigManager.get_provider_text_to_speech_config(
            model=SELLER_MODEL, provider=LlmProviders.EDENAI
        )

        assert isinstance(config, EdenAITextToSpeechConfig)


class TestRequestTransformation:
    def test_body_is_the_openai_speech_request_without_empty_fields(self):
        request = EdenAITextToSpeechConfig().transform_text_to_speech_request(
            model=SELLER_MODEL,
            input="hello there",
            voice="alloy",
            optional_params={"response_format": "wav", "speed": None},
            litellm_params={},
            headers={},
        )

        assert request["dict_body"] == {
            "model": SELLER_MODEL,
            "input": "hello there",
            "voice": "alloy",
            "response_format": "wav",
        }

    def test_a_missing_voice_is_left_for_eden_to_reject(self):
        request = EdenAITextToSpeechConfig().transform_text_to_speech_request(
            model=SELLER_MODEL, input="hello", voice=None, optional_params={}, litellm_params={}, headers={}
        )

        assert "voice" not in request["dict_body"]

    def test_missing_key_is_an_authentication_error_before_any_request(self, no_eden_key, respx_mock):
        with pytest.raises(litellm.AuthenticationError, match="EDENAI_API_KEY"):
            litellm.speech(model=MODEL, input="hello", voice="alloy")
        assert not respx_mock.calls


class TestSpeech:
    def test_posts_to_eden_with_the_bearer_key_and_returns_the_audio(self, eden_key, respx_mock):
        respx_mock.post(EDEN_SPEECH_URL).mock(return_value=_eden_audio())

        response = litellm.speech(model=MODEL, input="hello there", voice="alloy", response_format="mp3", speed=1.2)

        assert isinstance(response, HttpxBinaryResponseContent)
        assert response.content == AUDIO
        assert respx_mock.calls.last.request.headers["Authorization"] == f"Bearer {eden_key}"
        assert _request_body(respx_mock) == {
            "model": SELLER_MODEL,
            "input": "hello there",
            "voice": "alloy",
            "response_format": "mp3",
            "speed": 1.2,
        }

    def test_the_cost_header_becomes_the_response_cost(self, eden_key, respx_mock):
        respx_mock.post(EDEN_SPEECH_URL).mock(return_value=_eden_audio())

        response = litellm.speech(model=MODEL, input="hello there", voice="alloy")

        assert response._hidden_params["response_cost"] == EDEN_REPORTED_COST

    def test_an_answer_without_the_cost_header_leaves_pricing_to_the_price_map(self):
        response = EdenAITextToSpeechConfig().transform_text_to_speech_response(
            model=SELLER_MODEL, raw_response=_eden_audio(cost=None), logging_obj=None
        )

        assert "response_cost" not in response._hidden_params

    @pytest.mark.asyncio
    async def test_async_call_logs_the_header_cost_as_spend(self, eden_key, httpx_transport, spend_capture, respx_mock):
        respx_mock.post(EDEN_SPEECH_URL).mock(return_value=_eden_audio())

        response = await litellm.aspeech(
            model=MODEL, input="hello there", voice="alloy", litellm_call_id=spend_capture.call_id
        )
        await spend_capture.settle()

        assert response.content == AUDIO
        assert spend_capture.costs == [EDEN_REPORTED_COST]


class TestErrors:
    def test_middleware_401_surfaces_as_an_eden_error_with_the_status_code(self, eden_key, respx_mock):
        """`litellm.speech` does not map provider errors onto the OpenAI exception classes the way
        chat does, so the proxy relies on the status code the provider exception carries."""
        respx_mock.post(EDEN_SPEECH_URL).mock(return_value=httpx.Response(401, json={"detail": "Invalid token."}))

        with pytest.raises(EdenAIException, match="Invalid token") as excinfo:
            litellm.speech(model=MODEL, input="hello", voice="alloy")
        assert excinfo.value.status_code == 401

    @pytest.mark.asyncio
    async def test_async_401_maps_to_authentication_error(self, eden_key, httpx_transport, respx_mock):
        respx_mock.post(EDEN_SPEECH_URL).mock(return_value=httpx.Response(401, json={"detail": "Invalid token."}))

        with pytest.raises(litellm.AuthenticationError, match="Invalid token"):
            await litellm.aspeech(model=MODEL, input="hello", voice="alloy")
