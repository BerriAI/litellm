"""Eden AI `/v3/audio/transcriptions`: OpenAI's speech-to-text API served by Eden's gateway, which
reports the real per-request cost at the top level of the JSON body."""

import httpx
import pytest

import litellm
from litellm.cost_calculator import get_response_cost_from_hidden_params
from litellm.llms.edenai.audio_transcription.transformation import EdenAIAudioTranscriptionConfig
from litellm.llms.edenai.common_utils import EdenAIException
from litellm.types.utils import LlmProviders, TranscriptionResponse
from litellm.utils import ProviderConfigManager

EDEN_BASE = "https://api.edenai.run/v3"
EDEN_TRANSCRIPTIONS_URL = f"{EDEN_BASE}/audio/transcriptions"
EDEN_REPORTED_COST = 0.0042
MODEL = "edenai/openai/whisper-1"
SELLER_MODEL = "openai/whisper-1"
AUDIO_FILE = ("hello.mp3", b"ID3\x04\x00fake-mp3-bytes", "audio/mpeg")


def _eden_transcription(cost: float | None = EDEN_REPORTED_COST) -> dict:
    """Live `/v3/audio/transcriptions` body: Whisper's verbose shape plus Eden's top-level `cost`
    and `provider`, with `duration` present whatever `response_format` was asked for."""
    body = {
        "text": "Hello there.",
        "usage": {"type": "duration", "seconds": 1.0},
        "language": "english",
        "task": "transcribe",
        "duration": 0.62,
        "words": None,
        "segments": [{"id": 0, "start": 0.0, "end": 0.8, "text": " Hello there."}],
        "provider": "openai",
    }
    return body if cost is None else {**body, "cost": cost}


def _multipart_body(respx_mock) -> str:
    return respx_mock.calls.last.request.content.decode(errors="replace")


class TestRegistration:
    def test_eden_is_a_native_transcription_provider(self):
        config = ProviderConfigManager.get_provider_audio_transcription_config(
            model=SELLER_MODEL, provider=LlmProviders.EDENAI
        )

        assert isinstance(config, EdenAIAudioTranscriptionConfig)


class TestRequestTransformation:
    def test_sends_the_file_as_multipart_without_forcing_verbose_json(self):
        request = EdenAIAudioTranscriptionConfig().transform_audio_transcription_request(
            model=SELLER_MODEL, audio_file=AUDIO_FILE, optional_params={"language": "en"}, litellm_params={}
        )

        assert request.data == {"model": SELLER_MODEL, "language": "en"}
        assert request.files == {"file": AUDIO_FILE}

    def test_sdk_style_extra_body_is_flattened_into_form_fields(self):
        """LiteLLM parks `model` and any non-OpenAI kwarg under `extra_body` for the OpenAI SDK, and a
        nested dict cannot ride in a multipart form."""
        request = EdenAIAudioTranscriptionConfig().transform_audio_transcription_request(
            model=SELLER_MODEL,
            audio_file=AUDIO_FILE,
            optional_params={"language": "en", "extra_body": {"model": SELLER_MODEL, "user": "u-1"}},
            litellm_params={},
        )

        assert request.data == {"model": SELLER_MODEL, "language": "en", "user": "u-1"}

    def test_missing_key_is_an_authentication_error_before_any_request(self, no_eden_key, respx_mock):
        with pytest.raises(litellm.AuthenticationError, match="EDENAI_API_KEY"):
            litellm.transcription(model=MODEL, file=AUDIO_FILE)
        assert not respx_mock.calls


class TestTranscription:
    def test_posts_multipart_to_eden_with_the_bearer_key_and_the_seller_model_id(self, eden_key, respx_mock):
        respx_mock.post(EDEN_TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=_eden_transcription()))

        response = litellm.transcription(model=MODEL, file=AUDIO_FILE, language="en", temperature=0)

        assert isinstance(response, TranscriptionResponse)
        assert response.text == "Hello there."
        request = respx_mock.calls.last.request
        assert request.headers["Authorization"] == f"Bearer {eden_key}"
        assert request.headers["Content-Type"].startswith("multipart/form-data")
        body = _multipart_body(respx_mock)
        assert f'name="model"\r\n\r\n{SELLER_MODEL}' in body
        assert 'name="language"\r\n\r\nen' in body
        assert 'name="temperature"\r\n\r\n0' in body
        assert 'name="file"; filename="hello.mp3"' in body
        assert "verbose_json" not in body

    def test_eden_reported_cost_beats_the_price_map(self, eden_key, respx_mock):
        respx_mock.post(EDEN_TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=_eden_transcription()))

        response = litellm.transcription(model=MODEL, file=AUDIO_FILE)

        assert get_response_cost_from_hidden_params(response._hidden_params) == EDEN_REPORTED_COST
        assert response._hidden_params["response_cost"] == EDEN_REPORTED_COST

    def test_a_body_without_cost_leaves_pricing_to_the_price_map(self, eden_key, respx_mock):
        respx_mock.post(EDEN_TRANSCRIPTIONS_URL).mock(
            return_value=httpx.Response(200, json=_eden_transcription(cost=None))
        )

        response = litellm.transcription(model=MODEL, file=AUDIO_FILE)

        assert get_response_cost_from_hidden_params(response._hidden_params) is None
        assert response.duration == 0.62
        assert response.usage is not None
        assert response.usage.seconds == 1.0

    def test_a_plain_text_answer_is_the_transcript(self, eden_key, respx_mock):
        respx_mock.post(EDEN_TRANSCRIPTIONS_URL).mock(
            return_value=httpx.Response(200, text="Hello there.", headers={"content-type": "text/plain"})
        )

        response = litellm.transcription(model=MODEL, file=AUDIO_FILE, response_format="text")

        assert response.text == "Hello there."
        assert 'name="response_format"\r\n\r\ntext' in _multipart_body(respx_mock)

    @pytest.mark.asyncio
    async def test_async_call_tracks_the_same_cost(self, eden_key, httpx_transport, respx_mock):
        respx_mock.post(EDEN_TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=_eden_transcription()))

        response = await litellm.atranscription(model=MODEL, file=AUDIO_FILE)

        assert response.text == "Hello there."
        assert response._hidden_params["response_cost"] == EDEN_REPORTED_COST


class TestErrors:
    def test_sync_401_surfaces_as_an_eden_error_with_the_status_code(self, eden_key, respx_mock):
        """`litellm.transcription` does not map provider errors onto the OpenAI exception classes the
        way its async twin does, so the proxy relies on the status code the provider exception carries."""
        respx_mock.post(EDEN_TRANSCRIPTIONS_URL).mock(
            return_value=httpx.Response(401, json={"detail": "Invalid token."})
        )

        with pytest.raises(EdenAIException, match="Invalid token") as excinfo:
            litellm.transcription(model=MODEL, file=AUDIO_FILE)
        assert excinfo.value.status_code == 401

    @pytest.mark.asyncio
    async def test_async_401_maps_to_authentication_error(self, eden_key, httpx_transport, respx_mock):
        respx_mock.post(EDEN_TRANSCRIPTIONS_URL).mock(
            return_value=httpx.Response(401, json={"detail": "Invalid token."})
        )

        with pytest.raises(litellm.AuthenticationError, match="Invalid token"):
            await litellm.atranscription(model=MODEL, file=AUDIO_FILE)
