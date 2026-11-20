"""deAPI `/v1/audio/transcriptions`: every `response_format`, plain `text` included, reports the charge in
the `X-DeAPI-Cost` header, so cost tracking does not depend on asking for `verbose_json`."""

import httpx
import pytest

import litellm
from litellm.cost_calculator import get_response_cost_from_hidden_params
from litellm.llms.deapi.audio_transcription.transformation import DeapiAudioTranscriptionConfig
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

DEAPI_TRANSCRIPTIONS_URL = "https://oai.deapi.ai/v1/audio/transcriptions"
MODEL = "deapi/WhisperLargeV3Ct2"
SELLER_MODEL = "WhisperLargeV3Ct2"
REPORTED_COST = 0.00112
AUDIO_FILE = ("hello.mp3", b"ID3\x04\x00fake-mp3-bytes", "audio/mpeg")
COST_HEADER = {"X-DeAPI-Cost": f"{REPORTED_COST:.8f}"}


class TestRegistration:
    def test_deapi_is_a_native_transcription_provider(self):
        config = ProviderConfigManager.get_provider_audio_transcription_config(
            model=SELLER_MODEL, provider=LlmProviders.DEAPI
        )

        assert isinstance(config, DeapiAudioTranscriptionConfig)


class TestRequestTransformation:
    def test_sends_the_file_as_multipart_without_forcing_verbose_json(self):
        request = DeapiAudioTranscriptionConfig().transform_audio_transcription_request(
            model=SELLER_MODEL, audio_file=AUDIO_FILE, optional_params={"language": "en"}, litellm_params={}
        )

        assert request.data == {"model": SELLER_MODEL, "language": "en"}
        assert request.files == {"file": AUDIO_FILE}

    def test_sdk_style_extra_body_is_flattened_into_form_fields(self):
        request = DeapiAudioTranscriptionConfig().transform_audio_transcription_request(
            model=SELLER_MODEL,
            audio_file=AUDIO_FILE,
            optional_params={"language": "en", "extra_body": {"model": SELLER_MODEL, "user": "u-1"}},
            litellm_params={},
        )

        assert request.data == {"model": SELLER_MODEL, "language": "en", "user": "u-1"}

    def test_missing_key_is_an_authentication_error_before_any_request(self, no_deapi_key, respx_mock):
        with pytest.raises(litellm.AuthenticationError, match="DEAPI_API_KEY"):
            litellm.transcription(model=MODEL, file=AUDIO_FILE)
        assert not respx_mock.calls


class TestTranscription:
    def test_a_json_answer_without_usage_is_priced_from_the_header(self, deapi_key, respx_mock):
        respx_mock.post(DEAPI_TRANSCRIPTIONS_URL).mock(
            return_value=httpx.Response(200, json={"text": "Hello there."}, headers=COST_HEADER)
        )

        response = litellm.transcription(model=MODEL, file=AUDIO_FILE, response_format="json")

        assert response.text == "Hello there."
        assert respx_mock.calls.last.request.headers["Authorization"].startswith("Bearer ")
        assert get_response_cost_from_hidden_params(response._hidden_params) == REPORTED_COST

    def test_a_plain_text_answer_is_priced_from_the_header(self, deapi_key, respx_mock):
        respx_mock.post(DEAPI_TRANSCRIPTIONS_URL).mock(
            return_value=httpx.Response(
                200, content=b"Hello there.", headers={**COST_HEADER, "content-type": "text/plain; charset=utf-8"}
            )
        )

        response = litellm.transcription(model=MODEL, file=AUDIO_FILE, response_format="text")

        assert response.text == "Hello there."
        assert get_response_cost_from_hidden_params(response._hidden_params) == REPORTED_COST

    def test_a_verbose_answer_keeps_its_segments_and_cost(self, deapi_key, respx_mock):
        body = {
            "task": "transcribe",
            "language": "english",
            "duration": 4.2,
            "text": "Hello there.",
            "segments": [{"id": 0, "start": 0.0, "end": 1.2, "text": "Hello there."}],
            "usage": {"type": "duration", "seconds": 4.2, "cost": REPORTED_COST},
        }
        respx_mock.post(DEAPI_TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=body, headers=COST_HEADER))

        response = litellm.transcription(model=MODEL, file=AUDIO_FILE, response_format="verbose_json")

        assert response.text == "Hello there."
        assert response["segments"][0]["end"] == 1.2
        assert get_response_cost_from_hidden_params(response._hidden_params) == REPORTED_COST

    @pytest.mark.asyncio
    async def test_async_call_logs_the_header_cost_as_spend(
        self, deapi_key, httpx_transport, spend_capture, respx_mock
    ):
        respx_mock.post(DEAPI_TRANSCRIPTIONS_URL).mock(
            return_value=httpx.Response(200, json={"text": "Hello there."}, headers=COST_HEADER)
        )

        await litellm.atranscription(model=MODEL, file=AUDIO_FILE, litellm_call_id=spend_capture.call_id)
        await spend_capture.settle()

        assert spend_capture.costs == [REPORTED_COST]
