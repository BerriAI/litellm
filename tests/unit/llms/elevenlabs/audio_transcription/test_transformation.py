import traceback
from collections.abc import Iterator
from datetime import datetime
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import respx
from pydantic import ValidationError

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.elevenlabs.audio_transcription.transformation import ElevenLabsAudioTranscriptionConfig
from litellm.llms.elevenlabs.text_to_speech.transformation import ElevenLabsTextToSpeechConfig
from litellm.types.utils import TranscriptionResponse
from typing import Any, Dict
from unittest.mock import patch, MagicMock

ELEVENLABS_API_BASE: Final = "https://api.elevenlabs.io"
ELEVENLABS_TRANSCRIPTION_URL: Final = f"{ELEVENLABS_API_BASE}/v1/speech-to-text"
ELEVENLABS_API_KEY: Final = "test-elevenlabs-key"


@pytest.fixture
def elevenlabs_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", ELEVENLABS_API_KEY)


@pytest.fixture
def _elevenlabs_httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    client_cache: Final = LLMClientCache()
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", client_cache)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "force_ipv4", False)
    monkeypatch.setattr(litellm, "sync_transport", None, raising=False)
    yield
    client_cache.flush_cache()


def _form_fields(request: httpx.Request) -> list[tuple[str, str]]:
    head: Final = f"Content-Type: {request.headers['Content-Type']}\r\n\r\n".encode()
    message: Final = BytesParser(_class=EmailMessage, policy=policy.HTTP).parsebytes(head + request.content)
    assert isinstance(message, EmailMessage)
    return [
        (str(part.get_param("name", header="content-disposition")), str(part.get_content()))
        for part in message.iter_parts()
        if part.get_filename() is None
    ]


def _mock_transcription_endpoint(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post(ELEVENLABS_TRANSCRIPTION_URL).mock(
        return_value=httpx.Response(200, json={"text": "ok", "language_code": "en", "words": []})
    )


def _sent_fields(route: respx.Route) -> dict[str, list[str]]:
    fields: Final = _form_fields(route.calls.last.request)
    return {name: [value for field_name, value in fields if field_name == name] for name, _ in fields}


@pytest.mark.usefixtures("elevenlabs_api_key_env", "_elevenlabs_httpx_transport")
@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"keyterms": ["BOI report", "EIN"]}, {"keyterms": ["BOI report", "EIN"]}),
        ({"keyterms": ["EIN"]}, {"keyterms": ["EIN"]}),
        ({"diarize": True, "tag_audio_events": False}, {"diarize": ["true"], "tag_audio_events": ["false"]}),
        ({"num_speakers": 2, "language": "en"}, {"language_code": ["en"], "num_speakers": ["2"]}),
        ({"keyterms": ("EIN", "BOI report")}, {"keyterms": ["EIN", "BOI report"]}),
        ({"keyterms": ["EIN", None]}, {"keyterms": ["EIN"]}),
        ({"keyterms": []}, {}),
        ({"additional_formats": [{"format": "srt"}]}, {"additional_formats": ['[{"format": "srt"}]']}),
        ({"webhook_metadata": {"job": "a1", "retry": False}}, {"webhook_metadata": ['{"job": "a1", "retry": false}']}),
        ({"extra_body": {"keyterms": ["EIN"], "diarize": True}}, {"keyterms": ["EIN"], "diarize": ["true"]}),
        (
            {"keyterms": ["TOP"], "diarize": False, "extra_body": {"keyterms": ["EXTRA"], "diarize": True}},
            {"keyterms": ["EXTRA"], "diarize": ["true"]},
        ),
        ({"extra_body": {"diarize": None, "keyterms": ["EIN"]}}, {"keyterms": ["EIN"]}),
        ({"extra_body": {"model_id": "scribe_v1", "extra_body": {"diarize": True}}}, {}),
        ({"model_id": "scribe_v1"}, {}),
        ({"additional_formats": [{"format": "srt"}, None]}, {"additional_formats": ['[{"format": "srt"}]']}),
        ({"custom_rows": [["a", "b"], "c"]}, {"custom_rows": ['[["a", "b"], "c"]']}),
        (
            {"webhook_metadata": {"sent_at": datetime(2026, 1, 1), "job": MappingProxyType({"id": 1})}},
            {"webhook_metadata": ['{"sent_at": "2026-01-01 00:00:00", "job": {"id": 1}}']},
        ),
        (
            {"webhook_metadata": MappingProxyType({1: "a"}), "custom_rows": [{"job": MappingProxyType({2: "b"})}]},
            {"webhook_metadata": ['{"1": "a"}'], "custom_rows": ['[{"job": {"2": "b"}}]']},
        ),
    ],
    ids=[
        "keyterms-multi",
        "keyterms-single",
        "bools",
        "scalars",
        "keyterms-tuple",
        "keyterms-none-item",
        "keyterms-empty",
        "list-of-objects-as-json",
        "object-as-json",
        "extra-body-flattened",
        "extra-body-wins-over-top-level",
        "extra-body-none-dropped",
        "extra-body-cannot-set-model-or-nest",
        "top-level-model-id-cannot-override-model",
        "json-list-none-item-dropped",
        "nested-list-as-json",
        "datetime-as-text-mapping-as-json",
        "non-string-mapping-keys-as-json",
    ],
)
def test_transcription_sends_each_param_as_its_own_multipart_text_part(
    respx_mock: respx.MockRouter, params: dict[str, object], expected: dict[str, list[str]]
):
    route: Final = _mock_transcription_endpoint(respx_mock)

    litellm.transcription(model="elevenlabs/scribe_v2", file=b"fake audio", **params)

    assert _sent_fields(route) == {"model_id": ["scribe_v2"], **expected}


@pytest.mark.usefixtures("elevenlabs_api_key_env", "_elevenlabs_httpx_transport")
async def test_router_deployment_keyterms_default_is_sent_as_repeated_parts(respx_mock: respx.MockRouter):
    route: Final = _mock_transcription_endpoint(respx_mock)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "scribe",
                "litellm_params": {"model": "elevenlabs/scribe_v2", "api_key": "k", "keyterms": ["BOI report", "EIN"]},
            }
        ]
    )

    await router.atranscription(model="scribe", file=b"fake audio")

    assert _sent_fields(route) == {"model_id": ["scribe_v2"], "keyterms": ["BOI report", "EIN"]}


def _transform(payload: object) -> TranscriptionResponse:
    return ElevenLabsAudioTranscriptionConfig().transform_audio_transcription_response(
        raw_response=httpx.Response(200, json=payload)
    )


def test_transform_audio_transcription_response_keeps_only_spoken_words():
    payload = {
        "language_code": "en",
        "text": "Hello world",
        "words": [
            {"type": "word", "text": "Hello", "start": 0.0, "end": 0.4, "speaker_id": "speaker_0"},
            {"type": "spacing", "text": " ", "start": 0.4, "end": 0.5},
            {"type": "audio_event", "text": "(laughter)", "start": 0.5, "end": 0.9},
            {"type": "word", "text": "world", "start": 0.9, "end": 1.3},
        ],
    }

    response = _transform(payload)

    assert response.text == "Hello world"
    assert response["task"] == "transcribe"
    assert response["language"] == "en"
    assert response["words"] == [
        {"word": "Hello", "start": 0.0, "end": 0.4},
        {"word": "world", "start": 0.9, "end": 1.3},
    ]
    assert response._hidden_params == payload


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ({"type": "word"}, [{"word": "", "start": 0, "end": 0}]),
        ({"type": "word", "text": None, "start": None, "end": None}, [{"word": None, "start": None, "end": None}]),
        ({"type": "word", "text": 7, "start": "0.1", "end": [2]}, [{"word": 7, "start": "0.1", "end": [2]}]),
        ({"text": "untyped"}, []),
        ({"type": None, "text": "untyped"}, []),
        ({}, []),
    ],
)
def test_transform_audio_transcription_response_maps_one_word(
    word: dict[str, object], expected: list[dict[str, object]]
):
    assert _transform({"text": "t", "words": [word]})["words"] == expected


@pytest.mark.parametrize("words", [[], "", {}])
def test_transform_audio_transcription_response_with_empty_words_has_empty_word_list(words: object):
    assert _transform({"text": "t", "words": words})["words"] == []


@pytest.mark.parametrize(
    ("payload", "expected_text", "expected_language"),
    [
        ({}, "", "unknown"),
        ({"text": None, "language_code": None}, None, None),
        ({"text": "bonjour", "language_code": "fr"}, "bonjour", "fr"),
        ({"text": "hola", "language_code": ["es"]}, "hola", ["es"]),
    ],
)
def test_transform_audio_transcription_response_without_words_key_has_no_word_list(
    payload: dict[str, object], expected_text: str | None, expected_language: object
):
    response = _transform(payload)

    assert response.text == expected_text
    assert response["language"] == expected_language
    assert "words" not in response
    assert response._hidden_params == payload


@pytest.mark.parametrize(
    "payload",
    [
        ["sensitive", "an", "object"],
        "sensitive text",
        {"text": ["sensitive", "text"]},
        {"text": "sensitive", "words": None},
        {"text": "sensitive", "words": 7},
        {"text": "sensitive", "words": "not a list"},
        {"text": "sensitive", "words": ["not an object"]},
        {"text": "sensitive", "words": [{"type": "word", "text": "ok"}, None]},
    ],
)
def test_transform_audio_transcription_response_keeps_the_body_out_of_malformed_payload_errors(payload: object):
    raw_response = httpx.Response(200, json=payload)

    with pytest.raises(ValueError, match="Error transforming ElevenLabs response: ") as exc_info:
        ElevenLabsAudioTranscriptionConfig().transform_audio_transcription_response(raw_response=raw_response)

    assert "sensitive" not in "".join(traceback.format_exception(exc_info.value))


def test_transform_audio_transcription_response_names_the_failing_lookup_of_a_malformed_payload():
    raw_response = httpx.Response(200, json={"text": "sensitive", "words": [None]})

    with pytest.raises(ValueError, match="Error transforming ElevenLabs response: ") as exc_info:
        ElevenLabsAudioTranscriptionConfig().transform_audio_transcription_response(raw_response=raw_response)

    failing_lookup: Final = exc_info.value.__context__
    assert isinstance(failing_lookup, ValidationError)
    assert str(exc_info.value) == f"Error transforming ElevenLabs response: {failing_lookup}"
    assert "sensitive" not in "".join(traceback.format_exception(exc_info.value))


class TestElevenLabsAudioTranscription:
    @pytest.mark.usefixtures("elevenlabs_api_key_env")
    def test_elevenlabs_diarize_parameter_passthrough(self):
        """
        Test that provider-specific parameters like diarize=True get passed through
        to the ElevenLabs request form data.
        """
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = (
            '{"text": "Four score and seven years ago", "language_code": "en"}'
        )
        mock_response.json.return_value = {
            "text": "Four score and seven years ago",
            "language_code": "en",
            "words": [
                {"type": "word", "text": "Four", "start": 0.0, "end": 0.5},
                {"type": "word", "text": "score", "start": 0.5, "end": 1.0},
            ],
        }

        # Create a mock audio file
        audio_content = b"fake audio data"

        captured_request_data = {}

        def mock_post(*args, **kwargs):
            # Capture the request data for verification
            captured_request_data.update(
                {
                    "url": kwargs.get("url"),
                    "data": kwargs.get("data"),
                    "files": kwargs.get("files"),
                    "headers": kwargs.get("headers"),
                    "json": kwargs.get("json"),
                }
            )
            return mock_response

        # Mock the HTTPHandler.post method which is what actually makes the request
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        with patch.object(HTTPHandler, "post", side_effect=mock_post):
            try:
                result = litellm.transcription(
                    model="elevenlabs/scribe_v1",
                    file=audio_content,
                    diarize=True,  # This should be passed through to the form data
                    language="en",  # This should be mapped to language_code
                    temperature=0.5,  # This should also be passed through
                    custom_param="test_value",  # This should also be passed through
                )

                # Verify the request was made with correct form data
                assert "speech-to-text" in captured_request_data["url"]

                # Check that form data contains the expected parameters
                form_data = captured_request_data["data"]
                assert form_data is not None, "Form data should not be None"

                print(f"✅ Captured form data: {form_data}")

                # Check basic required parameters
                assert "model_id" in form_data, "model_id should be in form data"
                assert (
                    form_data["model_id"] == "scribe_v1"
                ), f"Expected model_id 'scribe_v1', got {form_data['model_id']}"

                # Check that diarize parameter is passed through
                assert (
                    "diarize" in form_data
                ), f"diarize should be in form data. Got: {list(form_data.keys())}"
                assert (
                    form_data["diarize"] == "true"
                ), f"Expected diarize='true', got {form_data['diarize']}"

                # Check that OpenAI language parameter is mapped correctly
                assert (
                    "language_code" in form_data
                ), "language_code should be in form data"
                assert (
                    form_data["language_code"] == "en"
                ), f"Expected language_code='en', got {form_data['language_code']}"

                # Check that temperature is passed through
                assert "temperature" in form_data, "temperature should be in form data"
                assert (
                    form_data["temperature"] == "0.5"
                ), f"Expected temperature='0.5', got {form_data['temperature']}"

                # Check that custom parameters are passed through
                assert (
                    "custom_param" in form_data
                ), "custom_param should be in form data"
                assert (
                    form_data["custom_param"] == "test_value"
                ), f"Expected custom_param='test_value', got {form_data['custom_param']}"

                # Check that files are included
                files = captured_request_data["files"]
                assert files is not None, "Files should not be None"
                assert "file" in files, "file should be in files"

                print("✅ All parameter passthrough tests passed!")

            except Exception as e:
                print(f"❌ Test failed: {e}")
                print(f"Captured request data: {captured_request_data}")
                raise


class TestElevenLabsTextToSpeechTransformation:
    @pytest.fixture(scope="class")
    def config(self) -> ElevenLabsTextToSpeechConfig:
        return ElevenLabsTextToSpeechConfig()

    def test_map_openai_params_maps_voice_and_speed(self, config):
        kwargs: Dict[str, Any] = {}
        mapped_voice, mapped_params = config.map_openai_params(
            model="eleven_multilingual_v2",
            optional_params={
                "response_format": "mp3",
                "speed": 1.25,
                "model_id": "eleven_multilingual_v2",
            },
            voice="alloy",
            kwargs=kwargs,
        )

        assert mapped_voice == config.VOICE_MAPPINGS["alloy"]
        assert mapped_params["voice_settings"]["speed"] == pytest.approx(1.25)
        assert (
            kwargs[config.ELEVENLABS_QUERY_PARAMS_KEY]["output_format"]
            == "mp3_44100_128"
        )

    def test_transform_request_and_url(self, config):
        kwargs: Dict[str, Any] = {}
        voice_id, optional_params = config.map_openai_params(
            model="eleven_multilingual_v2",
            optional_params={
                "response_format": "pcm",
                "model_id": "eleven_multilingual_v2",
                "pronunciation_dictionary_locators": [
                    {"pronunciation_dictionary_id": "dict_1"}
                ],
            },
            voice="alloy",
            kwargs=kwargs,
        )

        litellm_params: Dict[str, Any] = {
            config.ELEVENLABS_VOICE_ID_KEY: voice_id,
            config.ELEVENLABS_QUERY_PARAMS_KEY: kwargs[
                config.ELEVENLABS_QUERY_PARAMS_KEY
            ],
        }

        headers = config.validate_environment(
            headers={}, model="eleven_multilingual_v2", api_key="test-key"
        )

        request_data = config.transform_text_to_speech_request(
            model="eleven_multilingual_v2",
            input="Hello world",
            voice=voice_id,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        assert request_data["dict_body"]["text"] == "Hello world"
        assert request_data["dict_body"]["model_id"] == "eleven_multilingual_v2"
        assert request_data["dict_body"]["pronunciation_dictionary_locators"] == [
            {"pronunciation_dictionary_id": "dict_1"}
        ]

        url = config.get_complete_url(
            model="eleven_multilingual_v2",
            api_base=None,
            litellm_params=litellm_params,
        )

        assert voice_id in url
        assert "output_format=pcm_44100" in url
