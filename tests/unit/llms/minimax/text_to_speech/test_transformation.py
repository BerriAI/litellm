import asyncio, base64, importlib, litellm
from typing import Final
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

from litellm.llms.minimax.text_to_speech.transformation import MinimaxException, MinimaxTextToSpeechConfig
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm import speech
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome

_AUDIO: Final = b"ID3\x04minimax-audio"
_REQUEST: Final = httpx.Request("POST", "https://api.minimax.io/v1/t2a_v2")


def _transform(payload: object, status_code: int = 200) -> HttpxBinaryResponseContent:
    return MinimaxTextToSpeechConfig().transform_text_to_speech_response(
        model="speech-02-hd",
        raw_response=httpx.Response(
            status_code,
            json=payload,
            request=_REQUEST,
            headers={"content-encoding": "identity", "x-trace": "abc"},
        ),
        logging_obj=Mock(),
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"data": {"audio": _AUDIO.hex()}, "status": 0, "extra_info": {"audio_length": 5}},
        {"data": {"audio": _AUDIO.hex(), "audio_url": ""}},
        {"data": {"audio": ""}, "audio_file": _AUDIO.hex()},
        {"data": {"audio": None}, "audio_file": base64.b64encode(_AUDIO).decode()},
        {"base_resp": {"status_code": 0}, "audio_file": base64.b64encode(_AUDIO).decode()},
    ],
)
def test_transform_text_to_speech_response_decodes_hex_or_base64_audio(payload: dict[str, object]):
    response = _transform(payload).response

    assert response.status_code == 200
    assert response.content == _AUDIO
    assert response.headers["content-length"] == str(len(_AUDIO))
    assert response.headers["x-trace"] == "abc"
    assert "content-encoding" not in response.headers


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"status": 2, "ced": "invalid api key"}, "invalid api key"),
        ({"status": 2}, "Unknown error"),
        ({"status": 1004, "ced": ""}, "API returned status 1004"),
        ({"status": "failed", "ced": None, "data": {"audio": _AUDIO.hex()}}, "API returned status failed"),
    ],
)
def test_transform_text_to_speech_response_reports_api_status_errors(payload: dict[str, object], detail: str):
    with pytest.raises(MinimaxException) as exc_info:
        _transform(payload, status_code=401)

    assert exc_info.value.message == f"MiniMax TTS error: {detail}"
    assert exc_info.value.status_code == 401


def test_transform_text_to_speech_response_refuses_url_output():
    with pytest.raises(MinimaxException) as exc_info:
        _transform({"data": {"audio_url": "https://cdn.example/a.mp3", "audio": _AUDIO.hex()}})

    assert exc_info.value.message == (
        "URL output format is not yet supported. Use 'hex' format or fetch from URL: https://cdn.example/a.mp3"
    )
    assert exc_info.value.status_code == 500


@pytest.mark.parametrize(
    ("payload", "keys"),
    [
        ({}, []),
        ({"data": {}, "status": 0}, ["data", "status"]),
        ({"data": {"audio": ""}, "audio_file": ""}, ["data", "audio_file"]),
        ({"data": {"audio": None}, "audio_file": None}, ["data", "audio_file"]),
        ({"audio_file": 0, "data": {"audio": []}}, ["audio_file", "data"]),
    ],
)
def test_transform_text_to_speech_response_without_audio_lists_the_response_keys(
    payload: dict[str, object], keys: list[str]
):
    with pytest.raises(MinimaxException) as exc_info:
        _transform(payload)

    assert exc_info.value.message == f"No audio data in MiniMax response. Response keys: {keys}"
    assert exc_info.value.status_code == 500


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "an", "object"],
        "not an object",
        {"data": None},
        {"data": "not an object"},
        {"data": ["not", "an", "object"]},
        {"data": {"audio": 7}},
        {"data": {"audio": ["49", "44"]}},
        {"audio_file": {"hex": "4944"}},
    ],
)
def test_transform_text_to_speech_response_wraps_malformed_payloads_without_echoing_them(payload: object):
    with pytest.raises(MinimaxException) as exc_info:
        _transform(payload)

    assert exc_info.value.message.startswith("Error processing MiniMax response: ")
    assert "input_value" not in exc_info.value.message
    assert exc_info.value.status_code == 500


def test_transform_text_to_speech_response_reports_undecodable_audio():
    with pytest.raises(MinimaxException) as exc_info:
        _transform({"data": {"audio": "zzz"}})

    assert exc_info.value.message.startswith("Failed to decode audio data: ")
    assert exc_info.value.status_code == 500


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)

@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
def setup_and_teardown(event_loop):
    import litellm

    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in _SCALAR_DEFAULTS:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    importlib.reload(litellm)
    asyncio.set_event_loop(event_loop)
    yield
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)
    pending = asyncio.all_tasks(event_loop)
    for task in pending:
        task.cancel()
    if pending:
        event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

_SCALAR_DEFAULTS = {
    "num_retries": getattr(litellm, "num_retries", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
    "cohere_key": getattr(litellm, "cohere_key", None),
}

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
class TestMinimaxTextToSpeechConfig:
    """Test MiniMax TTS configuration and parameter mapping"""

    def test_get_supported_openai_params(self):
        """Test that supported OpenAI params are correctly defined"""
        config = MinimaxTextToSpeechConfig()
        supported_params = config.get_supported_openai_params("speech-2.6-hd")

        assert "voice" in supported_params
        assert "response_format" in supported_params
        assert "speed" in supported_params

    def test_voice_mapping(self):
        """Test OpenAI voice to MiniMax voice_id mapping"""
        config = MinimaxTextToSpeechConfig()

        # Test OpenAI voice mappings
        assert config._extract_voice_id("alloy") == "male-qn-qingse"
        assert config._extract_voice_id("echo") == "male-qn-jingying"
        assert config._extract_voice_id("nova") == "female-yujie"

        # Test custom voice passthrough
        assert config._extract_voice_id("custom-voice-id") == "custom-voice-id"

    def test_format_mapping(self):
        """Test response format mapping"""
        config = MinimaxTextToSpeechConfig()

        assert config.FORMAT_MAPPINGS["mp3"] == "mp3"
        assert config.FORMAT_MAPPINGS["pcm"] == "pcm"
        assert config.FORMAT_MAPPINGS["wav"] == "wav"
        assert config.FORMAT_MAPPINGS["flac"] == "flac"

    def test_map_openai_params_basic(self):
        """Test basic parameter mapping from OpenAI to MiniMax format"""
        config = MinimaxTextToSpeechConfig()

        optional_params = {
            "response_format": "mp3",
            "speed": 1.5,
        }

        voice, mapped_params = config.map_openai_params(
            model="speech-2.6-hd",
            optional_params=optional_params,
            voice="alloy",
        )

        assert voice == "male-qn-qingse"
        assert mapped_params["format"] == "mp3"
        assert mapped_params["speed"] == 1.5
        assert mapped_params["voice_id"] == "male-qn-qingse"

    def test_map_openai_params_speed_clamping(self):
        """Test that speed is clamped to MiniMax's supported range"""
        config = MinimaxTextToSpeechConfig()

        # Test speed too high
        optional_params = {"speed": 5.0}
        _, mapped_params = config.map_openai_params(
            model="speech-2.6-hd",
            optional_params=optional_params,
            voice="alloy",
        )
        assert mapped_params["speed"] == 2.0  # Clamped to max

        # Test speed too low
        optional_params = {"speed": 0.1}
        _, mapped_params = config.map_openai_params(
            model="speech-2.6-hd",
            optional_params=optional_params,
            voice="alloy",
        )
        assert mapped_params["speed"] == 0.5  # Clamped to min

    def test_map_openai_params_with_extra_body(self):
        """Test that extra_body parameters are passed through"""
        config = MinimaxTextToSpeechConfig()

        optional_params = {
            "extra_body": {
                "vol": 1.5,
                "pitch": 2,
                "sample_rate": 24000,
            }
        }

        _, mapped_params = config.map_openai_params(
            model="speech-2.6-hd",
            optional_params=optional_params,
            voice="alloy",
        )

        assert mapped_params["vol"] == 1.5
        assert mapped_params["pitch"] == 2
        assert mapped_params["sample_rate"] == 24000

    def test_validate_environment_with_api_key(self):
        """Test environment validation with API key"""
        config = MinimaxTextToSpeechConfig()
        headers = {}

        result_headers = config.validate_environment(
            headers=headers,
            model="speech-2.6-hd",
            api_key="test-api-key",
        )

        assert "Authorization" in result_headers
        assert result_headers["Authorization"] == "Bearer test-api-key"
        assert result_headers["Content-Type"] == "application/json"

    def test_validate_environment_missing_api_key(self):
        """Test that validation fails without API key"""
        config = MinimaxTextToSpeechConfig()
        headers = {}

        # Mock both litellm.api_key and get_secret_str to return None
        import litellm

        original_api_key = litellm.api_key
        try:
            litellm.api_key = None
            with patch(
                "litellm.llms.minimax.text_to_speech.transformation.get_secret_str",
                return_value=None,
            ):
                with pytest.raises(ValueError, match="MiniMax API key is required"):
                    config.validate_environment(
                        headers=headers,
                        model="speech-2.6-hd",
                        api_key=None,
                    )
        finally:
            litellm.api_key = original_api_key

    def test_transform_text_to_speech_request(self):
        """Test request transformation to MiniMax format"""
        config = MinimaxTextToSpeechConfig()

        optional_params = {
            "voice_id": "male-qn-qingse",
            "speed": 1.2,
            "format": "mp3",
            "vol": 1.0,
            "pitch": 0,
            "sample_rate": 32000,
            "bitrate": 128000,
            "channel": 1,
        }

        result = config.transform_text_to_speech_request(
            model="speech-2.6-hd",
            input="Hello, world!",
            voice="male-qn-qingse",
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert "dict_body" in result
        body = result["dict_body"]

        assert body["model"] == "speech-2.6-hd"
        assert body["text"] == "Hello, world!"
        assert body["stream"] is False
        assert body["voice_setting"]["voice_id"] == "male-qn-qingse"
        assert body["voice_setting"]["speed"] == 1.2
        assert body["audio_setting"]["format"] == "mp3"
        assert body["audio_setting"]["sample_rate"] == 32000

    def test_get_complete_url(self):
        """Test URL construction"""
        config = MinimaxTextToSpeechConfig()

        url = config.get_complete_url(
            model="speech-2.6-hd",
            api_base=None,
            litellm_params={},
        )

        assert url == "https://api.minimax.io/v1/t2a_v2"

    def test_get_complete_url_custom_base(self):
        """Test URL construction with custom API base"""
        config = MinimaxTextToSpeechConfig()

        url = config.get_complete_url(
            model="speech-2.6-hd",
            api_base="https://custom.api.com",
            litellm_params={},
        )

        assert url == "https://custom.api.com/v1/t2a_v2"

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
class TestMinimaxSpeechIntegration:
    """Integration tests for MiniMax TTS via litellm.speech()"""

    def test_speech_mock_response(self):
        """Test speech synthesis with mocked response"""

        # Create mock audio data (hex-encoded as MiniMax returns)
        mock_audio_bytes = b"fake audio data for testing"
        mock_audio_hex = mock_audio_bytes.hex()

        mock_response_json = {
            "data": {"audio": mock_audio_hex, "status": 0, "ced": ""},
            "extra_info": {},
        }

        with patch("litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.text_to_speech_handler") as mock_tts:
            # Create a mock httpx.Response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.json.return_value = mock_response_json
            mock_response.content = mock_audio_bytes

            # Mock the response wrapper
            from litellm.types.llms.openai import HttpxBinaryResponseContent

            mock_binary_response = HttpxBinaryResponseContent(mock_response)
            mock_tts.return_value = mock_binary_response

            # This would normally make a real API call
            # but we're mocking it for testing
            response = speech(
                model="minimax/speech-2.6-hd",
                voice="alloy",
                input="Test input",
                api_key="test-key",
            )

            # Verify the mock was called
            assert mock_tts.called

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
class TestMinimaxProviderRegistration:
    """Test that MiniMax is properly registered as a provider"""

    def test_minimax_in_llm_providers(self):
        """Test that MINIMAX is in LlmProviders enum"""
        from litellm.types.utils import LlmProviders

        assert hasattr(LlmProviders, "MINIMAX")
        assert LlmProviders.MINIMAX.value == "minimax"

    def test_minimax_in_provider_list(self):
        """Test that minimax is in the provider list"""
        assert litellm.LlmProviders.MINIMAX in litellm.provider_list

    def test_get_provider_text_to_speech_config(self):
        """Test that MiniMax TTS config can be retrieved"""
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_text_to_speech_config(
            model="speech-2.6-hd",
            provider=litellm.LlmProviders.MINIMAX,
        )

        assert config is not None
        assert isinstance(config, MinimaxTextToSpeechConfig)

    def test_get_llm_provider_minimax(self):
        """Test that get_llm_provider correctly identifies MiniMax models"""
        from litellm import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(model="minimax/speech-2.6-hd")

        assert model == "speech-2.6-hd"
        assert provider == "minimax"

if __name__ == "__main__":
    # Run basic tests
    test_config = TestMinimaxTextToSpeechConfig()
    test_config.test_get_supported_openai_params()
    test_config.test_voice_mapping()
    test_config.test_format_mapping()
    test_config.test_map_openai_params_basic()
    test_config.test_map_openai_params_speed_clamping()
    test_config.test_transform_text_to_speech_request()
    test_config.test_get_complete_url()

    test_registration = TestMinimaxProviderRegistration()
    test_registration.test_minimax_in_llm_providers()
    test_registration.test_minimax_in_provider_list()
    test_registration.test_get_provider_text_to_speech_config()
    test_registration.test_get_llm_provider_minimax()

    print("All basic tests passed!")
