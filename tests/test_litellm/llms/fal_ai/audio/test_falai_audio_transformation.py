from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.fal_ai.audio.transformation import FalAIAudioConfig
from litellm.llms.fal_ai.utils import normalize_fal_model_id

ELEVEN_V3 = "fal_ai/fal-ai/elevenlabs/tts/eleven-v3"
ELEVEN_V3_ID = "fal-ai/elevenlabs/tts/eleven-v3"
FAL_API_BASE = "https://queue.fal.run"
SUBMIT_PAYLOAD = {
    "request_id": "test-rid",
    "status_url": f"{FAL_API_BASE}/fal-ai/elevenlabs/requests/test-rid/status",
    "response_url": f"{FAL_API_BASE}/fal-ai/elevenlabs/requests/test-rid",
}
RESULT_PAYLOAD = {
    "audio": {
        "url": "https://v3b.fal.media/files/x/output.mp3",
        "content_type": "audio/mpeg",
    }
}


def _resp(json_payload=None, content=b"", status_code=200, request=None):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_payload
    resp.content = content
    resp.request = request
    resp.raise_for_status = MagicMock()
    return resp


class TestFalAIAudioBasics:
    def setup_method(self):
        self.config = FalAIAudioConfig()

    def test_validate_environment_uses_fal_ai_api_key(self, monkeypatch):
        monkeypatch.setenv("FAL_AI_API_KEY", "key-123")
        headers = self.config.validate_environment(headers={}, model=ELEVEN_V3)
        assert headers["Authorization"] == "Key key-123"
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_falls_back_to_fal_key(self, monkeypatch):
        monkeypatch.delenv("FAL_AI_API_KEY", raising=False)
        monkeypatch.setenv("FAL_KEY", "fallback")
        headers = self.config.validate_environment(headers={}, model=ELEVEN_V3)
        assert headers["Authorization"] == "Key fallback"

    def test_validate_environment_raises_when_missing(self, monkeypatch):
        monkeypatch.delenv("FAL_AI_API_KEY", raising=False)
        monkeypatch.delenv("FAL_KEY", raising=False)
        with pytest.raises(ValueError, match="fal.ai API key is required"):
            self.config.validate_environment(headers={}, model=ELEVEN_V3)

    def test_get_complete_url_default(self, monkeypatch):
        monkeypatch.delenv("FAL_AI_API_BASE", raising=False)
        assert self.config.get_complete_url(
            model=ELEVEN_V3, api_base=None, litellm_params={}
        ) == f"{FAL_API_BASE}/{ELEVEN_V3_ID}"

    def test_get_complete_url_strips_trailing_slash(self):
        assert self.config.get_complete_url(
            model=ELEVEN_V3,
            api_base="https://custom.example.com/",
            litellm_params={},
        ) == f"https://custom.example.com/{ELEVEN_V3_ID}"

    def test_normalize_model_id_strips_prefix(self):
        assert normalize_fal_model_id(ELEVEN_V3) == ELEVEN_V3_ID
        assert normalize_fal_model_id(ELEVEN_V3_ID) == ELEVEN_V3_ID

    def test_normalize_model_id_rejects_empty(self):
        with pytest.raises(ValueError, match="empty after stripping"):
            normalize_fal_model_id("fal_ai/")

    def test_transform_request_carries_text_voice_and_extras(self):
        request = self.config.transform_text_to_speech_request(
            model=ELEVEN_V3,
            input="hello",
            voice="Aria",
            optional_params={
                "stability": 0.5,
                "extra_body": {"language_code": "en"},
                "response_format": "mp3",
            },
            litellm_params={},
            headers={},
        )
        body = request["dict_body"]
        assert body["text"] == "hello"
        assert body["prompt"] == "hello"
        assert body["voice"] == "Aria"
        assert body["stability"] == 0.5
        assert body["language_code"] == "en"
        assert "response_format" not in body
        assert "extra_body" not in body

    def test_extract_audio_url_supports_known_shapes(self):
        assert self.config._extract_audio_url({"audio": {"url": "x"}}) == "x"
        assert self.config._extract_audio_url({"audio_url": "y"}) == "y"
        assert self.config._extract_audio_url({"audio_file": {"url": "z"}}) == "z"

    def test_extract_audio_url_raises_when_missing(self):
        with pytest.raises(ValueError, match="missing audio url"):
            self.config._extract_audio_url({"other": "shape"})

    def test_extract_audio_url_raises_on_error_payload(self):
        with pytest.raises(ValueError, match="audio generation failed"):
            self.config._extract_audio_url({"error": "boom"})


class TestFalAIAudioResponsePolling:
    def setup_method(self):
        self.config = FalAIAudioConfig()

    def _submit_response(self):
        request = httpx.Request(
            "POST",
            f"{FAL_API_BASE}/{ELEVEN_V3_ID}",
            headers={"Authorization": "Key key-123"},
        )
        return _resp(json_payload=SUBMIT_PAYLOAD, request=request)

    def test_response_polls_until_complete_then_downloads(self, monkeypatch):
        binary_payload = b"audio-bytes-12345"
        binary_resp = _resp(content=binary_payload)
        result_resp = _resp(json_payload=RESULT_PAYLOAD)
        in_progress = _resp(json_payload={"status": "IN_PROGRESS"})
        completed = _resp(json_payload={"status": "COMPLETED"})

        client = MagicMock()
        client.get.side_effect = [in_progress, completed, result_resp, binary_resp]

        monkeypatch.setattr(
            "litellm.llms.fal_ai.audio.transformation._get_httpx_client",
            lambda: client,
        )
        monkeypatch.setattr(
            "litellm.llms.fal_ai.audio.transformation.time.sleep", lambda _s: None
        )

        out = self.config.transform_text_to_speech_response(
            model=ELEVEN_V3,
            raw_response=self._submit_response(),
            logging_obj=MagicMock(),
        )

        assert out.response.content == binary_payload
        get_urls = [c.kwargs.get("url") or c.args[0] for c in client.get.call_args_list]
        assert get_urls[0] == SUBMIT_PAYLOAD["status_url"]
        assert get_urls[1] == SUBMIT_PAYLOAD["status_url"]
        assert get_urls[2] == SUBMIT_PAYLOAD["response_url"]
        assert get_urls[3] == RESULT_PAYLOAD["audio"]["url"]

    def test_response_forwards_authorization_to_poll(self, monkeypatch):
        client = MagicMock()
        client.get.side_effect = [
            _resp(json_payload={"status": "COMPLETED"}),
            _resp(json_payload=RESULT_PAYLOAD),
            _resp(content=b"audio"),
        ]
        monkeypatch.setattr(
            "litellm.llms.fal_ai.audio.transformation._get_httpx_client",
            lambda: client,
        )
        monkeypatch.setattr(
            "litellm.llms.fal_ai.audio.transformation.time.sleep", lambda _s: None
        )

        self.config.transform_text_to_speech_response(
            model=ELEVEN_V3,
            raw_response=self._submit_response(),
            logging_obj=MagicMock(),
        )

        status_call = client.get.call_args_list[0]
        assert status_call.kwargs["headers"]["Authorization"] == "Key key-123"

    def test_response_raises_on_failed_status(self, monkeypatch):
        client = MagicMock()
        client.get.side_effect = [_resp(json_payload={"status": "FAILED"})]
        monkeypatch.setattr(
            "litellm.llms.fal_ai.audio.transformation._get_httpx_client",
            lambda: client,
        )
        monkeypatch.setattr(
            "litellm.llms.fal_ai.audio.transformation.time.sleep", lambda _s: None
        )

        with pytest.raises(RuntimeError, match="status=FAILED"):
            self.config.transform_text_to_speech_response(
                model=ELEVEN_V3,
                raw_response=self._submit_response(),
                logging_obj=MagicMock(),
            )

    def test_response_raises_when_submit_payload_missing_urls(self):
        request = httpx.Request("POST", f"{FAL_API_BASE}/{ELEVEN_V3_ID}")
        bad_submit = _resp(json_payload={"request_id": "x"}, request=request)
        with pytest.raises(ValueError, match="missing status_url/response_url"):
            self.config.transform_text_to_speech_response(
                model=ELEVEN_V3,
                raw_response=bad_submit,
                logging_obj=MagicMock(),
            )

    def test_response_stashes_decoded_duration(self, monkeypatch):
        client = MagicMock()
        client.get.side_effect = [
            _resp(json_payload={"status": "COMPLETED"}),
            _resp(json_payload=RESULT_PAYLOAD),
            _resp(content=b"RIFFfake-wav-bytes"),
        ]
        monkeypatch.setattr(
            "litellm.llms.fal_ai.audio.transformation._get_httpx_client",
            lambda: client,
        )
        monkeypatch.setattr(
            "litellm.llms.fal_ai.audio.transformation.time.sleep", lambda _s: None
        )
        monkeypatch.setattr(
            "litellm.llms.fal_ai.audio.transformation.calculate_request_duration",
            lambda _content: 12.5,
        )

        out = self.config.transform_text_to_speech_response(
            model=ELEVEN_V3,
            raw_response=self._submit_response(),
            logging_obj=MagicMock(),
        )
        assert out._hidden_params["audio_output_duration"] == 12.5

    def test_response_omits_duration_when_undeterminable(self, monkeypatch):
        client = MagicMock()
        client.get.side_effect = [
            _resp(json_payload={"status": "COMPLETED"}),
            _resp(json_payload=RESULT_PAYLOAD),
            _resp(content=b"audio"),
        ]
        monkeypatch.setattr(
            "litellm.llms.fal_ai.audio.transformation._get_httpx_client",
            lambda: client,
        )
        monkeypatch.setattr(
            "litellm.llms.fal_ai.audio.transformation.time.sleep", lambda _s: None
        )
        monkeypatch.setattr(
            "litellm.llms.fal_ai.audio.transformation.calculate_request_duration",
            lambda _content: None,
        )

        out = self.config.transform_text_to_speech_response(
            model=ELEVEN_V3,
            raw_response=self._submit_response(),
            logging_obj=MagicMock(),
        )
        assert "audio_output_duration" not in out._hidden_params


CHAR_PRICED_MODELS = [
    "fal_ai/fal-ai/elevenlabs/tts/eleven-v3",
    "fal_ai/fal-ai/elevenlabs/tts/turbo-v2.5",
    "fal_ai/fal-ai/elevenlabs/tts/multilingual-v2",
    "fal_ai/fal-ai/minimax/speech-2.8-hd",
    "fal_ai/fal-ai/minimax/speech-2.8-turbo",
    "fal_ai/fal-ai/kokoro/american-english",
    "fal_ai/fal-ai/orpheus-tts",
    "fal_ai/fal-ai/dia-tts",
    "fal_ai/fal-ai/inworld-tts",
]
PER_SECOND_MODELS = [
    "fal_ai/fal-ai/elevenlabs/music",
    "fal_ai/fal-ai/elevenlabs/sound-effects/v2",
    "fal_ai/fal-ai/mmaudio-v2/text-to-audio",
]
FLAT_MODELS = [
    "fal_ai/fal-ai/lyria3/pro",
    "fal_ai/fal-ai/minimax-music/v2.6",
    "fal_ai/fal-ai/stable-audio-25/text-to-audio",
    "fal_ai/fal-ai/stable-audio-3/medium/text-to-audio",
]


def _backup_entry(model_id):
    from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

    entry = GetModelCostMap.load_local_model_cost_map().get(model_id)
    assert entry is not None, f"{model_id} missing from local backup model cost map"
    assert entry["litellm_provider"] == "fal_ai"
    assert entry["mode"] == "audio_speech"
    assert "/v1/audio/speech" in entry["supported_endpoints"]
    assert entry["supported_output_modalities"] == ["audio"]
    return entry


@pytest.mark.parametrize("model_id", CHAR_PRICED_MODELS)
def test_tts_models_priced_per_character(model_id):
    entry = _backup_entry(model_id)
    assert isinstance(entry["input_cost_per_character"], (int, float))
    assert "output_cost_per_second" not in entry


@pytest.mark.parametrize("model_id", PER_SECOND_MODELS)
def test_music_models_priced_per_second(model_id):
    entry = _backup_entry(model_id)
    assert isinstance(entry["output_cost_per_second"], (int, float))


@pytest.mark.parametrize("model_id", FLAT_MODELS)
def test_music_models_priced_flat_per_audio(model_id):
    entry = _backup_entry(model_id)
    assert isinstance(entry["output_cost_per_audio"], (int, float))


def test_provider_config_manager_returns_fal_ai_audio_config():
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_text_to_speech_config(
        model=ELEVEN_V3, provider=LlmProviders.FAL_AI
    )
    assert isinstance(config, FalAIAudioConfig)
