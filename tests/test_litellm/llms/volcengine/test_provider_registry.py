import asyncio
import json
from pathlib import Path

import litellm
import pytest
from litellm.llms.volcengine.audio_transcription.transformation import (
    VolcEngineAudioTranscriptionConfig,
)
from litellm.llms.volcengine.realtime.transformation import VolcEngineRealtimeConfig
from litellm.llms.volcengine.text_to_speech.transformation import (
    VolcEngineTextToSpeechConfig,
)
from litellm.realtime_api import main as realtime_main
from litellm.utils import ProviderConfigManager


class FakeLogging:
    litellm_trace_id = "trace-id"

    def update_from_kwargs(self, **kwargs):
        pass


def test_provider_registry_returns_volcengine_audio_configs():
    transcription_config = (
        ProviderConfigManager.get_provider_audio_transcription_config(
            model="volcengine/volc.seedasr.sauc.duration",
            provider=litellm.LlmProviders.VOLCENGINE,
        )
    )
    speech_config = ProviderConfigManager.get_provider_text_to_speech_config(
        model="volcengine/seed-tts-2.0",
        provider=litellm.LlmProviders.VOLCENGINE,
    )
    realtime_config = ProviderConfigManager.get_provider_realtime_config(
        model="volcengine/volc.speech.dialog",
        provider=litellm.LlmProviders.VOLCENGINE,
    )

    assert isinstance(transcription_config, VolcEngineAudioTranscriptionConfig)
    assert isinstance(speech_config, VolcEngineTextToSpeechConfig)
    assert isinstance(realtime_config, VolcEngineRealtimeConfig)


def test_volcengine_realtime_uses_dynamic_provider_params(monkeypatch):
    captured_kwargs = {}

    async def mock_async_realtime(**kwargs):
        captured_kwargs.update(kwargs)

    def mock_get_llm_provider(model, api_base, api_key):
        return (
            "volc.speech.dialog",
            "volcengine",
            "speech-key",
            "wss://example.test/realtime",
        )

    monkeypatch.setattr(realtime_main, "get_llm_provider", mock_get_llm_provider)
    monkeypatch.setattr(
        realtime_main.base_llm_http_handler,
        "async_realtime",
        mock_async_realtime,
    )

    asyncio.run(
        realtime_main._arealtime(
            model="volcengine/volc.speech.dialog",
            websocket=object(),
            litellm_logging_obj=FakeLogging(),
        )
    )

    assert captured_kwargs["api_base"] == "wss://example.test/realtime"
    assert captured_kwargs["api_key"] == "speech-key"
    assert isinstance(captured_kwargs["provider_config"], VolcEngineRealtimeConfig)


def test_public_audio_apis_dispatch_to_volcengine_provider():
    with pytest.raises(Exception, match="response_format=json"):
        litellm.transcription(
            model="volcengine/volc.seedasr.sauc.duration",
            file=b"fake-audio",
            api_key="speech-api-key",
            response_format="text",
        )

    with pytest.raises(Exception, match="response_format=pcm or wav"):
        litellm.speech(
            model="volcengine/seed-tts-2.0",
            input="hello",
            voice="alloy",
            api_key="speech-api-key",
            response_format="mp3",
        )


def test_volcengine_audio_configs_only_advertise_supported_openai_params():
    assert VolcEngineAudioTranscriptionConfig().get_supported_openai_params(
        "volc.seedasr.sauc.duration"
    ) == ["language", "response_format"]
    assert VolcEngineTextToSpeechConfig().get_supported_openai_params(
        "seed-tts-2.0"
    ) == ["voice", "response_format"]


def test_volcengine_model_cost_metadata_is_complete():
    repo_root = Path(__file__).parents[4]
    model_cost = json.loads(
        (repo_root / "model_prices_and_context_window.json").read_text()
    )
    bundled_model_cost = json.loads(
        (repo_root / "litellm/model_prices_and_context_window_backup.json").read_text()
    )

    for key in (
        "volcengine/volc.bigasr.sauc.duration",
        "volcengine/volc.bigasr.sauc.concurrent",
        "volcengine/volc.seedasr.sauc.duration",
        "volcengine/volc.seedasr.sauc.concurrent",
        "volcengine/volc.speech.dialog",
        "volcengine/seed-tts-2.0",
        "volcengine/seed-tts-2.0-standard",
        "volcengine/seed-tts-2.0-expressive",
        "volcengine/seed-tts-1.0",
        "volcengine/seed-tts-1.0-concurr",
        "volcengine/seed-icl-2.0",
        "volcengine/seed-icl-1.0",
        "volcengine/seed-icl-1.0-concurr",
    ):
        assert bundled_model_cost[key] == model_cost[key]

    asr = model_cost["volcengine/volc.bigasr.sauc.duration"]
    assert asr["litellm_provider"] == "volcengine"
    assert asr["mode"] == "audio_transcription"
    assert asr["input_cost_per_second"] > 0
    assert asr["output_cost_per_second"] == 0.0
    assert asr["supported_endpoints"] == ["/v1/audio/transcriptions"]
    assert asr["supports_audio_input"] is True
    assert asr["supports_audio_output"] is False
    assert asr["metadata"]["resource_id"] == "volc.bigasr.sauc.duration"
    assert asr["metadata"]["sample_rate_hz"] == 16000
    assert asr["source"].startswith("https://www.volcengine.com/docs/")
    assert (
        model_cost["volcengine/volc.bigasr.sauc.concurrent"]["metadata"]["resource_id"]
        == "volc.bigasr.sauc.concurrent"
    )
    assert (
        model_cost["volcengine/volc.seedasr.sauc.duration"]["metadata"]["resource_id"]
        == "volc.seedasr.sauc.duration"
    )
    assert (
        model_cost["volcengine/volc.seedasr.sauc.concurrent"]["metadata"]["resource_id"]
        == "volc.seedasr.sauc.concurrent"
    )

    realtime = model_cost["volcengine/volc.speech.dialog"]
    assert realtime["litellm_provider"] == "volcengine"
    assert realtime["mode"] == "realtime"
    assert realtime["supported_endpoints"] == ["/v1/realtime"]
    assert realtime["supports_audio_input"] is True
    assert realtime["supports_audio_output"] is True
    assert realtime["metadata"]["resource_id"] == "volc.speech.dialog"
    assert realtime["metadata"]["input_audio_sample_rate_hz"] == 16000
    assert realtime["metadata"]["output_audio_sample_rate_hz"] == 24000

    tts = model_cost["volcengine/seed-tts-2.0"]
    assert tts["litellm_provider"] == "volcengine"
    assert tts["health_check_voice"] == "zh_female_vv_uranus_bigtts"
    assert tts["mode"] == "audio_speech"
    assert tts["input_cost_per_character"] > 0
    assert tts["supported_endpoints"] == ["/v1/audio/speech"]
    assert tts["supports_audio_input"] is False
    assert tts["supports_audio_output"] is True
    assert tts["metadata"]["resource_id"] == "seed-tts-2.0"
    assert tts["metadata"]["default_voice"] == "zh_female_vv_uranus_bigtts"
    assert tts["metadata"]["supported_response_formats"] == ["pcm", "wav"]
    assert tts["source"].startswith("https://www.volcengine.com/docs/")
    expressive = model_cost["volcengine/seed-tts-2.0-expressive"]
    assert expressive["metadata"]["resource_id"] == "seed-tts-2.0"
    assert expressive["metadata"]["request_model"] == "seed-tts-2.0-expressive"
    assert model_cost["volcengine/seed-tts-1.0"]["metadata"]["resource_id"] == (
        "seed-tts-1.0"
    )
    assert model_cost["volcengine/seed-icl-2.0"]["metadata"]["resource_id"] == (
        "seed-icl-2.0"
    )


def test_volcengine_endpoint_support_metadata_is_complete():
    repo_root = Path(__file__).parents[4]
    root_support = json.loads(
        (repo_root / "provider_endpoints_support.json").read_text()
    )
    backup_support = json.loads(
        (repo_root / "litellm/provider_endpoints_support_backup.json").read_text()
    )

    for support in (root_support, backup_support):
        endpoints = support["providers"]["volcengine"]["endpoints"]
        assert endpoints["audio_transcriptions"] is True
        assert endpoints["audio_speech"] is True
        assert endpoints["realtime"] is True
