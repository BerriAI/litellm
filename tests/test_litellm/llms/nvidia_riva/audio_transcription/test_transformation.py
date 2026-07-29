"""
Unit tests for NvidiaRivaAudioTranscriptionConfig.

These tests do not require ``nvidia-riva-client`` or any audio libs to be
installed; the transformation layer is intentionally pure-Python on dicts.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
)
from litellm.llms.nvidia_riva.audio_transcription.transformation import (
    NvidiaRivaAudioTranscriptionConfig,
)
from litellm.llms.nvidia_riva.common_utils import NvidiaRivaException


@pytest.fixture
def cfg():
    return NvidiaRivaAudioTranscriptionConfig()


def test_supported_openai_params(cfg):
    params = cfg.get_supported_openai_params(model="nvidia/parakeet-ctc-1_1b-asr")
    assert "language" in params
    assert "response_format" in params
    assert "timestamp_granularities" in params


def test_map_language_normalizes_bare_codes(cfg):
    out = cfg.map_openai_params(
        non_default_params={"language": "en"},
        optional_params={},
        model="m",
        drop_params=False,
    )
    assert out["language_code"] == "en-US"


def test_map_language_passes_through_bcp47(cfg):
    out = cfg.map_openai_params(
        non_default_params={"language": "de-DE"},
        optional_params={},
        model="m",
        drop_params=False,
    )
    assert out["language_code"] == "de-DE"


def test_map_language_es_defaults_to_castilian_spain(cfg):
    """
    Bare ``es`` is ISO-639 Spanish; in BCP-47 it conventionally resolves to
    es-ES (Castilian / Spain), not es-US. Routing every Spanish caller to a
    US-tuned Riva model would silently degrade accuracy.
    """
    out = cfg.map_openai_params(
        non_default_params={"language": "es"},
        optional_params={},
        model="m",
        drop_params=False,
    )
    assert out["language_code"] == "es-ES"


def test_map_timestamp_granularities_word_enables_word_offsets(cfg):
    out = cfg.map_openai_params(
        non_default_params={"timestamp_granularities": ["word"]},
        optional_params={},
        model="m",
        drop_params=False,
    )
    assert out["enable_word_time_offsets"] is True
    assert out["timestamp_granularities"] == ["word"]


def test_map_timestamp_granularities_segment_only_does_not_enable_word_offsets(cfg):
    out = cfg.map_openai_params(
        non_default_params={"timestamp_granularities": ["segment"]},
        optional_params={},
        model="m",
        drop_params=False,
    )
    assert "enable_word_time_offsets" not in out


def test_transform_request_builds_recognition_config(cfg):
    result = cfg.transform_audio_transcription_request(
        model="nvidia/parakeet-ctc-1_1b-asr",
        audio_file=b"fake-audio",
        optional_params={
            "language_code": "en-US",
            "enable_word_time_offsets": True,
            "nvcf_function_id": "abc-123",
            "use_ssl": True,
            "riva_model_name": "parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer",
        },
        litellm_params={
            "api_base": "grpc.nvcf.nvidia.com:443",
            "api_key": "nvapi-xxx",
        },
    )

    assert isinstance(result, AudioTranscriptionRequestData)
    payload = result.data
    assert payload["recognition_config"]["language_code"] == "en-US"
    assert payload["recognition_config"]["sample_rate_hertz"] == 16000
    assert payload["recognition_config"]["audio_channel_count"] == 1
    assert payload["recognition_config"]["encoding"] == "LINEAR_PCM"
    assert payload["recognition_config"]["enable_word_time_offsets"] is True
    assert (
        payload["recognition_config"]["model"]
        == "parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer"
    )
    assert "audio_file" not in payload
    assert "auth" not in payload


def test_transform_request_default_riva_model_is_empty_for_auto_select(cfg):
    """
    Riva auto-selects the deployed model when ``model`` is empty. This is
    the right default because internal NVIDIA deployment names change
    across versions/regions.
    """
    result = cfg.transform_audio_transcription_request(
        model="nvidia/parakeet-ctc-1_1b-asr",
        audio_file=b"fake-audio",
        optional_params={"language_code": "en-US"},
        litellm_params={"api_base": "grpc.nvcf.nvidia.com:443"},
    )
    assert result.data["recognition_config"]["model"] == ""


def test_chunking_strategy_server_vad_maps_to_endpointing_config(cfg):
    result = cfg.transform_audio_transcription_request(
        model="m",
        audio_file=b"x",
        optional_params={
            "chunking_strategy": {
                "type": "server_vad",
                "threshold": 0.5,
                "silence_duration_ms": 700,
                "prefix_padding_ms": 250,
            }
        },
        litellm_params={"api_base": "localhost:50051"},
    )
    ep = result.data["recognition_config"].get("endpointing_config")
    assert ep is not None
    assert ep["start_threshold"] == 0.5
    assert ep["stop_threshold"] == 0.5
    assert ep["stop_history"] == 700
    assert ep["stop_history_eou"] == 250


def test_chunking_strategy_auto_leaves_endpointing_config_unset(cfg):
    result = cfg.transform_audio_transcription_request(
        model="m",
        audio_file=b"x",
        optional_params={"chunking_strategy": "auto"},
        litellm_params={"api_base": "localhost:50051"},
    )
    assert "endpointing_config" not in result.data["recognition_config"]


def test_explicit_endpointing_config_pass_through(cfg):
    result = cfg.transform_audio_transcription_request(
        model="m",
        audio_file=b"x",
        optional_params={
            "endpointing_config": {"stop_history": 1200, "start_threshold": 0.3}
        },
        litellm_params={"api_base": "localhost:50051"},
    )
    ep = result.data["recognition_config"]["endpointing_config"]
    assert ep == {"stop_history": 1200, "start_threshold": 0.3}


def test_build_transcription_response_text_format():
    final_results = [
        {"transcript": "Hello,", "words": []},
        {"transcript": " this is parakeet.", "words": []},
    ]
    response = NvidiaRivaAudioTranscriptionConfig.build_transcription_response(
        final_results=final_results,
        response_format="json",
        duration_seconds=2.4,
        timestamp_granularities=None,
    )
    assert response.text == "Hello, this is parakeet."
    assert response["task"] == "transcribe"
    # duration is only attached for verbose_json
    assert "duration" not in response


def test_build_transcription_response_skips_empty_chunks():
    final_results = [
        {"transcript": "", "words": []},
        {"transcript": "actual content", "words": []},
        {"transcript": "", "words": []},
    ]
    response = NvidiaRivaAudioTranscriptionConfig.build_transcription_response(
        final_results=final_results,
        response_format="json",
        duration_seconds=1.0,
        timestamp_granularities=None,
    )
    assert response.text == "actual content"


def test_build_transcription_response_verbose_json_with_words():
    final_results = [
        {
            "transcript": "Hello,",
            "words": [
                {"word": "Hello,", "start_time_ms": 0, "end_time_ms": 320},
            ],
        },
        {
            "transcript": " world.",
            "words": [
                {"word": "world.", "start_time_ms": 480, "end_time_ms": 870},
            ],
        },
    ]
    response = NvidiaRivaAudioTranscriptionConfig.build_transcription_response(
        final_results=final_results,
        response_format="verbose_json",
        duration_seconds=2.475,
        timestamp_granularities=["word"],
    )

    assert response.text == "Hello, world."
    assert response["duration"] == 2.475
    words = response["words"]
    assert words[0]["word"] == "Hello,"
    # Riva returns ms; OpenAI exposes seconds.
    assert words[0]["start"] == pytest.approx(0.0)
    assert words[0]["end"] == pytest.approx(0.32)
    assert words[1]["start"] == pytest.approx(0.48)
    assert words[1]["end"] == pytest.approx(0.87)


def test_build_transcription_response_verbose_json_without_word_granularity_omits_words():
    final_results = [
        {
            "transcript": "Hi.",
            "words": [
                {"word": "Hi.", "start_time_ms": 0, "end_time_ms": 200},
            ],
        }
    ]
    response = NvidiaRivaAudioTranscriptionConfig.build_transcription_response(
        final_results=final_results,
        response_format="verbose_json",
        duration_seconds=0.2,
        timestamp_granularities=["segment"],
    )
    assert "words" not in response


def test_transform_response_not_used_raises_clear_error(cfg):
    with pytest.raises(NotImplementedError):
        cfg.transform_audio_transcription_response(raw_response=None)  # type: ignore[arg-type]


def test_get_error_class_returns_nvidia_riva_exception(cfg):
    err = cfg.get_error_class(error_message="bad", status_code=401, headers={})
    assert isinstance(err, NvidiaRivaException)
    assert err.status_code == 401
