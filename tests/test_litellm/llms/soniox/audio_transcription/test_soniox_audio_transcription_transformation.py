"""Tests for SonioxAudioTranscriptionConfig."""

import json
from typing import Any, Dict, Optional
from unittest.mock import patch

import httpx
import pytest

from litellm.llms.soniox.audio_transcription.transformation import (
    SonioxAudioTranscriptionConfig,
)
from litellm.llms.soniox.common_utils import SonioxException


def _make_response(payload: Dict[str, Any], status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        content=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
    )


class TestGetSupportedOpenAIParams:
    def test_should_only_advertise_language(self):
        cfg = SonioxAudioTranscriptionConfig()
        assert cfg.get_supported_openai_params(model="stt-async-v4") == ["language"]


class TestMapOpenAIParams:
    def test_should_translate_language_to_language_hints(self):
        cfg = SonioxAudioTranscriptionConfig()
        result = cfg.map_openai_params(
            non_default_params={"language": "en"},
            optional_params={},
            model="stt-async-v4",
            drop_params=False,
        )
        assert result["language_hints"] == ["en"]

    def test_should_prepend_language_to_existing_hints(self):
        cfg = SonioxAudioTranscriptionConfig()
        result = cfg.map_openai_params(
            non_default_params={"language": "en"},
            optional_params={"language_hints": ["fr"]},
            model="stt-async-v4",
            drop_params=False,
        )
        assert result["language_hints"] == ["en", "fr"]

    def test_should_not_duplicate_language_already_in_hints(self):
        cfg = SonioxAudioTranscriptionConfig()
        result = cfg.map_openai_params(
            non_default_params={"language": "en"},
            optional_params={"language_hints": ["en", "fr"]},
            model="stt-async-v4",
            drop_params=False,
        )
        assert result["language_hints"] == ["en", "fr"]

    def test_should_passthrough_soniox_native_kwargs(self):
        cfg = SonioxAudioTranscriptionConfig()
        result = cfg.map_openai_params(
            non_default_params={
                "enable_speaker_diarization": True,
                "enable_language_identification": True,
                "context": "medical conversation",
                "audio_url": "https://example.com/a.wav",
            },
            optional_params={},
            model="stt-async-v4",
            drop_params=False,
        )
        assert result["enable_speaker_diarization"] is True
        assert result["enable_language_identification"] is True
        assert result["context"] == "medical conversation"
        assert result["audio_url"] == "https://example.com/a.wav"

    def test_should_passthrough_handler_only_kwargs(self):
        cfg = SonioxAudioTranscriptionConfig()
        result = cfg.map_openai_params(
            non_default_params={
                "soniox_polling_interval": 0.5,
                "soniox_max_polling_attempts": 10,
                "soniox_cleanup": ["file"],
            },
            optional_params={},
            model="stt-async-v4",
            drop_params=False,
        )
        assert result["soniox_polling_interval"] == 0.5
        assert result["soniox_max_polling_attempts"] == 10
        assert result["soniox_cleanup"] == ["file"]


class TestValidateEnvironment:
    def test_should_set_bearer_token_from_api_key(self):
        cfg = SonioxAudioTranscriptionConfig()
        headers = cfg.validate_environment(
            headers={},
            model="stt-async-v4",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="sk-test",
        )
        assert headers["Authorization"] == "Bearer sk-test"

    def test_should_resolve_key_from_env(self, monkeypatch):
        monkeypatch.setenv("SONIOX_API_KEY", "env-key")
        cfg = SonioxAudioTranscriptionConfig()
        headers = cfg.validate_environment(
            headers={},
            model="stt-async-v4",
            messages=[],
            optional_params={},
            litellm_params={},
        )
        assert headers["Authorization"] == "Bearer env-key"

    def test_should_raise_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("SONIOX_API_KEY", raising=False)
        cfg = SonioxAudioTranscriptionConfig()
        with pytest.raises(SonioxException) as exc_info:
            cfg.validate_environment(
                headers={},
                model="stt-async-v4",
                messages=[],
                optional_params={},
                litellm_params={},
            )
        assert exc_info.value.status_code == 401

    def test_should_merge_caller_headers(self):
        cfg = SonioxAudioTranscriptionConfig()
        headers = cfg.validate_environment(
            headers={"X-Trace-Id": "abc"},
            model="stt-async-v4",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="sk-test",
        )
        assert headers["X-Trace-Id"] == "abc"
        assert headers["Authorization"] == "Bearer sk-test"


class TestGetCompleteUrl:
    def test_should_return_default_base(self):
        cfg = SonioxAudioTranscriptionConfig()
        url = cfg.get_complete_url(
            api_base=None,
            api_key="sk-test",
            model="stt-async-v4",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://api.soniox.com"

    def test_should_strip_trailing_slash_from_custom_base(self):
        cfg = SonioxAudioTranscriptionConfig()
        url = cfg.get_complete_url(
            api_base="https://custom.example.com/",
            api_key="sk-test",
            model="stt-async-v4",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.example.com"


class TestTransformAudioTranscriptionRequest:
    def test_should_build_minimal_body_with_model(self):
        cfg = SonioxAudioTranscriptionConfig()
        result = cfg.transform_audio_transcription_request(
            model="stt-async-v4",
            audio_file=None,
            optional_params={},
            litellm_params={},
        )
        assert result.data == {"model": "stt-async-v4"}
        assert result.files is None
        assert result.content_type == "application/json"

    def test_should_include_passthrough_params_in_body(self):
        cfg = SonioxAudioTranscriptionConfig()
        result = cfg.transform_audio_transcription_request(
            model="stt-async-v4",
            audio_file=None,
            optional_params={
                "audio_url": "https://example.com/a.wav",
                "language_hints": ["en"],
                "enable_speaker_diarization": True,
                "soniox_polling_interval": 0.5,  # handler-only, must NOT appear
            },
            litellm_params={},
        )
        body = result.data
        assert body["audio_url"] == "https://example.com/a.wav"
        assert body["language_hints"] == ["en"]
        assert body["enable_speaker_diarization"] is True
        assert "soniox_polling_interval" not in body


class TestTransformAudioTranscriptionResponse:
    def test_should_build_response_from_plain_transcript_payload(self):
        cfg = SonioxAudioTranscriptionConfig()
        resp = cfg.transform_audio_transcription_response(
            _make_response({"id": "tx_1", "text": "hello world"}),
        )
        assert resp.text == "hello world"
        assert resp["task"] == "transcribe"

    def test_should_build_response_from_envelope_payload(self):
        cfg = SonioxAudioTranscriptionConfig()
        resp = cfg.transform_audio_transcription_response(
            _make_response(
                {
                    "transcription": {"id": "tx_1", "audio_duration_ms": 2500},
                    "transcript": {"text": "hello world", "tokens": []},
                }
            ),
        )
        assert resp.text == "hello world"
        assert resp["duration"] == pytest.approx(2.5)

    def test_should_render_speaker_tags_when_diarization_present(self):
        cfg = SonioxAudioTranscriptionConfig()
        payload = {
            "transcript": {
                "text": "ignored fallback",
                "tokens": [
                    {"text": "hello", "speaker": 1},
                    {"text": " world", "speaker": 2},
                ],
            }
        }
        resp = cfg._build_response_from_payload(payload)
        assert "Speaker 1:" in resp.text
        assert "Speaker 2:" in resp.text

    def test_should_set_language_when_all_tokens_share_one(self):
        cfg = SonioxAudioTranscriptionConfig()
        payload = {
            "transcript": {
                "tokens": [
                    {"text": "hello", "language": "en"},
                    {"text": " world", "language": "en"},
                ]
            }
        }
        resp = cfg._build_response_from_payload(payload)
        assert resp["language"] == "en"

    def test_should_stash_raw_payload_in_hidden_params(self):
        cfg = SonioxAudioTranscriptionConfig()
        payload = {
            "transcription": {"id": "tx_1"},
            "transcript": {"text": "hi", "tokens": []},
        }
        resp = cfg._build_response_from_payload(payload)
        raw = resp._hidden_params["soniox_raw"]
        assert raw["transcription"]["id"] == "tx_1"
        assert raw["transcript"]["text"] == "hi"

    def test_should_raise_on_invalid_json(self):
        cfg = SonioxAudioTranscriptionConfig()
        bad = httpx.Response(status_code=200, content=b"not json")
        with pytest.raises(SonioxException):
            cfg.transform_audio_transcription_response(bad)

    def test_should_concat_token_texts_when_no_text_field_or_tags(self):
        cfg = SonioxAudioTranscriptionConfig()
        payload = {
            "transcript": {
                "tokens": [
                    {"text": "hello"},
                    {"text": " world"},
                ],
            }
        }
        resp = cfg._build_response_from_payload(payload)
        assert resp.text == "hello world"

    def test_should_return_empty_text_for_empty_payload(self):
        cfg = SonioxAudioTranscriptionConfig()
        resp = cfg._build_response_from_payload({})
        assert resp.text == ""

    def test_should_skip_duration_when_audio_duration_ms_is_invalid(self):
        cfg = SonioxAudioTranscriptionConfig()
        payload = {
            "transcription": {"audio_duration_ms": "not-a-number"},
            "transcript": {"text": "hi", "tokens": []},
        }
        resp = cfg._build_response_from_payload(payload)
        assert "duration" not in resp.model_dump()


class TestRenderSonioxTokens:
    def test_should_return_empty_string_for_no_tokens(self):
        from litellm.llms.soniox.common_utils import render_soniox_tokens

        assert render_soniox_tokens([]) == ""


class TestGetErrorClass:
    def test_should_return_soniox_exception(self):
        cfg = SonioxAudioTranscriptionConfig()
        err = cfg.get_error_class(error_message="boom", status_code=500, headers={})
        assert isinstance(err, SonioxException)
        assert err.status_code == 500
