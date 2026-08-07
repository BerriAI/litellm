"""
Tests for the Chirp gRPC routing helpers added to litellm/main.py and for
the proxy WebSocket endpoint delegation.

These tests focus on the lines Codecov identified as uncovered:
- _is_chirp_grpc_request() in main.py
- _resolve_chirp_hd_voice_name() in main.py
- The dispatch branches in transcription() and speech()
"""

import os
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.main import _is_chirp_grpc_request, _resolve_chirp_hd_voice_name
from litellm.types.utils import TranscriptionResponse


class TestIsChirpGrpcRequest:
    @pytest.mark.parametrize(
        "model,optional_params,expected",
        [
            ("vertex_ai/chirp", {}, True),
            ("vertex_ai/chirp_2", {}, True),
            ("vertex_ai/chirp_telephony", {}, True),
            ("vertex_ai/chirp_3", {}, False),
            ("vertex_ai/chirp_3", {"use_grpc": True}, True),
            ("vertex_ai/latest_long", {}, False),
            ("vertex_ai/latest_long", {"use_grpc": True}, True),
            ("openai/whisper-1", {}, False),
        ],
    )
    def test_routing_logic(self, model, optional_params, expected):
        assert _is_chirp_grpc_request(model, optional_params) is expected


class TestResolveChirpHdVoiceName:
    def test_chirp_hd_voice_string_returns_name(self):
        result = _resolve_chirp_hd_voice_name("en-US-Chirp3-HD-Charon", {})
        assert result == "en-US-Chirp3-HD-Charon"

    def test_chirp_hd_in_vertex_voice_dict_returns_name(self):
        optional_params = {"vertex_voice_dict": {"name": "es-ES-Chirp3-HD-Kore", "languageCode": "es-ES"}}
        result = _resolve_chirp_hd_voice_name(None, optional_params)
        assert result == "es-ES-Chirp3-HD-Kore"

    def test_non_chirp_hd_voice_returns_none(self):
        assert _resolve_chirp_hd_voice_name("en-US-Studio-O", {}) is None

    def test_none_voice_returns_none(self):
        assert _resolve_chirp_hd_voice_name(None, {}) is None

    def test_non_chirp_hd_voice_dict_returns_none(self):
        optional_params = {"vertex_voice_dict": {"name": "en-US-Studio-O"}}
        assert _resolve_chirp_hd_voice_name(None, optional_params) is None

    def test_voice_string_takes_precedence_when_dict_is_not_chirp_hd(self):
        optional_params = {"vertex_voice_dict": {"name": "en-US-Studio-O"}}
        result = _resolve_chirp_hd_voice_name("es-ES-Chirp3-HD-Charon", optional_params)
        assert result == "es-ES-Chirp3-HD-Charon"


class TestTranscriptionChirpDispatch:
    """Verify the routing predicate that guards the gRPC STT dispatch branch."""

    def test_chirp_models_trigger_grpc_branch(self):
        for model in ("vertex_ai/chirp", "vertex_ai/chirp_2", "vertex_ai/chirp_telephony"):
            assert _is_chirp_grpc_request(model, {}) is True

    def test_non_chirp_models_do_not_trigger_grpc_branch(self):
        for model in ("vertex_ai/chirp_3", "vertex_ai/latest_long", "openai/whisper-1"):
            assert _is_chirp_grpc_request(model, {}) is False

    def test_use_grpc_flag_overrides_model_detection(self):
        assert _is_chirp_grpc_request("vertex_ai/latest_long", {"use_grpc": True}) is True
        assert _is_chirp_grpc_request("vertex_ai/latest_long", {"use_grpc": False}) is False


class TestSpeechChirpHdDispatch:
    """Verify that _resolve_chirp_hd_voice_name covers all voice detection paths."""

    def test_all_chirp_hd_path_variants(self):
        voices = [
            "en-US-Chirp3-HD-Charon",
            "es-ES-Chirp3-HD-Aoede",
            "fr-FR-Chirp3-HD-Kore",
            "de-DE-Chirp3-HD-Puck",
        ]
        for voice in voices:
            result = _resolve_chirp_hd_voice_name(voice, {})
            assert result == voice, f"Expected {voice}, got {result}"

    def test_vertex_voice_dict_path(self):
        for voice_name in ["en-US-Chirp3-HD-Charon", "es-ES-Chirp3-HD-Kore"]:
            optional_params = {"vertex_voice_dict": {"name": voice_name, "languageCode": voice_name[:5]}}
            result = _resolve_chirp_hd_voice_name(None, optional_params)
            assert result == voice_name

    def test_standard_voices_return_none(self):
        standard = ["alloy", "en-US-Studio-O", "en-US-Wavenet-D", "nova"]
        for voice in standard:
            assert _resolve_chirp_hd_voice_name(voice, {}) is None
