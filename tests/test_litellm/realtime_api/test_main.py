import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

import pytest

from litellm.realtime_api import main as realtime_main
from litellm.realtime_api.main import _with_resolved_session_model


class FakeLogging:
    def update_from_kwargs(self, **kwargs):
        pass


def test_resolves_top_level_session_model():
    resolved = _with_resolved_session_model({"model": "alias/gpt-realtime"}, "gpt-realtime")
    assert resolved == {"model": "gpt-realtime"}


def test_session_without_model_is_returned_unchanged():
    session = {"type": "realtime", "audio": {"input": {}}}
    assert _with_resolved_session_model(session, "gpt-realtime") == session


def test_does_not_clobber_flat_transcription_model():
    """The nested transcription model is a different model than the realtime
    conversation model and must not be overwritten with the routing model."""
    resolved = _with_resolved_session_model(
        {"model": "gpt-4o-realtime-preview", "input_audio_transcription": {"model": "whisper-1"}},
        "gpt-4o-realtime-preview",
    )
    assert resolved["input_audio_transcription"]["model"] == "whisper-1"


def test_does_not_clobber_nested_audio_transcription_model():
    resolved = _with_resolved_session_model(
        {
            "model": "gpt-4o-realtime-preview",
            "audio": {"input": {"transcription": {"model": "whisper-1"}}},
        },
        "gpt-4o-realtime-preview",
    )
    assert resolved["audio"]["input"]["transcription"]["model"] == "whisper-1"


def test_original_session_is_not_mutated():
    session = {"model": "alias/gpt-realtime"}
    _with_resolved_session_model(session, "gpt-realtime")
    assert session == {"model": "alias/gpt-realtime"}


def _run_client_secret(session, model, monkeypatch):
    captured = {}

    async def mock_handler(**kwargs):
        captured.update(kwargs)
        return object()

    def mock_get_llm_provider(model, api_base, api_key):
        return model, "openai", None, api_base

    monkeypatch.setattr(realtime_main, "get_llm_provider", mock_get_llm_provider)
    monkeypatch.setattr(
        realtime_main.base_llm_http_handler,
        "async_realtime_client_secret_handler",
        mock_handler,
    )

    asyncio.run(
        realtime_main.acreate_realtime_client_secret.__wrapped__(
            model=model,
            session=session,
            litellm_logging_obj=FakeLogging(),
        )
    )
    return captured


def test_client_secret_session_model_takes_priority_over_top_level(monkeypatch):
    """Backwards-compatible ordering: an explicit session.model wins over the
    top-level model, matching the proxy's own resolution order."""
    captured = _run_client_secret(
        session={"model": "gpt-realtime-session"},
        model="gpt-realtime-top-level",
        monkeypatch=monkeypatch,
    )
    assert captured["model"] == "gpt-realtime-session"
    assert captured["request_data"]["session"]["model"] == "gpt-realtime-session"


def test_client_secret_forwards_nested_transcription_model_untouched(monkeypatch):
    captured = _run_client_secret(
        session={
            "model": "gpt-4o-realtime-preview",
            "input_audio_transcription": {"model": "whisper-1"},
        },
        model=None,
        monkeypatch=monkeypatch,
    )
    session = captured["request_data"]["session"]
    assert session["model"] == "gpt-4o-realtime-preview"
    assert session["input_audio_transcription"]["model"] == "whisper-1"
