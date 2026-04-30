"""Tests for SonioxAudioTranscriptionHandler."""
import asyncio
import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.soniox.audio_transcription.handler import (
    SonioxAudioTranscriptionHandler,
)
from litellm.llms.soniox.audio_transcription.transformation import (
    SonioxAudioTranscriptionConfig,
)
from litellm.llms.soniox.common_utils import SonioxException
from litellm.types.utils import TranscriptionResponse


def _make_response(payload: Dict[str, Any], status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        content=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
    )


class _MockSyncClient(HTTPHandler):
    """Sync HTTP client that records calls and replays scripted responses."""

    def __init__(self, responses: Dict[str, List[httpx.Response]]):
        # Skip parent __init__ (don't open real httpx client).
        self._responses = responses
        self.calls: List[Dict[str, Any]] = []

    def _next(self, method: str, url: str) -> httpx.Response:
        key = f"{method.upper()} {url}"
        bucket = self._responses.get(key)
        if not bucket:
            raise AssertionError(f"Unexpected call: {key}")
        return bucket.pop(0)

    def post(self, url, headers=None, json=None, files=None, data=None, timeout=None, **kw):  # type: ignore[override]
        self.calls.append({"method": "POST", "url": url, "json": json, "files": files})
        return self._next("POST", url)

    def get(self, url, headers=None, timeout=None, **kw):  # type: ignore[override]
        self.calls.append({"method": "GET", "url": url})
        return self._next("GET", url)

    def delete(self, url, headers=None, timeout=None, **kw):  # type: ignore[override]
        self.calls.append({"method": "DELETE", "url": url})
        return self._next("DELETE", url)


class _MockAsyncClient(AsyncHTTPHandler):
    def __init__(self, responses: Dict[str, List[httpx.Response]]):
        self._responses = responses
        self.calls: List[Dict[str, Any]] = []

    def _next(self, method: str, url: str) -> httpx.Response:
        key = f"{method.upper()} {url}"
        bucket = self._responses.get(key)
        if not bucket:
            raise AssertionError(f"Unexpected call: {key}")
        return bucket.pop(0)

    async def post(self, url, headers=None, json=None, files=None, data=None, timeout=None, **kw):  # type: ignore[override]
        self.calls.append({"method": "POST", "url": url, "json": json, "files": files})
        return self._next("POST", url)

    async def get(self, url, headers=None, timeout=None, **kw):  # type: ignore[override]
        self.calls.append({"method": "GET", "url": url})
        return self._next("GET", url)

    async def delete(self, url, headers=None, timeout=None, **kw):  # type: ignore[override]
        self.calls.append({"method": "DELETE", "url": url})
        return self._next("DELETE", url)


def _make_logging_obj() -> MagicMock:
    obj = MagicMock()
    obj.pre_call = MagicMock()
    obj.post_call = MagicMock()
    return obj


def _common_call_kwargs(client) -> Dict[str, Any]:
    return {
        "model": "stt-async-v4",
        "model_response": TranscriptionResponse(),
        "timeout": 30.0,
        "max_retries": 0,
        "logging_obj": _make_logging_obj(),
        "api_key": "sk-test",
        "api_base": None,
        "client": client,
        "headers": {},
    }


class TestSyncAudioUrl:
    def test_should_create_poll_fetch_and_cleanup_when_audio_url_supplied(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda *_: None)
        responses = {
            "POST https://api.soniox.com/v1/transcriptions": [
                _make_response({"id": "tx_1", "status": "queued"})
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1": [
                _make_response({"id": "tx_1", "status": "completed", "audio_duration_ms": 1500}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1/transcript": [
                _make_response({"text": "hello world", "tokens": []}),
            ],
            "DELETE https://api.soniox.com/v1/transcriptions/tx_1": [
                _make_response({"deleted": True}),
            ],
        }
        client = _MockSyncClient(responses)

        handler = SonioxAudioTranscriptionHandler()
        resp = handler.audio_transcriptions(
            audio_file=None,
            optional_params={"audio_url": "https://example.com/a.wav"},
            litellm_params={},
            atranscription=False,
            **_common_call_kwargs(client),
        )

        assert resp.text == "hello world"
        assert resp["duration"] == pytest.approx(1.5)
        assert resp._hidden_params["custom_llm_provider"] == "soniox"
        # POST body should contain audio_url, no file_id.
        post_call = next(c for c in client.calls if c["method"] == "POST")
        assert post_call["json"]["audio_url"] == "https://example.com/a.wav"
        assert "file_id" not in post_call["json"]
        # Cleanup must have deleted the transcription record.
        assert any(c["method"] == "DELETE" for c in client.calls)


class TestSyncFileUpload:
    def test_should_upload_then_transcribe_then_cleanup_both(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda *_: None)
        responses = {
            "POST https://api.soniox.com/v1/files": [
                _make_response({"id": "file_1"}),
            ],
            "POST https://api.soniox.com/v1/transcriptions": [
                _make_response({"id": "tx_1"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1": [
                _make_response({"status": "completed"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1/transcript": [
                _make_response({"text": "uploaded ok", "tokens": []}),
            ],
            "DELETE https://api.soniox.com/v1/transcriptions/tx_1": [
                _make_response({}),
            ],
            "DELETE https://api.soniox.com/v1/files/file_1": [
                _make_response({}),
            ],
        }
        client = _MockSyncClient(responses)

        handler = SonioxAudioTranscriptionHandler()
        resp = handler.audio_transcriptions(
            audio_file=("clip.wav", b"RIFFfake", "audio/wav"),
            optional_params={},
            litellm_params={},
            atranscription=False,
            **_common_call_kwargs(client),
        )

        assert resp.text == "uploaded ok"
        deletes = [c["url"] for c in client.calls if c["method"] == "DELETE"]
        assert "https://api.soniox.com/v1/transcriptions/tx_1" in deletes
        assert "https://api.soniox.com/v1/files/file_1" in deletes


class TestSyncPolling:
    def test_should_poll_until_status_is_completed(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda *_: None)
        responses = {
            "POST https://api.soniox.com/v1/transcriptions": [
                _make_response({"id": "tx_1"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1": [
                _make_response({"status": "queued"}),
                _make_response({"status": "processing"}),
                _make_response({"status": "completed"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1/transcript": [
                _make_response({"text": "done", "tokens": []}),
            ],
            "DELETE https://api.soniox.com/v1/transcriptions/tx_1": [
                _make_response({}),
            ],
        }
        client = _MockSyncClient(responses)

        resp = SonioxAudioTranscriptionHandler().audio_transcriptions(
            audio_file=None,
            optional_params={
                "audio_url": "https://example.com/a.wav",
                "soniox_polling_interval": 0,
            },
            litellm_params={},
            atranscription=False,
            **_common_call_kwargs(client),
        )
        assert resp.text == "done"

    def test_should_raise_when_status_is_error(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda *_: None)
        responses = {
            "POST https://api.soniox.com/v1/transcriptions": [
                _make_response({"id": "tx_1"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1": [
                _make_response({"status": "error", "error_message": "bad audio"}),
            ],
        }
        client = _MockSyncClient(responses)

        with pytest.raises(SonioxException) as exc_info:
            SonioxAudioTranscriptionHandler().audio_transcriptions(
                audio_file=None,
                optional_params={"audio_url": "https://example.com/a.wav"},
                litellm_params={},
                atranscription=False,
                **_common_call_kwargs(client),
            )
        assert "bad audio" in str(exc_info.value)

    def test_should_raise_when_polling_attempts_exceeded(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda *_: None)
        responses = {
            "POST https://api.soniox.com/v1/transcriptions": [
                _make_response({"id": "tx_1"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1": [
                _make_response({"status": "processing"}),
                _make_response({"status": "processing"}),
            ],
        }
        client = _MockSyncClient(responses)

        with pytest.raises(SonioxException) as exc_info:
            SonioxAudioTranscriptionHandler().audio_transcriptions(
                audio_file=None,
                optional_params={
                    "audio_url": "https://example.com/a.wav",
                    "soniox_polling_interval": 0,
                    "soniox_max_polling_attempts": 2,
                },
                litellm_params={},
                atranscription=False,
                **_common_call_kwargs(client),
            )
        assert exc_info.value.status_code == 504


class TestSyncCleanupBehavior:
    def test_should_skip_cleanup_when_disabled(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda *_: None)
        responses = {
            "POST https://api.soniox.com/v1/transcriptions": [
                _make_response({"id": "tx_1"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1": [
                _make_response({"status": "completed"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1/transcript": [
                _make_response({"text": "no cleanup", "tokens": []}),
            ],
        }
        client = _MockSyncClient(responses)

        SonioxAudioTranscriptionHandler().audio_transcriptions(
            audio_file=None,
            optional_params={
                "audio_url": "https://example.com/a.wav",
                "soniox_cleanup": [],
            },
            litellm_params={},
            atranscription=False,
            **_common_call_kwargs(client),
        )
        assert not any(c["method"] == "DELETE" for c in client.calls)

    def test_should_cleanup_even_on_error(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda *_: None)
        responses = {
            "POST https://api.soniox.com/v1/files": [
                _make_response({"id": "file_99"}),
            ],
            "POST https://api.soniox.com/v1/transcriptions": [
                _make_response({"id": "tx_99"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_99": [
                _make_response({"status": "error", "error_message": "boom"}),
            ],
            "DELETE https://api.soniox.com/v1/transcriptions/tx_99": [
                _make_response({}),
            ],
            "DELETE https://api.soniox.com/v1/files/file_99": [
                _make_response({}),
            ],
        }
        client = _MockSyncClient(responses)

        with pytest.raises(SonioxException):
            SonioxAudioTranscriptionHandler().audio_transcriptions(
                audio_file=("clip.wav", b"x", "audio/wav"),
                optional_params={},
                litellm_params={},
                atranscription=False,
                **_common_call_kwargs(client),
            )
        deletes = [c["url"] for c in client.calls if c["method"] == "DELETE"]
        assert any("/v1/files/file_99" in u for u in deletes)


class TestMissingInput:
    def test_should_raise_when_no_audio_input_provided(self):
        client = _MockSyncClient({})
        with pytest.raises(SonioxException) as exc_info:
            SonioxAudioTranscriptionHandler().audio_transcriptions(
                audio_file=None,
                optional_params={},
                litellm_params={},
                atranscription=False,
                **_common_call_kwargs(client),
            )
        assert exc_info.value.status_code == 400


class TestAsyncFlow:
    def test_should_run_async_audio_url_flow(self, monkeypatch):
        async def _no_sleep(*_a, **_kw):
            return None

        monkeypatch.setattr(asyncio, "sleep", _no_sleep)

        responses = {
            "POST https://api.soniox.com/v1/transcriptions": [
                _make_response({"id": "tx_async"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_async": [
                _make_response({"status": "completed"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_async/transcript": [
                _make_response({"text": "async ok", "tokens": []}),
            ],
            "DELETE https://api.soniox.com/v1/transcriptions/tx_async": [
                _make_response({}),
            ],
        }
        client = _MockAsyncClient(responses)

        coro = SonioxAudioTranscriptionHandler().audio_transcriptions(
            audio_file=None,
            optional_params={"audio_url": "https://example.com/a.wav"},
            litellm_params={},
            atranscription=True,
            **_common_call_kwargs(client),
        )
        resp = asyncio.new_event_loop().run_until_complete(coro)
        assert resp.text == "async ok"
        assert resp._hidden_params["custom_llm_provider"] == "soniox"
