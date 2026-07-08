"""Tests for SonioxAudioTranscriptionHandler."""

import asyncio
import json
from typing import Any, Dict, List
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
        self.calls.append({"method": "GET", "url": url, "timeout": timeout})
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
        self.calls.append({"method": "GET", "url": url, "timeout": timeout})
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
    def test_should_create_poll_fetch_and_cleanup_when_audio_url_supplied(
        self, monkeypatch
    ):
        monkeypatch.setattr("time.sleep", lambda *_: None)
        responses = {
            "POST https://api.soniox.com/v1/transcriptions": [
                _make_response({"id": "tx_1", "status": "queued"})
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1": [
                _make_response(
                    {"id": "tx_1", "status": "completed", "audio_duration_ms": 1500}
                ),
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


class TestGetRequestTimeoutForwarding:
    def test_sync_should_forward_timeout_to_poll_and_transcript_gets(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda *_: None)
        responses = {
            "POST https://api.soniox.com/v1/transcriptions": [
                _make_response({"id": "tx_1"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1": [
                _make_response({"status": "completed"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1/transcript": [
                _make_response({"text": "done", "tokens": []}),
            ],
        }
        client = _MockSyncClient(responses)

        SonioxAudioTranscriptionHandler().audio_transcriptions(
            audio_file=None,
            optional_params={
                "audio_url": "https://example.com/a.wav",
                "soniox_cleanup": None,
            },
            litellm_params={},
            atranscription=False,
            **_common_call_kwargs(client),
        )

        get_calls = [c for c in client.calls if c["method"] == "GET"]
        assert get_calls
        assert all(c["timeout"] == 30.0 for c in get_calls)

    def test_async_should_forward_timeout_to_poll_and_transcript_gets(self):
        responses = {
            "POST https://api.soniox.com/v1/transcriptions": [
                _make_response({"id": "tx_1"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1": [
                _make_response({"status": "completed"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1/transcript": [
                _make_response({"text": "done", "tokens": []}),
            ],
        }
        client = _MockAsyncClient(responses)

        asyncio.run(
            SonioxAudioTranscriptionHandler().audio_transcriptions(
                audio_file=None,
                optional_params={
                    "audio_url": "https://example.com/a.wav",
                    "soniox_cleanup": None,
                },
                litellm_params={},
                atranscription=True,
                **_common_call_kwargs(client),
            )
        )

        get_calls = [c for c in client.calls if c["method"] == "GET"]
        assert get_calls
        assert all(c["timeout"] == 30.0 for c in get_calls)


class TestPollLimitsClamping:
    """Server-side caps on caller-supplied poll settings.

    `soniox_polling_interval` and `soniox_max_polling_attempts` arrive as
    request kwargs from authenticated callers. They MUST be clamped server-side
    so a hostile caller cannot set a zero interval + huge attempt count to pin
    a worker on tight poll loops.
    """

    def test_should_clamp_poll_interval_to_minimum(self):
        from litellm.llms.soniox.common_utils import SONIOX_MIN_POLL_INTERVAL

        handler = SonioxAudioTranscriptionHandler()
        _, _, _, handler_opts = handler._prepare(
            audio_file=None,
            optional_params={
                "soniox_polling_interval": 0,
                "audio_url": "https://example.com/a.wav",
            },
            litellm_params={},
            api_key="sk-test",
            api_base=None,
            provider_config=SonioxAudioTranscriptionConfig(),
            headers={},
        )
        assert handler_opts["poll_interval"] == SONIOX_MIN_POLL_INTERVAL

    def test_should_clamp_negative_poll_interval_to_minimum(self):
        from litellm.llms.soniox.common_utils import SONIOX_MIN_POLL_INTERVAL

        handler = SonioxAudioTranscriptionHandler()
        _, _, _, handler_opts = handler._prepare(
            audio_file=None,
            optional_params={
                "soniox_polling_interval": -10,
                "audio_url": "https://example.com/a.wav",
            },
            litellm_params={},
            api_key="sk-test",
            api_base=None,
            provider_config=SonioxAudioTranscriptionConfig(),
            headers={},
        )
        assert handler_opts["poll_interval"] == SONIOX_MIN_POLL_INTERVAL

    def test_should_preserve_poll_interval_when_above_minimum(self):
        handler = SonioxAudioTranscriptionHandler()
        _, _, _, handler_opts = handler._prepare(
            audio_file=None,
            optional_params={
                "soniox_polling_interval": 5.0,
                "audio_url": "https://example.com/a.wav",
            },
            litellm_params={},
            api_key="sk-test",
            api_base=None,
            provider_config=SonioxAudioTranscriptionConfig(),
            headers={},
        )
        assert handler_opts["poll_interval"] == 5.0

    def test_should_clamp_max_attempts_to_upper_bound(self):
        from litellm.llms.soniox.common_utils import SONIOX_MAX_POLL_ATTEMPTS

        handler = SonioxAudioTranscriptionHandler()
        _, _, _, handler_opts = handler._prepare(
            audio_file=None,
            optional_params={
                "soniox_max_polling_attempts": 10**9,
                "audio_url": "https://example.com/a.wav",
            },
            litellm_params={},
            api_key="sk-test",
            api_base=None,
            provider_config=SonioxAudioTranscriptionConfig(),
            headers={},
        )
        assert handler_opts["max_attempts"] == SONIOX_MAX_POLL_ATTEMPTS

    def test_should_clamp_zero_max_attempts_to_one(self):
        handler = SonioxAudioTranscriptionHandler()
        _, _, _, handler_opts = handler._prepare(
            audio_file=None,
            optional_params={
                "soniox_max_polling_attempts": 0,
                "audio_url": "https://example.com/a.wav",
            },
            litellm_params={},
            api_key="sk-test",
            api_base=None,
            provider_config=SonioxAudioTranscriptionConfig(),
            headers={},
        )
        assert handler_opts["max_attempts"] == 1

    def test_should_preserve_max_attempts_within_bounds(self):
        handler = SonioxAudioTranscriptionHandler()
        _, _, _, handler_opts = handler._prepare(
            audio_file=None,
            optional_params={
                "soniox_max_polling_attempts": 10,
                "audio_url": "https://example.com/a.wav",
            },
            litellm_params={},
            api_key="sk-test",
            api_base=None,
            provider_config=SonioxAudioTranscriptionConfig(),
            headers={},
        )
        assert handler_opts["max_attempts"] == 10


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


class TestLoggingExceptionSafety:
    """Logging callbacks must never break a real Soniox call.

    `_safe_log_pre_call` and `_safe_log_post_call` wrap their `logging_obj`
    invocations in a broad `except Exception: pass` because callbacks come
    from third-party observability integrations and a misbehaving one must
    not abort the transcription.
    """

    def test_pre_call_should_swallow_logging_exception(self):
        logging_obj = MagicMock()
        logging_obj.pre_call.side_effect = RuntimeError("callback boom")
        # Must not raise.
        SonioxAudioTranscriptionHandler._safe_log_pre_call(
            logging_obj=logging_obj,
            api_key="sk-test",
            api_base="https://api.soniox.com",
            body={"model": "stt-async-v4"},
        )
        # Helper still attempted the call exactly once before swallowing.
        assert logging_obj.pre_call.call_count == 1

    def test_post_call_should_swallow_logging_exception(self):
        logging_obj = MagicMock()
        logging_obj.post_call.side_effect = RuntimeError("callback boom")
        # Must not raise.
        SonioxAudioTranscriptionHandler._safe_log_post_call(
            logging_obj=logging_obj,
            audio_file=None,
            api_key="sk-test",
            body={"model": "stt-async-v4"},
            original_response={"transcription": {}, "transcript": {}},
        )
        assert logging_obj.post_call.call_count == 1


class _RaisingDeleteSyncClient(_MockSyncClient):
    """Sync mock whose DELETE calls always raise.

    Used to drive the `_sync_cleanup` exception-swallowing branches: a failed
    DELETE during cleanup must not mask the transcription result (or the
    original error on the failure path).
    """

    def delete(self, url, headers=None, timeout=None, **kw):  # type: ignore[override]
        self.calls.append({"method": "DELETE", "url": url})
        raise httpx.ConnectError("delete failed")


class _RaisingDeleteAsyncClient(_MockAsyncClient):
    """Async counterpart of `_RaisingDeleteSyncClient`."""

    async def delete(self, url, headers=None, timeout=None, **kw):  # type: ignore[override]
        self.calls.append({"method": "DELETE", "url": url})
        raise httpx.ConnectError("delete failed")


class TestCleanupExceptionMasking:
    """Cleanup DELETE failures must be swallowed (best-effort).

    A failed DELETE leaves stale data on Soniox but must NOT replace the
    successful transcription result, nor mask the original error on the
    error path.
    """

    def test_sync_cleanup_should_swallow_delete_failures(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda *_: None)
        responses = {
            "POST https://api.soniox.com/v1/files": [
                _make_response({"id": "file_99"}),
            ],
            "POST https://api.soniox.com/v1/transcriptions": [
                _make_response({"id": "tx_99"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_99": [
                _make_response({"status": "completed"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_99/transcript": [
                _make_response({"text": "ok", "tokens": []}),
            ],
        }
        client = _RaisingDeleteSyncClient(responses)

        # Result must come through despite both DELETEs raising.
        resp = SonioxAudioTranscriptionHandler().audio_transcriptions(
            audio_file=("clip.wav", b"x", "audio/wav"),
            optional_params={"soniox_cleanup": ["file", "transcription"]},
            litellm_params={},
            atranscription=False,
            **_common_call_kwargs(client),
        )
        assert resp.text == "ok"
        # Both DELETEs were attempted (proving the except: pass paths ran).
        deletes = [c["url"] for c in client.calls if c["method"] == "DELETE"]
        assert any("/v1/transcriptions/tx_99" in u for u in deletes)
        assert any("/v1/files/file_99" in u for u in deletes)

    def test_async_cleanup_should_swallow_delete_failures(self, monkeypatch):
        async def _no_sleep(*_args, **_kwargs):
            return None

        monkeypatch.setattr("asyncio.sleep", _no_sleep)
        responses = {
            "POST https://api.soniox.com/v1/files": [
                _make_response({"id": "file_async"}),
            ],
            "POST https://api.soniox.com/v1/transcriptions": [
                _make_response({"id": "tx_async"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_async": [
                _make_response({"status": "completed"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_async/transcript": [
                _make_response({"text": "async ok", "tokens": []}),
            ],
        }
        client = _RaisingDeleteAsyncClient(responses)

        coro = SonioxAudioTranscriptionHandler().audio_transcriptions(
            audio_file=("clip.wav", b"x", "audio/wav"),
            optional_params={"soniox_cleanup": ["file", "transcription"]},
            litellm_params={},
            atranscription=True,
            **_common_call_kwargs(client),
        )
        resp = asyncio.new_event_loop().run_until_complete(coro)
        assert resp.text == "async ok"
        deletes = [c["url"] for c in client.calls if c["method"] == "DELETE"]
        assert any("/v1/transcriptions/tx_async" in u for u in deletes)
        assert any("/v1/files/file_async" in u for u in deletes)


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


class TestCleanupNormalization:
    def test_should_treat_none_cleanup_as_no_cleanup(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda *_: None)
        responses = {
            "POST https://api.soniox.com/v1/transcriptions": [
                _make_response({"id": "tx_1"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1": [
                _make_response({"status": "completed"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1/transcript": [
                _make_response({"text": "hi", "tokens": []}),
            ],
        }
        client = _MockSyncClient(responses)
        SonioxAudioTranscriptionHandler().audio_transcriptions(
            audio_file=None,
            optional_params={
                "audio_url": "https://example.com/a.wav",
                "soniox_cleanup": None,
            },
            litellm_params={},
            atranscription=False,
            **_common_call_kwargs(client),
        )
        assert not any(c["method"] == "DELETE" for c in client.calls)

    def test_should_accept_cleanup_as_single_string(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda *_: None)
        responses = {
            "POST https://api.soniox.com/v1/transcriptions": [
                _make_response({"id": "tx_1"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1": [
                _make_response({"status": "completed"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1/transcript": [
                _make_response({"text": "hi", "tokens": []}),
            ],
            "DELETE https://api.soniox.com/v1/transcriptions/tx_1": [
                _make_response({}),
            ],
        }
        client = _MockSyncClient(responses)
        SonioxAudioTranscriptionHandler().audio_transcriptions(
            audio_file=None,
            optional_params={
                "audio_url": "https://example.com/a.wav",
                "soniox_cleanup": "transcription",
            },
            litellm_params={},
            atranscription=False,
            **_common_call_kwargs(client),
        )
        deletes = [c["url"] for c in client.calls if c["method"] == "DELETE"]
        assert "https://api.soniox.com/v1/transcriptions/tx_1" in deletes


class TestErrorResponses:
    def test_should_raise_on_4xx_during_create_with_json_error(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda *_: None)
        responses = {
            "POST https://api.soniox.com/v1/transcriptions": [
                _make_response({"error_message": "invalid model"}, status_code=400),
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
        assert "invalid model" in str(exc_info.value)
        assert exc_info.value.status_code == 400

    def test_should_raise_on_4xx_during_create_with_non_json_body(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda *_: None)
        responses = {
            "POST https://api.soniox.com/v1/transcriptions": [
                httpx.Response(status_code=500, content=b"server exploded"),
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
        assert "server exploded" in str(exc_info.value)
        assert exc_info.value.status_code == 500


class TestPassthroughBodyBuilding:
    def test_should_skip_none_values_in_passthrough_body(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda *_: None)
        responses = {
            "POST https://api.soniox.com/v1/transcriptions": [
                _make_response({"id": "tx_1"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1": [
                _make_response({"status": "completed"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1/transcript": [
                _make_response({"text": "ok", "tokens": []}),
            ],
            "DELETE https://api.soniox.com/v1/transcriptions/tx_1": [
                _make_response({}),
            ],
        }
        client = _MockSyncClient(responses)
        # Pass a None-valued kwarg through the entire pipeline (it must not
        # appear in the create body).
        SonioxAudioTranscriptionHandler().audio_transcriptions(
            audio_file=None,
            optional_params={
                "audio_url": "https://example.com/a.wav",
                "context": None,
            },
            litellm_params={},
            atranscription=False,
            **_common_call_kwargs(client),
        )
        post_call = next(c for c in client.calls if c["method"] == "POST")
        assert "context" not in post_call["json"]


class TestSecretRedaction:
    """Secret-bearing fields must be redacted before reaching logging callbacks.

    `webhook_auth_header_value` is forwarded to Soniox so it can authenticate
    its webhook callbacks to the caller. It must NOT leak into LiteLLM logging
    callbacks: anyone with access to those sinks could otherwise forge webhook
    requests. The HTTP request to Soniox itself must still carry the real
    value.
    """

    def test_redact_helper_should_redact_known_secret_fields(self):
        body = {
            "model": "stt-async-v4",
            "audio_url": "https://example.com/a.wav",
            "webhook_url": "https://example.com/hook",
            "webhook_auth_header_name": "X-Webhook-Auth",
            "webhook_auth_header_value": "super-secret-token",
        }
        redacted = SonioxAudioTranscriptionHandler._redact_body_for_logging(body)
        assert redacted["webhook_auth_header_value"] == "[REDACTED]"
        # Non-secret fields untouched.
        assert redacted["model"] == "stt-async-v4"
        assert redacted["audio_url"] == "https://example.com/a.wav"
        assert redacted["webhook_url"] == "https://example.com/hook"
        assert redacted["webhook_auth_header_name"] == "X-Webhook-Auth"
        # Original body must not be mutated.
        assert body["webhook_auth_header_value"] == "super-secret-token"

    def test_redact_helper_should_no_op_when_no_secret_present(self):
        body = {"model": "stt-async-v4", "audio_url": "https://example.com/a.wav"}
        redacted = SonioxAudioTranscriptionHandler._redact_body_for_logging(body)
        assert redacted == body
        # Must not introduce a placeholder secret field.
        assert "webhook_auth_header_value" not in redacted

    def test_redact_helper_should_handle_empty_body(self):
        assert SonioxAudioTranscriptionHandler._redact_body_for_logging({}) == {}

    def test_redact_helper_should_skip_none_secret_value(self):
        # A None-valued secret field is treated as absent (the create-body
        # builder already drops Nones, but redact must agree).
        body = {"model": "stt-async-v4", "webhook_auth_header_value": None}
        redacted = SonioxAudioTranscriptionHandler._redact_body_for_logging(body)
        assert redacted["webhook_auth_header_value"] is None

    def test_should_redact_secret_in_pre_and_post_call_logging(self, monkeypatch):
        """End-to-end: real request body keeps the secret, logging hooks don't."""
        monkeypatch.setattr("time.sleep", lambda *_: None)
        responses = {
            "POST https://api.soniox.com/v1/transcriptions": [
                _make_response({"id": "tx_1"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1": [
                _make_response({"status": "completed"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1/transcript": [
                _make_response({"text": "ok", "tokens": []}),
            ],
            "DELETE https://api.soniox.com/v1/transcriptions/tx_1": [
                _make_response({}),
            ],
        }
        client = _MockSyncClient(responses)
        logging_obj = _make_logging_obj()

        call_kwargs = _common_call_kwargs(client)
        call_kwargs["logging_obj"] = logging_obj

        SonioxAudioTranscriptionHandler().audio_transcriptions(
            audio_file=None,
            optional_params={
                "audio_url": "https://example.com/a.wav",
                "webhook_url": "https://example.com/hook",
                "webhook_auth_header_name": "X-Webhook-Auth",
                "webhook_auth_header_value": "super-secret-token",
            },
            litellm_params={},
            atranscription=False,
            **call_kwargs,
        )

        # 1. Real Soniox request must carry the real secret.
        post_call = next(c for c in client.calls if c["method"] == "POST")
        assert post_call["json"]["webhook_auth_header_value"] == "super-secret-token"

        # 2. Pre-call logging must receive a redacted body.
        pre_call_body = logging_obj.pre_call.call_args.kwargs["additional_args"][
            "complete_input_dict"
        ]
        assert pre_call_body["webhook_auth_header_value"] == "[REDACTED]"
        # Non-secret fields unchanged.
        assert pre_call_body["webhook_url"] == "https://example.com/hook"
        assert pre_call_body["webhook_auth_header_name"] == "X-Webhook-Auth"

        # 3. Post-call logging must also receive a redacted body.
        post_call_body = logging_obj.post_call.call_args.kwargs["additional_args"][
            "complete_input_dict"
        ]
        assert post_call_body["webhook_auth_header_value"] == "[REDACTED]"


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

    def test_should_run_async_file_upload_flow(self, monkeypatch):
        async def _no_sleep(*_a, **_kw):
            return None

        monkeypatch.setattr(asyncio, "sleep", _no_sleep)

        responses = {
            "POST https://api.soniox.com/v1/files": [
                _make_response({"id": "file_async_1"}),
            ],
            "POST https://api.soniox.com/v1/transcriptions": [
                _make_response({"id": "tx_async_2"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_async_2": [
                _make_response({"status": "queued"}),
                _make_response({"status": "completed"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_async_2/transcript": [
                _make_response({"text": "async upload ok", "tokens": []}),
            ],
            "DELETE https://api.soniox.com/v1/transcriptions/tx_async_2": [
                _make_response({}),
            ],
            "DELETE https://api.soniox.com/v1/files/file_async_1": [
                _make_response({}),
            ],
        }
        client = _MockAsyncClient(responses)

        coro = SonioxAudioTranscriptionHandler().audio_transcriptions(
            audio_file=("clip.wav", b"RIFFfake", "audio/wav"),
            optional_params={"soniox_polling_interval": 0},
            litellm_params={},
            atranscription=True,
            **_common_call_kwargs(client),
        )
        resp = asyncio.new_event_loop().run_until_complete(coro)
        assert resp.text == "async upload ok"
        deletes = [c["url"] for c in client.calls if c["method"] == "DELETE"]
        assert "https://api.soniox.com/v1/files/file_async_1" in deletes

    def test_should_raise_async_when_status_is_error(self, monkeypatch):
        async def _no_sleep(*_a, **_kw):
            return None

        monkeypatch.setattr(asyncio, "sleep", _no_sleep)

        responses = {
            "POST https://api.soniox.com/v1/transcriptions": [
                _make_response({"id": "tx_err"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_err": [
                _make_response({"status": "error", "error_message": "async boom"}),
            ],
        }
        client = _MockAsyncClient(responses)

        coro = SonioxAudioTranscriptionHandler().audio_transcriptions(
            audio_file=None,
            optional_params={
                "audio_url": "https://example.com/a.wav",
                "soniox_cleanup": [],
            },
            litellm_params={},
            atranscription=True,
            **_common_call_kwargs(client),
        )
        with pytest.raises(SonioxException) as exc_info:
            asyncio.new_event_loop().run_until_complete(coro)
        assert "async boom" in str(exc_info.value)

    def test_should_raise_async_when_polling_attempts_exceeded(self, monkeypatch):
        async def _no_sleep(*_a, **_kw):
            return None

        monkeypatch.setattr(asyncio, "sleep", _no_sleep)

        responses = {
            "POST https://api.soniox.com/v1/transcriptions": [
                _make_response({"id": "tx_timeout"}),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_timeout": [
                _make_response({"status": "processing"}),
                _make_response({"status": "processing"}),
            ],
        }
        client = _MockAsyncClient(responses)

        coro = SonioxAudioTranscriptionHandler().audio_transcriptions(
            audio_file=None,
            optional_params={
                "audio_url": "https://example.com/a.wav",
                "soniox_polling_interval": 0,
                "soniox_max_polling_attempts": 2,
                "soniox_cleanup": [],
            },
            litellm_params={},
            atranscription=True,
            **_common_call_kwargs(client),
        )
        with pytest.raises(SonioxException) as exc_info:
            asyncio.new_event_loop().run_until_complete(coro)
        assert exc_info.value.status_code == 504

    def test_should_raise_async_when_no_audio_input_provided(self):
        client = _MockAsyncClient({})
        coro = SonioxAudioTranscriptionHandler().audio_transcriptions(
            audio_file=None,
            optional_params={},
            litellm_params={},
            atranscription=True,
            **_common_call_kwargs(client),
        )
        with pytest.raises(SonioxException) as exc_info:
            asyncio.new_event_loop().run_until_complete(coro)
        assert exc_info.value.status_code == 400


class TestSpendTracking:
    """Soniox transcriptions must be billed by audio duration.

    The handler stores ``audio_transcription_duration`` and the model is
    priced per second; if either is missing the cost collapses to $0 and an
    authenticated caller transcribes for free.
    """

    @pytest.fixture(autouse=True)
    def _use_local_model_cost_map(self, monkeypatch):
        import litellm

        original_model_cost = litellm.model_cost
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        litellm.model_cost = litellm.get_model_cost_map(url="")
        litellm.get_model_info.cache_clear()
        try:
            yield
        finally:
            litellm.model_cost = original_model_cost
            litellm.get_model_info.cache_clear()

    def test_should_charge_by_audio_duration(self, monkeypatch):
        import litellm

        monkeypatch.setattr("time.sleep", lambda *_: None)
        responses = {
            "POST https://api.soniox.com/v1/transcriptions": [
                _make_response({"id": "tx_1", "status": "queued"})
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1": [
                _make_response(
                    {"id": "tx_1", "status": "completed", "audio_duration_ms": 600000}
                ),
            ],
            "GET https://api.soniox.com/v1/transcriptions/tx_1/transcript": [
                _make_response({"text": "hello world", "tokens": []}),
            ],
            "DELETE https://api.soniox.com/v1/transcriptions/tx_1": [
                _make_response({"deleted": True}),
            ],
        }

        resp = SonioxAudioTranscriptionHandler().audio_transcriptions(
            audio_file=None,
            optional_params={"audio_url": "https://example.com/a.wav"},
            litellm_params={},
            atranscription=False,
            **_common_call_kwargs(_MockSyncClient(responses)),
        )

        assert resp._hidden_params["audio_transcription_duration"] == pytest.approx(
            600.0
        )

        cost = litellm.completion_cost(
            completion_response=resp,
            model="soniox/stt-async-v4",
            call_type="transcription",
        )
        # 10 minutes of audio billed at Soniox's ~$0.10/hour async rate.
        assert cost > 0
        assert cost == pytest.approx((0.10 / 3600) * 600.0, rel=1e-3)
