"""Behavior pins for ``proxy_server.py`` audio routes.

Pins (PR2):
    - POST /v1/audio/speech
    - POST /audio/speech
    - POST /v1/audio/transcriptions
    - POST /audio/transcriptions
"""

from __future__ import annotations

import io
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from litellm import Router
from litellm.proxy import proxy_server
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.common_utils.http_parsing_utils import get_form_data
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.router import RouterErrors
from litellm.types.utils import TranscriptionResponse

_TAG_ROUTING_DENIAL = f"{RouterErrors.no_deployments_with_tag_routing.value}. Passed model=tts-1 and tags=['x']"


@pytest.fixture
def patched_speech(monkeypatch, request):
    upstream_content_type = getattr(request, "param", "audio/mpeg")
    monkeypatch.setattr(proxy_server, "llm_router", MagicMock())
    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        MagicMock(
            pre_call_hook=AsyncMock(side_effect=lambda **kw: kw["data"]),
            post_call_failure_hook=AsyncMock(),
            post_call_response_headers_hook=AsyncMock(return_value={}),
            update_request_status=AsyncMock(),
        ),
    )

    async def _add_data(data, **kwargs):
        return data

    monkeypatch.setattr(proxy_server, "add_litellm_data_to_request", _add_data)

    async def _llm_call():
        return HttpxBinaryResponseContent(
            httpx.Response(
                status_code=200,
                headers={} if upstream_content_type is None else {"content-type": upstream_content_type},
                content=b"\x00\x01\x02",
            )
        )

    async def _fake_route_request(*args, **kwargs):
        return _llm_call()

    monkeypatch.setattr(proxy_server, "route_request", _fake_route_request)
    yield


@pytest.fixture
def patched_speech_error(monkeypatch):
    monkeypatch.setattr(proxy_server, "llm_router", MagicMock())
    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        MagicMock(
            pre_call_hook=AsyncMock(side_effect=lambda **kw: kw["data"]),
            post_call_failure_hook=AsyncMock(),
            post_call_response_headers_hook=AsyncMock(return_value={}),
            update_request_status=AsyncMock(),
        ),
    )

    async def _add_data(data, **kwargs):
        return data

    monkeypatch.setattr(proxy_server, "add_litellm_data_to_request", _add_data)

    async def _raise(*args, **kwargs):
        raise ValueError("speech boom")

    monkeypatch.setattr(proxy_server, "route_request", _raise)
    yield


@pytest.fixture
def patched_speech_provider_rejection(monkeypatch, patched_speech_error):
    import litellm

    async def _raise(*args, **kwargs):
        raise litellm.BadRequestError(
            message=(
                "Gemini TTS only produces raw PCM16 audio, so response_format='mp3' is not supported."
                " Supported response formats: pcm, wav."
            ),
            model="gemini-3.1-flash-tts-preview",
            llm_provider="gemini",
        )

    monkeypatch.setattr(proxy_server, "route_request", _raise)
    yield


@pytest.fixture
def patched_transcription(monkeypatch):
    router = MagicMock()
    router.model_names = ["whisper-1"]
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        MagicMock(
            pre_call_hook=AsyncMock(side_effect=lambda **kw: kw["data"]),
            post_call_failure_hook=AsyncMock(),
            post_call_response_headers_hook=AsyncMock(return_value={}),
            update_request_status=AsyncMock(),
        ),
    )

    async def _add_data(data, **kwargs):
        return data

    monkeypatch.setattr(proxy_server, "add_litellm_data_to_request", _add_data)
    monkeypatch.setattr(proxy_server, "check_file_size_under_limit", lambda **kwargs: True)

    async def _form_data(request):
        from starlette.datastructures import FormData, UploadFile

        upload = UploadFile(
            filename="audio.mp3",
            file=io.BytesIO(b"\x00\x01\x02"),
        )
        return FormData([("file", upload), ("model", "whisper-1")])

    monkeypatch.setattr(proxy_server, "get_form_data", _form_data)

    async def _llm_call():
        return {"text": "hello world"}

    async def _fake_route_request(*args, **kwargs):
        return _llm_call()

    monkeypatch.setattr(proxy_server, "route_request", _fake_route_request)
    yield


@pytest.fixture
def patched_transcription_error(monkeypatch: pytest.MonkeyPatch, patched_transcription: None) -> AsyncMock:
    monkeypatch.setattr(proxy_server, "get_form_data", get_form_data)

    async def _raise(*args, **kwargs):
        raise ValueError("transcription boom")

    route_request: Final[AsyncMock] = AsyncMock(side_effect=_raise)
    monkeypatch.setattr(proxy_server, "route_request", route_request)
    return route_request


@pytest.fixture
def patched_transcription_http(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    transcript: Final = getattr(request, "param", "hello world")
    router: Final = MagicMock()
    router.model_names = ["whisper-1"]
    router.get_deployment_by_model_group_name.return_value = None
    monkeypatch.setattr(proxy_server, "llm_router", router)

    async def _response_headers(data: dict[str, object], **kwargs: object) -> dict[str, str]:
        return {
            "x-callback-trace": "callback-value",
            **(
                {"Content-Type": "application/json", "Content-Length": "999"}
                if data.get("response_format") == "text"
                else {}
            ),
        }

    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        MagicMock(
            pre_call_hook=AsyncMock(side_effect=lambda **kw: kw["data"]),
            post_call_failure_hook=AsyncMock(),
            post_call_response_headers_hook=AsyncMock(side_effect=_response_headers),
            update_request_status=AsyncMock(),
        ),
    )

    async def _add_data(data: dict[str, object], **kwargs: object) -> dict[str, object]:
        return data

    monkeypatch.setattr(proxy_server, "add_litellm_data_to_request", _add_data)

    async def _llm_call() -> TranscriptionResponse | dict[str, str]:
        if not isinstance(transcript, str):
            return {"text": "not a TranscriptionResponse"}
        response: Final = TranscriptionResponse(text=transcript)
        response.hidden_params = {
            "litellm_call_id": "transcription-call-id",
            "model_id": "transcription-model-id",
            "cache_key": "transcription-cache-key",
        }
        return response

    route_request: Final[AsyncMock] = AsyncMock(return_value=_llm_call())
    monkeypatch.setattr(proxy_server, "route_request", route_request)
    return transcript, route_request


@pytest.mark.parametrize("path", ["/v1/audio/speech", "/audio/speech"])
def test_audio_speech_happy_path(client, auth_as, patched_speech, path):
    """Pins ``POST /v1/audio/speech`` and ``POST /audio/speech`` (happy)."""
    payload = {"model": "tts-1", "input": "Hi", "voice": "alloy"}
    with auth_as():
        response = client.post(path, json=payload)
    assert response.status_code == 200
    response_summary = {
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type", ""),
        "body_bytes": response.content,
    }
    assert response_summary == {
        "status_code": 200,
        "content_type": "audio/mpeg",
        "body_bytes": b"\x00\x01\x02",
    }


@pytest.mark.parametrize(
    ("patched_speech", "response_format", "expected_content_type"),
    [
        ("audio/wav", "wav", "audio/wav"),
        ("audio/flac", "flac", "audio/flac"),
        ("audio/pcm", "pcm", "audio/pcm"),
        ("audio/wav", "mp3", "audio/wav"),
        ("application/json", "flac", "audio/flac"),
        (None, "wav", "audio/wav"),
        (None, None, "audio/mpeg"),
    ],
    indirect=["patched_speech"],
)
def test_audio_speech_content_type_matches_audio_format(
    client, auth_as, patched_speech, response_format, expected_content_type
):
    """Regression for LIT-6482: /v1/audio/speech mislabeled wav/flac/pcm as audio/mpeg."""
    payload = {
        "model": "tts-1",
        "input": "Hi",
        "voice": "alloy",
        **({} if response_format is None else {"response_format": response_format}),
    }
    with auth_as():
        response = client.post("/v1/audio/speech", json=payload)
    assert response.status_code == 200
    assert response.headers.get("content-type", "").split(";")[0] == expected_content_type


@pytest.mark.parametrize("path", ["/v1/audio/speech", "/audio/speech"])
def test_audio_speech_error(client, auth_as, patched_speech_error, path):
    """Pins ``POST /v1/audio/speech`` and ``POST /audio/speech`` (error)."""
    payload = {"model": "tts-1", "input": "Hi", "voice": "alloy"}
    with auth_as():
        response = client.post(path, json=payload)
    assert response.status_code == 500
    error = response.json()["error"]
    assert (error["type"], error["param"]) == ("internal_server_error", None)


def test_audio_speech_bad_request_maps_to_400(client, auth_as, patched_speech_provider_rejection):
    """Regression for LIT-6501: a BadRequestError from the speech path surfaced as a generic 500."""
    payload = {"model": "gemini-tts", "input": "Hi", "voice": "Kore", "response_format": "mp3"}
    with auth_as():
        response = client.post("/v1/audio/speech", json=payload)
    assert response.status_code == 400
    error = response.json()["error"]
    assert "response_format='mp3'" in error["message"]
    assert "pcm" in error["message"]
    assert "wav" in error["message"]


@pytest.mark.parametrize("path", ["/v1/audio/transcriptions", "/audio/transcriptions"])
def test_audio_transcription_happy_path(client, auth_as, patched_transcription, path):
    """Pins ``POST /v1/audio/transcriptions`` / ``POST /audio/transcriptions`` (happy)."""
    files = {"file": ("audio.mp3", b"\x00\x01\x02", "audio/mpeg")}
    data = {"model": "whisper-1"}
    with auth_as():
        response = client.post(path, files=files, data=data)
    assert response.status_code == 200
    body = response.json()
    assert body == {"text": "hello world"}
    response_summary = {
        "status_code": response.status_code,
        "text_field": body["text"],
        "media_type_hint": response.headers.get("content-type", "").split(";")[0],
    }
    assert response_summary == {
        "status_code": 200,
        "text_field": "hello world",
        "media_type_hint": "application/json",
    }


@pytest.mark.parametrize("path", ["/v1/audio/transcriptions", "/audio/transcriptions"])
def test_audio_transcription_over_the_file_size_cap_is_rejected_as_a_bad_request(
    client, auth_as, patched_transcription, monkeypatch, path
):
    from litellm.proxy.common_utils.http_parsing_utils import check_file_size_under_limit

    capped_router = Router(
        model_list=[
            {"model_name": "whisper-1", "litellm_params": {"model": "openai/whisper-1", "max_file_size_mb": 0.000001}}
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", capped_router)
    monkeypatch.setattr(proxy_server, "premium_user", True)
    monkeypatch.setattr(proxy_server, "check_file_size_under_limit", check_file_size_under_limit)
    files = {"file": ("audio.mp3", b"\x00\x01\x02", "audio/mpeg")}
    with auth_as():
        response = client.post(path, files=files, data={"model": "whisper-1"})
    assert response.status_code == 400
    error = response.json()["error"]
    assert "File size is too large" in error["message"]
    assert error["param"] == "file"


def test_audio_speech_tag_routing_denial_is_typed_as_a_proxy_auth_error(
    client, auth_as, patched_speech_error, monkeypatch
):
    async def _deny(*args, **kwargs):
        raise ValueError(_TAG_ROUTING_DENIAL)

    monkeypatch.setattr(proxy_server, "route_request", _deny)
    with auth_as():
        response = client.post("/v1/audio/speech", json={"model": "tts-1", "input": "Hi", "voice": "alloy"})
    assert response.status_code == 401
    error = response.json()["error"]
    assert (error["type"], error["param"]) == ("auth_error", None)


@pytest.mark.parametrize(
    ("failure", "expected_status", "expected_type"),
    [
        pytest.param(ValueError(_TAG_ROUTING_DENIAL), 401, "auth_error", id="tag_routing_denial"),
        pytest.param(
            HTTPException(status_code=403, detail="Blocked by the content policy"),
            403,
            "permission_error",
            id="hook_policy_block",
        ),
    ],
)
def test_audio_transcription_failure_is_typed_by_who_rejected_it(
    client, auth_as, patched_transcription, monkeypatch, failure, expected_status, expected_type
):
    async def _deny(*args, **kwargs):
        raise failure

    monkeypatch.setattr(proxy_server, "route_request", _deny)
    files = {"file": ("audio.mp3", b"\x00\x01\x02", "audio/mpeg")}
    with auth_as():
        response = client.post("/v1/audio/transcriptions", files=files, data={"model": "whisper-1"})
    assert response.status_code == expected_status
    error = response.json()["error"]
    assert (error["type"], error["param"]) == (expected_type, None)


@pytest.mark.parametrize(
    "rejection",
    [
        pytest.param(
            ProxyException(
                message="Invalid type for 'metadata': expected an object, but got an integer instead.",
                type=ProxyErrorTypes.bad_request_error,
                param="metadata",
                code=400,
            ),
            id="bad_metadata",
        ),
        pytest.param(
            ProxyException(
                message="tag budgets are unavailable", type=ProxyErrorTypes.no_db_connection, param=None, code=503
            ),
            id="tag_budget_store_down",
        ),
    ],
)
def test_audio_transcription_keeps_the_status_of_a_proxy_rejection(
    client, auth_as, patched_transcription, monkeypatch, rejection
):
    async def _reject(data, **kwargs):
        raise rejection

    monkeypatch.setattr(proxy_server, "add_litellm_data_to_request", _reject)
    files = {"file": ("audio.mp3", b"\x00\x01\x02", "audio/mpeg")}
    with auth_as():
        response = client.post("/v1/audio/transcriptions", files=files, data={"model": "whisper-1"})
    assert response.status_code == int(rejection.code)
    assert response.json()["error"] == rejection.to_dict()


@pytest.mark.parametrize("path", ["/v1/audio/transcriptions", "/audio/transcriptions"])
@pytest.mark.parametrize("response_format", [None, "text"], ids=["default", "text"])
def test_audio_transcription_error(
    client: TestClient,
    auth_as: Callable[..., AbstractContextManager[None]],
    patched_transcription_error: AsyncMock,
    path: str,
    response_format: str | None,
) -> None:
    """Pins ``POST /v1/audio/transcriptions`` / ``POST /audio/transcriptions`` (error)."""
    files: Final = {"file": ("audio.mp3", b"\x00\x01\x02", "audio/mpeg")}
    data: Final[dict[str, str]] = {"model": "whisper-1"} | (
        {"response_format": response_format} if response_format is not None else {}
    )
    with auth_as():
        response: Final = client.post(
            path,
            files=files,
            data=data,
            headers={"x-litellm-call-id": "transcription-request-id"},
        )
    assert response.status_code == 500
    assert response.headers["content-type"] == "application/json"
    assert response.headers["x-litellm-call-id"] == "transcription-request-id"
    assert response.json() == {
        "error": {
            "message": "transcription boom",
            "type": "internal_server_error",
            "param": None,
            "code": "500",
        }
    }
    route_request_call: Final = patched_transcription_error.await_args
    assert route_request_call is not None
    assert route_request_call.kwargs["data"].get("response_format") == response_format


@pytest.mark.parametrize("path", ["/v1/audio/transcriptions", "/audio/transcriptions"], ids=["v1", "alias"])
@pytest.mark.parametrize(
    "patched_transcription_http",
    ["hello world", "Grüße, 世界\nline two\n", ""],
    indirect=True,
    ids=["plain", "unicode-multiline", "empty"],
)
def test_audio_transcription_text_format_returns_plain_text(client, auth_as, patched_transcription_http, path):
    transcript, route_request = patched_transcription_http
    files: Final = {"file": ("audio.mp3", b"\x00\x01\x02", "audio/mpeg")}
    data: Final = {"model": "whisper-1", "response_format": "text"}
    with auth_as():
        response: Final = client.post(
            path,
            files=files,
            data=data,
            headers={"x-litellm-call-id": "request-call-id"},
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert response.content == transcript.encode("utf-8")
    assert response.headers["content-length"] == str(len(transcript.encode("utf-8")))
    assert response.headers["x-litellm-call-id"] == "transcription-call-id"
    assert response.headers["x-litellm-model-id"] == "transcription-model-id"
    assert response.headers["x-litellm-cache-key"] == "transcription-cache-key"
    assert response.headers["x-callback-trace"] == "callback-value"
    assert route_request.await_args.kwargs["data"]["response_format"] == "text"


@pytest.mark.parametrize("path", ["/v1/audio/transcriptions", "/audio/transcriptions"], ids=["v1", "alias"])
@pytest.mark.parametrize("response_format", [None, "json"], ids=["default", "json"])
def test_audio_transcription_json_formats_keep_body_and_headers(
    client, auth_as, patched_transcription_http, path, response_format
):
    transcript, route_request = patched_transcription_http
    files: Final = {"file": ("audio.mp3", b"\x00\x01\x02", "audio/mpeg")}
    data: Final = {"model": "whisper-1"} | ({"response_format": response_format} if response_format is not None else {})
    with auth_as():
        response: Final = client.post(
            path,
            files=files,
            data=data,
            headers={"x-litellm-call-id": "request-call-id"},
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"text": transcript, "usage": None}
    assert response.headers["x-litellm-call-id"] == "transcription-call-id"
    assert response.headers["x-litellm-model-id"] == "transcription-model-id"
    assert response.headers["x-litellm-cache-key"] == "transcription-cache-key"
    assert response.headers["x-callback-trace"] == "callback-value"
    assert route_request.await_args.kwargs["data"].get("response_format") == response_format


@pytest.mark.parametrize("path", ["/v1/audio/transcriptions", "/audio/transcriptions"], ids=["v1", "alias"])
@pytest.mark.parametrize("patched_transcription_http", [None], indirect=True, ids=["malformed-response"])
def test_audio_transcription_text_format_rejects_malformed_response(
    client: TestClient,
    auth_as: Callable[..., AbstractContextManager[None]],
    patched_transcription_http: tuple[object, AsyncMock],
    path: str,
):
    files: Final = {"file": ("audio.mp3", b"\x00\x01\x02", "audio/mpeg")}
    data: Final = {"model": "whisper-1", "response_format": "text"}
    with auth_as():
        response: Final = client.post(path, files=files, data=data)

    assert response.status_code == 500
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {
        "error": {
            "message": "Audio transcription response must be a TranscriptionResponse",
            "type": "internal_server_error",
            "param": None,
            "code": "500",
        }
    }
    logging_obj: Final = proxy_server.proxy_logging_obj
    assert isinstance(logging_obj, MagicMock)
    failure_hook: Final = logging_obj.post_call_failure_hook
    assert isinstance(failure_hook, AsyncMock)
    assert failure_hook.await_count == 1
