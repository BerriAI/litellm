"""Behavior pins for ``proxy_server.py`` audio routes.

Pins (PR2):
    - POST /v1/audio/speech
    - POST /audio/speech
    - POST /v1/audio/transcriptions
    - POST /audio/transcriptions
"""

from __future__ import annotations

import io
import json
import math
import wave
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractContextManager
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx
from fastapi import HTTPException
from fastapi.testclient import TestClient
from openai import AsyncOpenAI
from starlette.datastructures import FormData, UploadFile

import litellm
from litellm import Router
from litellm.caching.caching import Cache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy import proxy_server
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.route_llm_request import route_request
from litellm.proxy.utils import _failure_fields_to_lift
from litellm.types.guardrails import GuardrailEventHooks, GuardrailStreamScope
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.proxy.policy_engine.pipeline_types import GuardrailPipeline, PipelineStep
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
def patched_transcription_error(monkeypatch, patched_transcription):
    async def _raise(*args, **kwargs):
        raise ValueError("transcription boom")

    monkeypatch.setattr(proxy_server, "route_request", _raise)
    yield


@pytest.fixture
def patched_transcription_stream(monkeypatch, patched_transcription):
    class _FakeEvent:
        def model_dump_json(self, exclude_unset: bool = False) -> str:
            return '{"type":"transcript.text.done","text":"hello world"}'

    class _FakeAsyncStream:
        def __init__(self):
            self.closed = False

        def __aiter__(self):
            async def _events():
                yield _FakeEvent()

            return _events()

        async def aclose(self):
            self.closed = True

    async def _form_data(request):
        from starlette.datastructures import FormData, UploadFile

        upload = UploadFile(
            filename="audio.mp3",
            file=io.BytesIO(b"\x00\x01\x02"),
        )
        return FormData([("file", upload), ("model", "gpt-transcribe"), ("stream", "true")])

    stream = _FakeAsyncStream()

    async def _llm_call():
        return stream

    async def _fake_route_request(*args, **kwargs):
        return _llm_call()

    monkeypatch.setattr(proxy_server, "get_form_data", _form_data)
    monkeypatch.setattr(proxy_server, "route_request", _fake_route_request)
    yield stream


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
def test_audio_transcription_error(client, auth_as, patched_transcription_error, path):
    """Pins ``POST /v1/audio/transcriptions`` / ``POST /audio/transcriptions`` (error)."""
    files = {"file": ("audio.mp3", b"\x00\x01\x02", "audio/mpeg")}
    data = {"model": "whisper-1"}
    with auth_as():
        response = client.post(path, files=files, data=data)
    assert response.status_code == 500
    assert len(response.content) > 0


@pytest.mark.parametrize("path", ["/v1/audio/transcriptions", "/audio/transcriptions"])
def test_audio_transcription_stream_returns_sse(client, auth_as, patched_transcription_stream, path):
    files = {"file": ("audio.mp3", b"\x00\x01\x02", "audio/mpeg")}
    data = {"model": "gpt-transcribe", "stream": "true"}
    with auth_as():
        response = client.post(path, files=files, data=data)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == 'data: {"type":"transcript.text.done","text":"hello world"}\n\ndata: [DONE]\n\n'
    assert patched_transcription_stream.closed is True


@pytest.mark.usefixtures("patched_transcription_stream")
@pytest.mark.parametrize("configuration", ["default", "policy"])
def test_streaming_transcription_keeps_streaming_with_output_guardrails(
    client: TestClient,
    auth_as: Callable[[], AbstractContextManager[None]],
    monkeypatch: pytest.MonkeyPatch,
    configuration: str,
) -> None:
    guardrail: Final = CustomGuardrail(
        guardrail_name="transcription-output",
        event_hook=GuardrailEventHooks.post_call,
        default_on=configuration == "default",
    )
    monkeypatch.setattr(litellm, "callbacks", [guardrail])
    if configuration == "policy":
        pipeline: Final = GuardrailPipeline(mode="post_call", steps=[PipelineStep(guardrail="transcription-output")])
        proxy_server.proxy_logging_obj.pre_call_hook.side_effect = lambda **kwargs: {
            **kwargs["data"],
            "metadata": {"_guardrail_pipelines": [("transcription-policy", pipeline)]},
        }

    with auth_as():
        response: Final = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", b"audio", "audio/wav")},
            data={"model": "gpt-transcribe", "stream": "true"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"text":"hello world"' in response.text


@pytest.mark.usefixtures("patched_transcription_stream")
def test_streamed_transcription_returns_the_litellm_call_and_model_headers(
    client: TestClient,
    auth_as: Callable[[], AbstractContextManager[None]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream: Final = AsyncOpenAI(
        api_key="sk-test",
        base_url="https://upstream.example/v1",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    content=b'data: {"type":"transcript.text.done","text":"hello world"}\n\n',
                    headers={"content-type": "text/event-stream"},
                )
            )
        ),
    )

    async def route_to_the_sdk(
        data: Mapping[str, object], route_type: str, llm_router: object, user_model: object
    ) -> object:
        return litellm.atranscription(
            model="openai/gpt-transcribe",
            file=data["file"],
            stream=True,
            client=upstream,
            litellm_call_id=data["litellm_call_id"],
            model_info={"id": "transcription-deployment"},
        )

    monkeypatch.setattr(proxy_server, "route_request", route_to_the_sdk)

    with auth_as():
        response: Final = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", b"audio", "audio/wav")},
            data={"model": "gpt-transcribe", "stream": "true"},
            headers={"x-litellm-call-id": "streamed-transcription-call"},
        )

    assert response.status_code == 200
    assert '"text":"hello world"' in response.text
    assert response.headers["x-litellm-call-id"] == "streamed-transcription-call"
    assert response.headers["x-litellm-model-id"] == "transcription-deployment"
    assert [name for name in response.headers if name.startswith("x-litellm-response-cost")] == []
    proxy_server.proxy_logging_obj.update_request_status.assert_called_once_with(
        litellm_call_id="streamed-transcription-call", status="success"
    )


class _TranscriptRedactor(CustomGuardrail):
    def __init__(self, stream_scope: GuardrailStreamScope = "both") -> None:
        super().__init__(
            guardrail_name="transcript-redactor", event_hook=GuardrailEventHooks.post_call, stream_scope=stream_scope
        )

    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: object, response: object
    ) -> None:
        if isinstance(response, TranscriptionResponse):
            response.text = "[REDACTED]"


def _streams_when_asked(request: httpx.Request) -> httpx.Response:
    if b'name="stream"' in request.content:
        return httpx.Response(
            200,
            content=b'data: {"type":"transcript.text.done","text":"my pin is 1234"}\n\n',
            headers={"content-type": "text/event-stream"},
        )
    return httpx.Response(200, json={"text": "my pin is 1234", "usage": {"type": "duration", "seconds": 2}})


def _sse_events(body: str) -> tuple[object, ...]:
    assert body.endswith("data: [DONE]\n\n")
    return tuple(json.loads(frame.removeprefix("data: ")) for frame in body.removesuffix("data: [DONE]\n\n").split("\n\n") if frame)


@pytest.mark.usefixtures("patched_transcription_stream")
@pytest.mark.parametrize("deployment_guardrails", (["transcript-redactor"], None))
def test_streamed_transcription_honors_a_deployment_post_call_guardrail(
    client: TestClient,
    auth_as: Callable[[], AbstractContextManager[None]],
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    deployment_guardrails: list[str] | None,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "callbacks", [_TranscriptRedactor()])
    respx_mock.post("https://transcribe-upstream.example/v1/audio/transcriptions").mock(side_effect=_streams_when_asked)
    deployment_params: Final = {
        "model": "openai/gpt-transcribe",
        "api_key": "sk-test",
        "api_base": "https://transcribe-upstream.example/v1",
    }
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-transcribe",
                    "litellm_params": (
                        {**deployment_params, "guardrails": deployment_guardrails}
                        if deployment_guardrails
                        else deployment_params
                    ),
                }
            ]
        ),
    )
    monkeypatch.setattr(proxy_server, "route_request", route_request)

    with auth_as():
        response: Final = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", b"audio", "audio/wav")},
            data={"model": "gpt-transcribe", "stream": "true"},
        )

    assert response.status_code == 200
    if deployment_guardrails:
        assert response.headers["content-type"].startswith("text/event-stream")
        assert _sse_events(response.text) == (
            {"type": "transcript.text.delta", "delta": "[REDACTED]"},
            {"type": "transcript.text.done", "text": "[REDACTED]", "usage": {"type": "duration", "seconds": 2}},
        )
    else:
        assert response.headers["content-type"].startswith("text/event-stream")
        assert '"text":"my pin is 1234"' in response.text


@pytest.mark.usefixtures("patched_transcription_stream")
@pytest.mark.parametrize(
    "stream_scope,upstream_streamed,expected_text",
    [("streaming", False, "[REDACTED]"), ("non_streaming", True, "my pin is 1234")],
)
def test_streamed_transcription_applies_a_deployment_guardrail_only_in_its_stream_scope(
    client: TestClient,
    auth_as: Callable[[], AbstractContextManager[None]],
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    stream_scope: GuardrailStreamScope,
    upstream_streamed: bool,
    expected_text: str,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "callbacks", [_TranscriptRedactor(stream_scope=stream_scope)])
    upstream_route: Final = respx_mock.post("https://transcribe-upstream.example/v1/audio/transcriptions").mock(
        side_effect=_streams_when_asked
    )
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-transcribe",
                    "litellm_params": {
                        "model": "openai/gpt-transcribe",
                        "api_key": "sk-test",
                        "api_base": "https://transcribe-upstream.example/v1",
                        "guardrails": ["transcript-redactor"],
                    },
                }
            ]
        ),
    )
    monkeypatch.setattr(proxy_server, "route_request", route_request)

    with auth_as():
        response: Final = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", b"audio", "audio/wav")},
            data={"model": "gpt-transcribe", "stream": "true"},
        )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    assert (b'name="stream"' in upstream_route.calls.last.request.content) is upstream_streamed
    assert _sse_events(response.text)[-1]["text"] == expected_text


@pytest.mark.usefixtures("patched_transcription_stream")
def test_cached_guarded_transcription_still_streams_to_a_stream_client(
    client: TestClient,
    auth_as: Callable[[], AbstractContextManager[None]],
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "callbacks", [_TranscriptRedactor()])
    monkeypatch.setattr(litellm, "cache", Cache(type="local"))
    upstream_route: Final = respx_mock.post("https://transcribe-upstream.example/v1/audio/transcriptions").mock(
        side_effect=_streams_when_asked
    )
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-transcribe",
                    "litellm_params": {
                        "model": "openai/gpt-transcribe",
                        "api_key": "sk-test",
                        "api_base": "https://transcribe-upstream.example/v1",
                        "guardrails": ["transcript-redactor"],
                    },
                }
            ],
            cache_responses=True,
        ),
    )
    monkeypatch.setattr(proxy_server, "route_request", route_request)

    with auth_as():
        responses: Final = [
            client.post(
                "/v1/audio/transcriptions",
                files={"file": ("sample.wav", b"audio", "audio/wav")},
                data={"model": "gpt-transcribe", "stream": "true"},
            )
            for _ in range(2)
        ]

    assert upstream_route.call_count == 1
    assert [response.headers["content-type"].split(";")[0] for response in responses] == ["text/event-stream"] * 2
    assert _sse_events(responses[1].text)[-1] == {
        "type": "transcript.text.done",
        "text": "[REDACTED]",
        "usage": {"type": "duration", "seconds": 2},
    }


@pytest.mark.usefixtures("patched_transcription_stream")
def test_streamed_transcription_frames_keep_the_provider_event_shape(
    client: TestClient,
    auth_as: Callable[[], AbstractContextManager[None]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_events: Final = (
        {"type": "transcript.text.delta", "delta": "hello"},
        {"type": "transcript.text.done", "text": "hello", "usage": {"type": "duration", "seconds": 2}},
    )
    upstream: Final = AsyncOpenAI(
        api_key="sk-test",
        base_url="https://upstream.example/v1",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    content="".join(f"data: {json.dumps(event)}\n\n" for event in provider_events).encode(),
                    headers={"content-type": "text/event-stream"},
                )
            )
        ),
    )

    async def route_to_the_sdk(
        data: Mapping[str, object], route_type: str, llm_router: object, user_model: object
    ) -> object:
        return litellm.atranscription(
            model="openai/gpt-transcribe", file=data["file"], stream=data["stream"], client=upstream
        )

    monkeypatch.setattr(proxy_server, "route_request", route_to_the_sdk)

    with auth_as():
        response: Final = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", b"audio", "audio/wav")},
            data={"model": "gpt-transcribe", "stream": "true"},
        )

    assert response.status_code == 200
    assert _sse_events(response.text) == provider_events


class _FirstDeltaThenReset(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'data: {"type":"transcript.text.delta","delta":"Good"}\n\n'
        raise httpx.ReadError("connection reset")


def _silent_wav(seconds: int) -> bytes:
    buffer: Final = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 16_000 * seconds)
    return buffer.getvalue()


@pytest.mark.usefixtures("patched_transcription")
def test_a_streamed_transcription_that_fails_mid_transcript_bills_the_uploaded_seconds(
    client: TestClient,
    auth_as: Callable[[], AbstractContextManager[None]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream: Final = AsyncOpenAI(
        api_key="sk-test",
        base_url="https://upstream.example/v1",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, stream=_FirstDeltaThenReset()
                )
            )
        ),
    )

    async def route_to_the_sdk(
        data: Mapping[str, object], route_type: str, llm_router: object, user_model: object
    ) -> object:
        return litellm.atranscription(model="openai/gpt-transcribe", file=data["file"], stream=True, client=upstream)

    four_second_upload: Final = _silent_wav(4)

    async def stream_form(request: object) -> FormData:
        return FormData(
            [
                ("file", UploadFile(filename="sample.wav", file=io.BytesIO(four_second_upload))),
                ("model", "gpt-transcribe"),
                ("stream", "true"),
            ]
        )

    monkeypatch.setattr(proxy_server, "get_form_data", stream_form)
    monkeypatch.setattr(proxy_server, "route_request", route_to_the_sdk)

    with auth_as():
        response: Final = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", four_second_upload, "audio/wav")},
            data={"model": "gpt-transcribe", "stream": "true"},
        )

    assert response.status_code == 200
    assert '"error"' in response.text
    failure_request: Final = proxy_server.proxy_logging_obj.post_call_failure_hook.await_args.kwargs["request_data"]
    billed_cost: Final = _failure_fields_to_lift(failure_request)["response_cost"]
    assert isinstance(billed_cost, float)
    assert math.isclose(billed_cost, 4 * litellm.model_cost["gpt-transcribe"]["input_cost_per_second"])


async def _scribe_stream_form(request: object) -> FormData:
    return FormData(
        [
            ("file", UploadFile(filename="sample.wav", file=io.BytesIO(b"audio"))),
            ("model", "scribe"),
            ("stream", "true"),
        ]
    )


@pytest.mark.usefixtures("patched_transcription")
def test_transcription_stream_form_field_reaches_other_providers_as_sent(
    client: TestClient,
    auth_as: Callable[[], AbstractContextManager[None]],
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    upstream: Final = respx_mock.post("https://elevenlabs.example/v1/speech-to-text").mock(
        return_value=httpx.Response(200, json={"text": "hello world", "language_code": "en"})
    )
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "scribe",
                    "litellm_params": {
                        "model": "elevenlabs/scribe_v1",
                        "api_key": "sk-test",
                        "api_base": "https://elevenlabs.example",
                    },
                }
            ]
        ),
    )
    monkeypatch.setattr(proxy_server, "route_request", route_request)
    monkeypatch.setattr(proxy_server, "get_form_data", _scribe_stream_form)

    with auth_as():
        response: Final = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", b"audio", "audio/wav")},
            data={"model": "scribe", "stream": "true"},
        )

    assert response.status_code == 200
    assert response.json()["text"] == "hello world"
    assert b'name="stream"\r\n\r\ntrue\r\n' in upstream.calls.last.request.content


def _streams_chat_and_transcription(request: httpx.Request) -> httpx.Response:
    sse_body: Final = (
        b'data: {"id":"chat-1","object":"chat.completion.chunk","created":0,"model":"gpt-4o",'
        b'"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        if request.url.path.endswith("/chat/completions")
        else b'data: {"type":"transcript.text.done","text":"hello world"}\n\n'
    )
    return httpx.Response(
        200,
        content=sse_body,
        headers={"content-type": "text/event-stream", "x-ratelimit-remaining-requests": "99"},
    )


def _litellm_header_names(response: httpx.Response) -> frozenset[str]:
    return frozenset(
        name
        for name in response.headers
        if name.startswith("x-litellm-") and not name.startswith("x-litellm-response-cost")
    )


def test_streamed_transcription_carries_the_headers_a_streamed_chat_completion_carries(
    client: TestClient,
    auth_as: Callable[[], AbstractContextManager[None]],
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.route(host="stream-upstream.example").mock(side_effect=_streams_chat_and_transcription)
    upstream: Final = {"api_key": "sk-test", "api_base": "https://stream-upstream.example/v1"}
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {"model_name": "gpt-4o", "litellm_params": {"model": "openai/gpt-4o", **upstream}},
                {"model_name": "gpt-transcribe", "litellm_params": {"model": "openai/gpt-transcribe", **upstream}},
            ]
        ),
    )

    with auth_as():
        chat: Final = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        )
        transcription: Final = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", b"audio", "audio/wav")},
            data={"model": "gpt-transcribe", "stream": "true"},
        )

    assert chat.status_code == 200
    assert transcription.status_code == 200
    assert transcription.headers["content-type"].startswith("text/event-stream")
    assert _litellm_header_names(chat) - _litellm_header_names(transcription) <= {"x-litellm-overhead-duration-ms"}
    assert transcription.headers["x-litellm-model-name"] == "openai/gpt-transcribe"
    assert transcription.headers["llm_provider-x-ratelimit-remaining-requests"] == "99"
    assert [name for name in transcription.headers if name.startswith("x-litellm-response-cost")] == []


class _FailsAfterTwoDeltas(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'data: {"type":"transcript.text.delta","delta":"hello "}\n\n'
        yield b'data: {"type":"transcript.text.delta","delta":"world"}\n\n'
        raise httpx.ReadError("upstream connection dropped")


@pytest.mark.usefixtures("patched_transcription_stream")
def test_streamed_transcription_error_mid_stream_ends_with_an_error_event_and_runs_the_failure_hook(
    client: TestClient,
    auth_as: Callable[[], AbstractContextManager[None]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream: Final = AsyncOpenAI(
        api_key="sk-test",
        base_url="https://upstream.example/v1",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, stream=_FailsAfterTwoDeltas(), headers={"content-type": "text/event-stream"}
                )
            )
        ),
    )

    async def route_to_the_sdk(
        data: Mapping[str, object], route_type: str, llm_router: object, user_model: object
    ) -> object:
        return litellm.atranscription(model="openai/gpt-transcribe", file=data["file"], stream=True, client=upstream)

    failure_hook: Final = AsyncMock(return_value=None)
    monkeypatch.setattr(proxy_server.proxy_logging_obj, "post_call_failure_hook", failure_hook)
    monkeypatch.setattr(proxy_server, "route_request", route_to_the_sdk)

    with auth_as():
        response: Final = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", b"audio", "audio/wav")},
            data={"model": "gpt-transcribe", "stream": "true"},
        )

    data_lines: Final = tuple(
        line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")
    )
    frames: Final = tuple(json.loads(line) for line in data_lines[:-1])
    assert response.status_code == 200
    assert tuple(frame.get("delta") for frame in frames[:2]) == ("hello ", "world")
    assert len(frames) == 3
    assert data_lines[-1] == "[DONE]"
    assert "upstream connection dropped" in frames[2]["error"]["message"]
    assert set(frames[2]["error"]) == {"message", "type", "param", "code"}
    assert (frames[2]["error"]["type"], frames[2]["error"]["param"], frames[2]["error"]["code"]) == (
        "internal_server_error",
        None,
        "500",
    )
    failure_hook.assert_awaited_once()
    assert isinstance(failure_hook.await_args.kwargs["original_exception"], litellm.APIConnectionError)
