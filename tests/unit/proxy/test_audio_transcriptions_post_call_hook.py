"""The /audio/transcriptions route runs the post-call hooks on the finished transcript."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import litellm.proxy.proxy_server as ps
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.proxy_server import app
from litellm.types.utils import TranscriptionResponse


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test-user"
    )
    monkeypatch.setattr(ps, "llm_router", None)
    monkeypatch.setattr(ps, "general_settings", {})
    yield TestClient(app)
    app.dependency_overrides.pop(ps.user_api_key_auth, None)


def _transcribe(client: TestClient, response_format: str = "json"):
    return client.post(
        "/v1/audio/transcriptions",
        headers={"Authorization": "Bearer sk-test"},
        data={"model": "whisper-1", "response_format": response_format},
        files={"file": ("a.wav", b"RIFF....WAVE", "audio/wav")},
    )


def test_the_transcript_goes_through_post_call_success_hook(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        ps, "route_request", AsyncMock(return_value=_awaitable(TranscriptionResponse(text="my code is 1234")))
    )
    hook = AsyncMock(return_value=TranscriptionResponse(text="my code is [MASKED]"))
    monkeypatch.setattr(ps.proxy_logging_obj, "post_call_success_hook", hook)

    result = _transcribe(client)

    assert result.status_code == 200
    assert result.json()["text"] == "my code is [MASKED]"
    assert hook.await_args.kwargs["response"].text == "my code is 1234"


def test_a_plain_text_transcript_is_screened_too(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        ps, "route_request", AsyncMock(return_value=_awaitable(TranscriptionResponse(text="my code is 1234")))
    )
    monkeypatch.setattr(
        ps.proxy_logging_obj,
        "post_call_success_hook",
        AsyncMock(return_value=TranscriptionResponse(text="my code is [MASKED]")),
    )

    assert _transcribe(client, "text").text == "my code is [MASKED]"


def test_a_guardrail_that_blocks_fails_the_request(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ps, "route_request", AsyncMock(return_value=_awaitable(TranscriptionResponse(text="secret"))))
    monkeypatch.setattr(ps.proxy_logging_obj, "post_call_success_hook", AsyncMock(side_effect=ValueError("blocked")))

    assert _transcribe(client).status_code >= 400


async def _awaitable(value: TranscriptionResponse) -> TranscriptionResponse:
    return value
