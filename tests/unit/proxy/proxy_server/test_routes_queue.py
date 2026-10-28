"""Behavior pins for ``proxy_server.py`` queue routes.

Pins (PR2):
    - POST /queue/chat/completions
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from litellm import Router
from litellm.proxy import proxy_server

from .conftest import normalize  # type: ignore[import-not-found]

HAPPY_RESPONSE = {
    "id": "chatcmpl-queue",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "queued reply"},
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    "priority": 0,
}


@pytest.fixture
def patched_queue(monkeypatch):
    router = MagicMock()
    router.split_request_priority = MagicMock(return_value=(0, {}))
    router.schedule_acompletion = AsyncMock(return_value=dict(HAPPY_RESPONSE))
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        MagicMock(post_call_failure_hook=AsyncMock()),
    )
    return router


@pytest.fixture
def queue_no_router(monkeypatch):
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        MagicMock(post_call_failure_hook=AsyncMock()),
    )
    yield


def test_queue_chat_completions_happy(client, auth_as, patched_queue):
    """Pins ``POST /queue/chat/completions`` (happy)."""
    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "priority": 0,
    }
    with auth_as():
        response = client.post("/queue/chat/completions", json=payload)
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "id": "<VOLATILE>",
        "object": "chat.completion",
        "created": "<VOLATILE>",
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "queued reply"},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "priority": 0,
    }


def test_queue_chat_completions_drops_client_sent_stream_usage_flags(
    client: TestClient, auth_as: Callable[..., AbstractContextManager[None]], patched_queue: MagicMock
) -> None:
    payload: Final = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "priority": 0,
        "_litellm_strip_stream_usage": True,
        "_litellm_stream_usage_injected": True,
    }
    with auth_as():
        response: Final = client.post("/queue/chat/completions", json=payload)
    assert response.status_code == 200
    forwarded: Final = patched_queue.schedule_acompletion.call_args.kwargs
    assert "_litellm_strip_stream_usage" not in forwarded
    assert "_litellm_stream_usage_injected" not in forwarded
    assert forwarded["messages"] == [{"role": "user", "content": "hi"}]


def test_queue_chat_completions_no_router_error(client, auth_as, queue_no_router):
    """Pins ``POST /queue/chat/completions`` (error: no llm_router)."""
    payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
    with auth_as():
        response = client.post("/queue/chat/completions", json=payload)
    assert response.status_code == 500
    assert len(response.content) > 0


def _queue_router(
    monkeypatch: pytest.MonkeyPatch, default_priority: int | None, mock_response: str = "queued"
) -> Router:
    router: Final = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "openai/gpt-4", "api_key": "sk-queue-test", "mock_response": mock_response},
            }
        ],
        default_priority=default_priority,
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock(post_call_failure_hook=AsyncMock()))
    return router


@pytest.mark.parametrize(
    ("default_priority", "priority_fields", "expected_priority"),
    [
        (3, {"priority": "high", "drop_params": True}, 3),
        (3, {"priority": "1"}, 1),
        (None, {"priority": 1.0}, 1),
        (3, {"priority": 1}, 1),
        (3, {}, 3),
    ],
    ids=("dropped-word-takes-default", "digit-string", "integral-float", "explicit", "absent-takes-default"),
)
def test_queue_chat_completions_reports_the_priority_the_router_queued_at(
    client,
    auth_as,
    monkeypatch: pytest.MonkeyPatch,
    default_priority: int | None,
    priority_fields: dict[str, object],
    expected_priority: int,
) -> None:
    router: Final = _queue_router(monkeypatch, default_priority)
    payload: Final = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}], **priority_fields}
    with patch.object(router.scheduler, "add_request", wraps=router.scheduler.add_request) as add_request:
        with auth_as():
            response: Final = client.post("/queue/chat/completions", json=payload)
    assert response.status_code == 200, response.text
    assert [call.kwargs["request"].priority for call in add_request.await_args_list] == [expected_priority]
    assert response.headers.get("x-litellm-priority") == str(expected_priority)


@pytest.mark.parametrize(
    ("default_priority", "priority_fields", "message"),
    [
        (None, {}, "priority is required"),
        (3, {"priority": "high"}, "priority must be an integer"),
        (None, {"priority": "high", "drop_params": True}, "priority must be an integer, got str 'high'"),
    ],
    ids=("no-priority-and-no-default", "malformed", "dropped-word-and-no-default"),
)
def test_queue_chat_completions_rejects_a_request_it_cannot_queue_as_an_invalid_request(
    client,
    auth_as,
    monkeypatch: pytest.MonkeyPatch,
    default_priority: int | None,
    priority_fields: dict[str, object],
    message: str,
) -> None:
    router: Final = _queue_router(monkeypatch, default_priority)
    payload: Final = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}], **priority_fields}
    with patch.object(router.scheduler, "add_request", wraps=router.scheduler.add_request) as add_request:
        with auth_as():
            response: Final = client.post("/queue/chat/completions", json=payload)
    assert response.status_code == 400, response.text
    error: Final = response.json()["error"]
    assert (error["type"], error["param"]) == ("invalid_request_error", "priority")
    assert message in error["message"]
    assert add_request.await_count == 0


def test_streamed_queue_responses_report_the_priority_the_router_queued_at(
    client, auth_as, monkeypatch: pytest.MonkeyPatch
) -> None:
    _queue_router(monkeypatch, default_priority=3)
    payload: Final = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}], "priority": "1", "stream": True}
    with auth_as():
        response: Final = client.post("/queue/chat/completions", json=payload)
    assert response.status_code == 200, response.text
    assert response.headers.get("x-litellm-priority") == "1"


def test_a_provider_bad_request_on_the_queue_route_keeps_its_own_message(
    client, auth_as, monkeypatch: pytest.MonkeyPatch
) -> None:
    _queue_router(monkeypatch, default_priority=3, mock_response="litellm.ContextWindowExceededError")
    payload: Final = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
    with auth_as():
        response: Final = client.post("/queue/chat/completions", json=payload)
    assert response.status_code == 400, response.text
    error: Final = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert "context window" in error["message"].lower()
    assert "Authentication Error" not in error["message"]
