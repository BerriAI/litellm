"""Behavior pins for ``proxy_server.py`` queue routes.

Pins (PR2):
    - POST /queue/chat/completions
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from litellm.litellm_core_utils.core_helpers import get_spend_log_metadata_from_kwargs
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


def test_queue_chat_completions_no_router_error(client, auth_as, queue_no_router):
    """Pins ``POST /queue/chat/completions`` (error: no llm_router)."""
    payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
    with auth_as():
        response = client.post("/queue/chat/completions", json=payload)
    assert response.status_code == 500
    assert len(response.content) > 0


def test_a_guarded_queued_spend_row_keeps_the_spend_logs_metadata_the_caller_sent(
    client: TestClient, auth_as: Callable[..., AbstractContextManager[None]], patched_queue: MagicMock
) -> None:
    with auth_as():
        response: Final = client.post(
            "/queue/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hi"}],
                "priority": 0,
                "metadata": {"spend_logs_metadata": {"q_k": "q_v"}},
            },
        )
    assert response.status_code == 200, response.text
    guardrail_planted: Final = {
        "litellm_params": {
            "metadata": patched_queue.schedule_acompletion.call_args.kwargs["metadata"],
            "litellm_metadata": {"user_api_key_hash": "hash"},
        }
    }

    assert get_spend_log_metadata_from_kwargs(guardrail_planted)["spend_logs_metadata"] == {"q_k": "q_v"}
