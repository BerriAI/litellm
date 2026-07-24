"""Behavior pins for ``proxy_server.py`` llm-utils routes.

Pins (PR2):
    - POST /utils/token_counter
    - GET /utils/supported_openai_params
    - POST /utils/transform_request
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.proxy import proxy_server

from .conftest import normalize  # type: ignore[import-not-found]

# ---------------------------------------------------------------------------
# POST /utils/token_counter
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_token_counter(monkeypatch):
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(litellm, "disable_token_counter", False, raising=False)
    monkeypatch.setattr(
        litellm.utils,
        "_select_tokenizer",
        lambda model, custom_tokenizer=None: {
            "type": "openai_tokenizer",
            "tokenizer": None,
        },
    )
    monkeypatch.setattr(litellm, "token_counter", lambda **kwargs: 7)
    yield


def test_token_counter_happy_path(client, auth_as, patched_token_counter):
    """Pins ``POST /utils/token_counter``."""
    payload = {"model": "gpt-4", "prompt": "Hi there"}
    with auth_as():
        response = client.post("/utils/token_counter", json=payload)
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "total_tokens": 7,
        "request_model": "gpt-4",
        "model_used": "gpt-4",
        "tokenizer_type": "openai_tokenizer",
        "original_response": None,
        "error": False,
        "error_message": None,
        "status_code": None,
    }


def test_token_counter_missing_input_returns_400(
    client, auth_as, patched_token_counter
):
    """Pins ``POST /utils/token_counter`` (error: missing input)."""
    with auth_as():
        response = client.post("/utils/token_counter", json={"model": "gpt-4"})
    assert response.status_code == 400
    assert "prompt or messages or contents" in response.text


# ---------------------------------------------------------------------------
# GET /utils/supported_openai_params
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_supported_params(monkeypatch):
    monkeypatch.setattr(
        litellm,
        "get_llm_provider",
        lambda model: (model, "openai", None, None),
    )
    monkeypatch.setattr(
        litellm,
        "get_supported_openai_params",
        lambda model, custom_llm_provider=None: ["max_tokens", "temperature", "top_p"],
    )
    yield


def test_supported_openai_params_happy_path(client, auth_as, patched_supported_params):
    """Pins ``GET /utils/supported_openai_params``."""
    with auth_as():
        response = client.get(
            "/utils/supported_openai_params", params={"model": "gpt-4"}
        )
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "supported_openai_params": ["max_tokens", "temperature", "top_p"],
    }


def test_supported_openai_params_invalid_model(client, auth_as, monkeypatch):
    """Pins ``GET /utils/supported_openai_params`` (error: unknown model)."""

    def _raise(model):
        raise Exception("unknown")

    monkeypatch.setattr(litellm, "get_llm_provider", _raise)
    with auth_as():
        response = client.get("/utils/supported_openai_params", params={"model": "??"})
    assert response.status_code == 400
    assert "Could not map model" in response.text


# ---------------------------------------------------------------------------
# POST /utils/transform_request
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_transform(monkeypatch):
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(proxy_server, "is_request_body_safe", lambda **kwargs: True)

    def _fake_return_raw_request(endpoint, kwargs):
        return {
            "raw_request_api_base": "https://api.openai.com/v1/chat/completions",
            "raw_request_body": kwargs,
            "raw_request_headers": {"Authorization": "Bearer redacted"},
        }

    monkeypatch.setattr("litellm.utils.return_raw_request", _fake_return_raw_request)
    yield


def test_transform_request_happy_path(client, auth_as, patched_transform):
    """Pins ``POST /utils/transform_request``."""
    payload = {"call_type": "completion", "request_body": {"model": "gpt-4"}}
    with auth_as():
        response = client.post("/utils/transform_request", json=payload)
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "raw_request_api_base": "https://api.openai.com/v1/chat/completions",
        "raw_request_body": {"model": "gpt-4"},
        "raw_request_headers": {"Authorization": "Bearer redacted"},
    }


def test_transform_request_unsafe_body(client, auth_as, monkeypatch):
    """Pins ``POST /utils/transform_request`` (error: unsafe body)."""
    monkeypatch.setattr(proxy_server, "llm_router", None)

    def _raise(**kwargs):
        raise ValueError("unsafe model")

    monkeypatch.setattr(proxy_server, "is_request_body_safe", _raise)
    payload = {"call_type": "completion", "request_body": {"model": "evil"}}
    with auth_as():
        response = client.post("/utils/transform_request", json=payload)
    assert response.status_code == 400
    assert "unsafe" in response.text or "error" in response.text


@pytest.mark.asyncio
async def test_transform_request_does_not_block_event_loop(monkeypatch):
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(proxy_server, "is_request_body_safe", lambda **kwargs: True)

    event_loop_progressed = threading.Event()

    def blocking_return_raw_request(endpoint, kwargs):
        if not event_loop_progressed.wait(timeout=1):
            raise AssertionError("event loop was blocked")
        return {
            "raw_request_api_base": "https://example.com/v1/chat/completions",
            "raw_request_body": kwargs,
            "raw_request_headers": {},
        }

    monkeypatch.setattr(
        "litellm.utils.return_raw_request", blocking_return_raw_request
    )

    async def heartbeat():
        event_loop_progressed.set()

    heartbeat_task = asyncio.create_task(heartbeat())
    await proxy_server.transform_request(
        proxy_server.TransformRequestBody(
            call_type="completion",
            request_body={"model": "openai/test-model"},
        )
    )
    await heartbeat_task
