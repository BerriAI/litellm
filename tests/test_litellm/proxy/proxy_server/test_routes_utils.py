"""Behavior pins for ``proxy_server.py`` llm-utils routes.

Pins (PR2):
    - POST /utils/token_counter
    - GET /utils/supported_openai_params
    - POST /utils/transform_request
"""

from __future__ import annotations

import asyncio
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


def test_token_counter_counts_off_the_event_loop(client, auth_as, patched_token_counter, monkeypatch):
    """
    A large prompt must not stall the proxy: the count runs in a worker thread, where there
    is no running event loop, rather than on the loop serving other requests.
    """
    counted_off_loop = []

    def recording_counter(**kwargs):
        try:
            asyncio.get_running_loop()
            counted_off_loop.append(False)
        except RuntimeError:
            counted_off_loop.append(True)
        return 7

    monkeypatch.setattr(litellm, "token_counter", recording_counter)

    with auth_as():
        response = client.post("/utils/token_counter", json={"model": "gpt-4", "prompt": "Hi there"})

    assert response.status_code == 200
    assert response.json()["total_tokens"] == 7
    assert counted_off_loop == [True]


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


def test_token_counter_fallback_counts_tools_system_and_anthropic_blocks(client, auth_as, monkeypatch):
    """The ``litellm.token_counter`` fallback counts the request's tools and system prompt, and Anthropic ``image``/``document`` blocks, instead of 500ing."""
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(litellm, "disable_token_counter", False, raising=False)
    system = [{"type": "text", "text": "You are a terse assistant. Answer in one sentence."}]
    tools = [
        {
            "name": "get_weather",
            "description": "Look up the current weather for a city",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        }
    ]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this file?"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}},
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0xLjQK"}},
            ],
        }
    ]

    def count(payload: dict) -> int:
        with auth_as():
            response = client.post("/utils/token_counter", json={"model": "claude-fable-5", **payload})
        assert response.status_code == 200, response.text
        return response.json()["total_tokens"]

    bare = count({"messages": messages})
    full = count({"messages": messages, "tools": tools, "system": system})

    assert bare == litellm.token_counter(model="claude-fable-5", messages=messages)
    assert full == litellm.token_counter(
        model="claude-fable-5",
        messages=[{"role": "system", "content": system}, *messages],
        tools=tools,
    )
    assert full > bare


def test_token_counter_fallback_prompt_with_tools_does_not_500(client, auth_as, monkeypatch):
    """Regression: a ``prompt`` request carrying ``tools`` but no ``messages`` still counts, because the fallback attaches tools only when counting messages (``token_counter`` rejects tools on the text path)."""
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(litellm, "disable_token_counter", False, raising=False)
    prompt = "count the tokens in this sentence please"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up the current weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }
    ]

    with auth_as():
        response = client.post(
            "/utils/token_counter", json={"model": "claude-fable-5", "prompt": prompt, "tools": tools}
        )

    assert response.status_code == 200, response.text
    assert response.json()["total_tokens"] == litellm.token_counter(model="claude-fable-5", text=prompt)
