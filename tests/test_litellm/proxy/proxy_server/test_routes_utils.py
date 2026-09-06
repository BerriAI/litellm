"""Behavior pins for ``proxy_server.py`` llm-utils routes.

Pins (PR2):
    - POST /utils/token_counter
    - GET /utils/supported_openai_params
    - POST /utils/transform_request
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.proxy import proxy_server
from litellm.router_utils import pattern_match_deployments
from litellm.types.utils import CredentialItem, TokenCountResponse

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


def test_token_counter_named_credentials_do_not_mutate_router_deployment(client, auth_as, monkeypatch):
    """Provider token counting must not pin rotating credentials to the router."""
    named_credential = CredentialItem(
        credential_name="rotating-bedrock",
        credential_values={"aws_session_token": "temporary-session-token"},
        credential_info={"custom_llm_provider": "bedrock"},
    )
    router_deployment = {
        "litellm_params": {
            "model": "bedrock/anthropic.claude-3-sonnet",
            "litellm_credential_name": "rotating-bedrock",
        },
        "model_info": {},
    }

    mock_router = MagicMock()
    mock_router.async_get_available_deployment = AsyncMock(return_value=router_deployment)
    mock_counter = MagicMock()
    mock_counter.should_use_token_counting_api.return_value = True
    mock_counter.count_tokens = AsyncMock(
        return_value=TokenCountResponse(
            total_tokens=1,
            request_model="claude-bedrock",
            model_used="anthropic.claude-3-sonnet",
            tokenizer_type="bedrock_api",
        )
    )

    monkeypatch.setattr(litellm, "credential_list", [named_credential])
    monkeypatch.setattr(proxy_server, "llm_router", mock_router)
    monkeypatch.setattr(
        proxy_server,
        "_get_provider_token_counter",
        lambda deployment, model: (
            mock_counter,
            "anthropic.claude-3-sonnet",
            "bedrock",
        ),
    )

    with auth_as():
        response = client.post(
            "/utils/token_counter?call_endpoint=true",
            json={
                "model": "claude-bedrock",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    hydrated_deployment = mock_counter.count_tokens.await_args.kwargs["deployment"]
    assert hydrated_deployment["litellm_params"]["aws_session_token"] == "temporary-session-token"
    assert "aws_session_token" not in router_deployment["litellm_params"]


# ---------------------------------------------------------------------------
# GET /utils/supported_openai_params
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_supported_params(monkeypatch):
    monkeypatch.setattr(proxy_server, "llm_router", None)
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
        response = client.get("/utils/supported_openai_params", params={"model": "gpt-4"})
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "supported_openai_params": ["max_tokens", "temperature", "top_p"],
    }


def test_supported_openai_params_resolves_router_alias(client, auth_as, monkeypatch):
    """A router alias absent from the cost map resolves through the deployment's underlying model."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "claude-opus-4-6-cached",
                "litellm_params": {"model": "anthropic/claude-opus-4-6", "api_key": "sk-test"},
            }
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)

    with auth_as():
        response = client.get("/utils/supported_openai_params", params={"model": "claude-opus-4-6-cached"})

    assert response.status_code == 200
    expected = litellm.get_supported_openai_params(model="claude-opus-4-6", custom_llm_provider="anthropic")
    assert response.json() == {"supported_openai_params": expected}
    assert "max_tokens" in response.json()["supported_openai_params"]


def test_supported_openai_params_declared_prefix_alias_resolves_through_router(client, auth_as, monkeypatch):
    """Regression: an alias whose name starts with an authenticating provider's prefix skipped
    router resolution and answered with that provider's params instead of the deployment's."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "github_copilot/gpt-4o",
                "litellm_params": {"model": "anthropic/claude-opus-4-6", "api_key": "sk-test"},
            }
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)

    with auth_as():
        response = client.get("/utils/supported_openai_params", params={"model": "github_copilot/gpt-4o"})

    assert response.status_code == 200
    expected = litellm.get_supported_openai_params(model="claude-opus-4-6", custom_llm_provider="anthropic")
    assert response.json() == {"supported_openai_params": expected}


def test_supported_openai_params_never_runs_oauth_for_authenticating_providers(client, auth_as, monkeypatch, tmp_path):
    """Regression: github_copilot/chatgpt names answer from their declaration; resolving them
    through ``get_llm_provider`` would run the provider's OAuth device flow and block the event loop."""
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(tmp_path))
    (tmp_path / "access-token").write_text("fake-access-token")
    (tmp_path / "api-key.json").write_text(
        json.dumps(
            {
                "token": "fake-api-key",
                "expires_at": 4102444800,
                "endpoints": {"api": "https://api.githubcopilot.com"},
            }
        )
    )
    router = litellm.Router(
        model_list=[
            {
                "model_name": "copilot-alias",
                "litellm_params": {"model": "github_copilot/gpt-4o"},
            },
            {
                "model_name": "openai/*",
                "litellm_params": {"model": "openai/*"},
            },
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)

    resolution_attempts: list[str] = []

    def _oauth_tripwire(model, *args, **kwargs):
        resolution_attempts.append(model)
        raise AssertionError("get_llm_provider would run the OAuth device flow")

    monkeypatch.setattr(litellm, "get_llm_provider", _oauth_tripwire)
    monkeypatch.setattr(pattern_match_deployments, "get_llm_provider", _oauth_tripwire)
    expected = litellm.get_supported_openai_params(model="gpt-4o", custom_llm_provider="github_copilot")

    with auth_as():
        via_alias = client.get("/utils/supported_openai_params", params={"model": "copilot-alias"})
        via_direct_name = client.get("/utils/supported_openai_params", params={"model": "github_copilot/gpt-4o"})

    assert via_alias.status_code == 200
    assert via_alias.json() == {"supported_openai_params": expected}
    assert via_direct_name.status_code == 200
    assert via_direct_name.json() == {"supported_openai_params": expected}
    assert resolution_attempts == []


def test_supported_openai_params_invalid_model(client, auth_as, monkeypatch):
    """Pins ``GET /utils/supported_openai_params`` (error: unknown model)."""

    def _raise(model):
        raise Exception("unknown")

    monkeypatch.setattr(proxy_server, "llm_router", None)
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
