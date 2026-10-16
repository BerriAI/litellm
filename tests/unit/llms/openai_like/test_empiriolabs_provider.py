"""
Unit tests for the EmpirioLabs OpenAI-like provider.
"""

import json
import os
import sys
from typing import Final

import pytest
import respx

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

import litellm
from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

EMPIRIOLABS_BASE_URL = "https://api.empiriolabs.ai/v1"


def _get_config():
    provider = JSONProviderRegistry.get("empiriolabs")
    assert provider is not None
    config_class = create_config_class(provider)
    return config_class()


def test_empiriolabs_provider_registered():
    provider = JSONProviderRegistry.get("empiriolabs")
    assert provider is not None
    assert provider.base_url == EMPIRIOLABS_BASE_URL
    assert provider.api_key_env == "EMPIRIOLABS_API_KEY"
    assert provider.api_base_env == "EMPIRIOLABS_API_BASE"


def test_empiriolabs_resolves_env_api_key(monkeypatch):
    config = _get_config()
    monkeypatch.setenv("EMPIRIOLABS_API_KEY", "test-key")
    api_base, api_key = config.get_openai_compatible_provider_info(None, None)
    assert api_base == EMPIRIOLABS_BASE_URL
    assert api_key == "test-key"


def test_empiriolabs_maps_max_completion_tokens():
    config = _get_config()
    params = config.map_openai_params(
        non_default_params={"max_completion_tokens": 256},
        optional_params={},
        model="empiriolabs/qwen3-7-plus",
        drop_params=False,
    )
    assert params.get("max_tokens") == 256
    assert "max_completion_tokens" not in params


def test_empiriolabs_complete_url_appends_endpoint():
    config = _get_config()
    url = config.get_complete_url(
        api_base=EMPIRIOLABS_BASE_URL,
        api_key="test-key",
        model="empiriolabs/qwen3-7-plus",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == f"{EMPIRIOLABS_BASE_URL}/chat/completions"


def test_empiriolabs_chat_models_still_route_through_the_json_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMPIRIOLABS_API_KEY", "environment-key")
    monkeypatch.delenv("EMPIRIOLABS_API_BASE", raising=False)

    with respx.mock() as upstream:
        route: Final = upstream.post(f"{EMPIRIOLABS_BASE_URL}/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_empiriolabs",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": "qwen3-7-plus",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from EmpirioLabs AI"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
            },
        )
        response: Final = litellm.completion(
            model="empiriolabs/qwen3-7-plus",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=16,
        )

    request: Final = route.calls.last.request
    assert litellm.get_llm_provider("empiriolabs/qwen3-7-plus") == (
        "qwen3-7-plus",
        "empiriolabs",
        "environment-key",
        EMPIRIOLABS_BASE_URL,
    )
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer environment-key"
    assert json.loads(request.content) == {
        "model": "qwen3-7-plus",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 16,
    }
    assert response.choices[0].message.content == "Hello from EmpirioLabs AI"
