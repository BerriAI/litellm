"""
Unit tests for the Abliteration OpenAI-like provider.
"""

import json
import os
import sys
from typing import Final

import pytest
import respx

import litellm
from litellm.utils import trim_messages

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

ABLITERATION_BASE_URL = "https://api.abliteration.ai/v1"


def _get_config():
    provider = JSONProviderRegistry.get("abliteration")
    assert provider is not None
    config_class = create_config_class(provider)
    return config_class()


def test_abliteration_provider_registered():
    provider = JSONProviderRegistry.get("abliteration")
    assert provider is not None
    assert provider.base_url == ABLITERATION_BASE_URL
    assert provider.api_key_env == "ABLITERATION_API_KEY"


def test_abliteration_resolves_env_api_key(monkeypatch):
    config = _get_config()
    monkeypatch.setenv("ABLITERATION_API_KEY", "test-key")
    api_base, api_key = config.get_openai_compatible_provider_info(None, None)
    assert api_base == ABLITERATION_BASE_URL
    assert api_key == "test-key"


def test_abliteration_complete_url_appends_endpoint():
    config = _get_config()
    url = config.get_complete_url(
        api_base=ABLITERATION_BASE_URL,
        api_key="test-key",
        model="abliteration/abliterated-model",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == f"{ABLITERATION_BASE_URL}/chat/completions"


def test_abliteration_chat_forwards_reasoning_and_accounts_for_cache(local_model_cost_map: None) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.abliteration.ai/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "abliterated-model-large-v2",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "total_tokens": 105,
                    "prompt_tokens_details": {"cached_tokens": 80},
                },
            },
        )
        response: Final = litellm.completion(
            model="abliteration/abliterated-model-large-v2",
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "Say hello"}],
            reasoning_effort="low",
        )
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert body["model"] == "abliterated-model-large-v2"
    assert body["reasoning_effort"] == "low"
    assert response.choices[0].message.content == "hello"
    rates: Final = litellm.model_cost["abliteration/abliterated-model-large-v2"]
    expected: Final = (
        20 * rates["input_cost_per_token"]
        + 80 * rates["cache_read_input_token_cost"]
        + 5 * rates["output_cost_per_token"]
    )
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)


def test_large_v2_model_group_does_not_offer_disabled_reasoning(local_model_cost_map: None) -> None:
    """Large v2 cannot disable reasoning per https://docs.abliteration.ai/capabilities/thinking (2026-10-26)."""
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "large-v2",
                "litellm_params": {
                    "model": "abliteration/abliterated-model-large-v2",
                    "api_key": "fake-test-key",
                },
            }
        ]
    )
    info: Final = router.get_model_group_info(model_group="large-v2")
    assert info is not None
    assert info.supported_reasoning_efforts
    assert "none" not in info.supported_reasoning_efforts
    assert "low" in info.supported_reasoning_efforts


def test_abliteration_models_trim_to_their_input_budget(local_model_cost_map: None) -> None:
    model_names: Final = tuple(model for model in litellm.model_cost if model.startswith("abliteration/"))
    assert model_names
    messages: Final = [{"role": "user", "content": "hello " * 500}]
    budget: Final = 100
    for model in model_names:
        trimmed: Final = trim_messages(
            messages,
            model=model,
            trim_ratio=budget / litellm.model_cost[model]["max_input_tokens"],
        )
        assert litellm.token_counter(model=model, messages=messages) > budget
        assert litellm.token_counter(model=model, messages=trimmed) <= budget
