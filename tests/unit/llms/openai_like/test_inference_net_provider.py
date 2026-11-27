import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage

_DEFAULT_API_BASE: Final = "https://api.inference.net/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_inference_net",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "moonshotai/kimi-k3",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Inference.net"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "total_tokens": 13,
        "prompt_tokens_details": {"cached_tokens": 4},
    },
}
INFERENCE_NET_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("inference_net/")))
INFERENCE_NET_CACHED_MODELS: Final = tuple(
    name for name in INFERENCE_NET_MODELS if "cache_read_input_token_cost" in litellm.model_cost[name]
)


def test_inference_net_is_a_registered_provider():
    assert litellm.LlmProviders.INFERENCE_NET.value == "inference_net"
    assert "inference_net" in litellm.provider_list
    assert "inference_net" in litellm.constants.openai_compatible_providers
    assert INFERENCE_NET_MODELS
    assert set(litellm.models_by_provider["inference_net"]) == set(INFERENCE_NET_MODELS)


@pytest.mark.parametrize(
    ("env", "expected_key"),
    [
        ({"INFERENCE_API_KEY": "primary-key"}, "primary-key"),
        ({"INFERENCE_NET_API_KEY": "fallback-key"}, "fallback-key"),
        ({"INFERENCE_API_KEY": "primary-key", "INFERENCE_NET_API_KEY": "fallback-key"}, "primary-key"),
    ],
    ids=["primary-only", "fallback-only", "primary-wins"],
)
def test_inference_net_reads_either_api_key_env(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], expected_key: str
):
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    monkeypatch.delenv("INFERENCE_NET_API_KEY", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    _, provider, api_key, _ = get_llm_provider(
        model="inference_net/deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "inference_net"
    assert api_key == expected_key


def test_inference_net_provider_resolution_keeps_slashes_in_model_ids(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCE_API_KEY", "inference-net-test-key")
    monkeypatch.delenv("INFERENCE_NET_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="inference_net/moonshotai/kimi-k3",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "moonshotai/kimi-k3"
    assert provider == "inference_net"
    assert api_key == "inference-net-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_inference_net_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCE_API_KEY", "inference-net-test-key")
    monkeypatch.setenv("INFERENCE_NET_API_BASE", "https://us.api.inference.net/v1")

    _, provider, _, api_base = get_llm_provider(
        model="inference_net/deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "inference_net"
    assert api_base == "https://us.api.inference.net/v1"


def test_inference_net_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCE_API_KEY", "inference-net-test-key")

    _, provider, api_key, _ = get_llm_provider(
        model="deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert provider == "inference_net"
    assert api_key == "inference-net-test-key"


@pytest.mark.parametrize("model", INFERENCE_NET_CACHED_MODELS)
def test_inference_net_bills_cache_hits_at_the_cache_read_rate(model: str):
    rates: Final = litellm.model_cost[model]
    cost: Final = litellm.completion_cost(
        completion_response=ModelResponse(
            model=model,
            usage=Usage(
                prompt_tokens=1_000_000,
                completion_tokens=1_000_000,
                total_tokens=2_000_000,
                prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=400_000),
            ),
        ),
        model=model,
        custom_llm_provider="inference_net",
    )

    assert cost == pytest.approx(
        600_000 * rates["input_cost_per_token"]
        + 400_000 * rates["cache_read_input_token_cost"]
        + 1_000_000 * rates["output_cost_per_token"]
    )


def test_inference_net_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("inference_net/")}

    assert tuple(sorted(entries)) == INFERENCE_NET_MODELS
    assert entries == {name: backup[name] for name in entries}


def test_inference_net_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="inference_net/moonshotai/kimi-k3",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            api_key="inference-net-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer inference-net-test-key"
    assert body["model"] == "moonshotai/kimi-k3"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Inference.net"
    assert response.usage.prompt_tokens_details.cached_tokens == 4
