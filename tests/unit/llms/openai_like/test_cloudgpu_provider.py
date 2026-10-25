import json
from typing import Final

import pytest
import respx

import litellm

CLOUDGPU_MODEL: Final = "cloudgpu/deepseek-v4-flash"
CLOUDGPU_MODEL_ID: Final = "deepseek-v4-flash"
CLOUDGPU_BASE: Final = "https://cloudgpu.app/v1"


def test_cloudgpu_json_config_exists():
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    assert JSONProviderRegistry.exists("cloudgpu")
    cfg = JSONProviderRegistry.get("cloudgpu")
    assert cfg.base_url == CLOUDGPU_BASE
    assert cfg.api_key_env == "CGW_API_KEY"
    assert cfg.api_base_env == "CGW_API_BASE"


def test_cloudgpu_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("CGW_API_KEY", "cloudgpu-test-key")
    monkeypatch.delenv("CGW_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model=CLOUDGPU_MODEL, custom_llm_provider=None, api_base=None, api_key=None
    )

    assert model == CLOUDGPU_MODEL_ID
    assert provider == "cloudgpu"
    assert api_key == "cloudgpu-test-key"
    assert api_base == CLOUDGPU_BASE


def test_cloudgpu_explicit_credentials_win(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("CGW_API_KEY", "cloudgpu-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model=CLOUDGPU_MODEL,
        custom_llm_provider=None,
        api_base="https://cloudgpu.internal.example/v1",
        api_key="cloudgpu-explicit-key",
    )

    assert provider == "cloudgpu"
    assert api_key == "cloudgpu-explicit-key"
    assert api_base == "https://cloudgpu.internal.example/v1"


def test_cloudgpu_chat_completion_request():
    with respx.mock() as upstream:
        route = upstream.post(f"{CLOUDGPU_BASE}/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_cloudgpu",
                "object": "chat.completion",
                "created": 1_793_000_000,
                "model": CLOUDGPU_MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from CloudGPU"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8},
            },
        )
        response = litellm.completion(
            model=CLOUDGPU_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="cloudgpu-test-key",
        )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer cloudgpu-test-key"
    assert body["model"] == CLOUDGPU_MODEL_ID
    assert response.choices[0].message.content == "Hello from CloudGPU"
