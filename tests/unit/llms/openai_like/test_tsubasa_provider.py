import json
from typing import Final

import httpx
import pytest

from litellm import LlmProviders, completion
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.utils import ProviderConfigManager, get_model_info


@pytest.mark.parametrize("model_id", ("tsubasa-fast", "tsubasa-pro"))
def test_tsubasa_routes_named_models_and_resolves_credentials(model_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TSUBASA_API_KEY", "test-credential")
    registration: Final = JSONProviderRegistry.get("tsubasa")
    assert registration is not None
    model, provider, api_key, api_base = get_llm_provider(model=f"tsubasa/{model_id}")
    assert (model, provider, api_key, api_base) == (
        model_id,
        registration.slug,
        "test-credential",
        registration.base_url,
    )
    config: Final = ProviderConfigManager.get_provider_chat_config(model=model_id, provider=LlmProviders.TSUBASA)
    assert config is not None
    assert (
        config.get_complete_url(api_base, api_key, model_id, {}, {}, stream=True)
        == f"{registration.base_url}/chat/completions"
    )
    assert config._get_openai_compatible_provider_info(None, "explicit-credential") == (
        registration.base_url,
        "explicit-credential",
    )
    assert config.map_openai_params({"max_completion_tokens": 64}, {}, model_id, False) == {"max_tokens": 64}
    info: Final = get_model_info(model=f"tsubasa/{model_id}")
    assert info["max_output_tokens"] <= info["max_input_tokens"]
    assert info["litellm_provider"] == registration.slug


@pytest.mark.parametrize("model_id", ("tsubasa-fast", "tsubasa-pro"))
@pytest.mark.parametrize("stream", (False, True))
def test_tsubasa_chat_transport(model_id: str, stream: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TSUBASA_API_KEY", "test-credential")
    registration = JSONProviderRegistry.get("tsubasa")
    assert registration is not None

    def send(_client: httpx.Client, request: httpx.Request, **_kwargs: object) -> httpx.Response:
        assert str(request.url) == f"{registration.base_url}/chat/completions"
        assert request.headers["authorization"] == "Bearer test-credential"
        body = json.loads(request.content)
        assert body["model"] == model_id
        assert body["messages"] == [{"role": "user", "content": "Hello"}]
        assert body["max_tokens"] == 32
        assert set(body) <= {"model", "messages", "max_tokens", "stream", "stream_options"}
        result = {
            "id": "chatcmpl-contract",
            "created": 1,
            "model": model_id,
            "object": "chat.completion.chunk" if stream else "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "delta" if stream else "message": {"role": "assistant", "content": "Hello"},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        if stream:
            return httpx.Response(
                200,
                request=request,
                headers={"content-type": "text/event-stream"},
                content=f"data: {json.dumps(result)}\n\ndata: [DONE]\n\n",
            )
        return httpx.Response(200, request=request, json=result)

    monkeypatch.setattr(httpx.Client, "send", send)
    result = completion(
        model=f"tsubasa/{model_id}", messages=[{"role": "user", "content": "Hello"}], max_tokens=32, stream=stream
    )
    if stream:
        assert "".join(chunk.choices[0].delta.content or "" for chunk in result if chunk.choices) == "Hello"
    else:
        assert result.choices[0].message.content == "Hello"
        assert result.usage.total_tokens == 2
