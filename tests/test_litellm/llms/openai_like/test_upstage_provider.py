"""
Tests for the Upstage (Solar) provider configuration and integration.
"""

import json
from pathlib import Path

import httpx
import pytest

import litellm
from litellm import completion

CHAT_COMPLETION_RESPONSE = {
    "id": "chatcmpl-upstage-test",
    "object": "chat.completion",
    "created": 1787686209,
    "model": "solar-pro4-260806",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Solar"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 11, "completion_tokens": 24, "total_tokens": 35},
}

STREAMING_RESPONSE = (
    'data: {"id":"chatcmpl-upstage-stream","object":"chat.completion.chunk","created":1787686266,'
    '"model":"solar-pro4-260806","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},'
    '"finish_reason":null}]}\n\n'
    'data: {"id":"chatcmpl-upstage-stream","object":"chat.completion.chunk","created":1787686266,'
    '"model":"solar-pro4-260806","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


class TestUpstageProviderIdentity:
    def test_upstage_is_a_registered_provider(self):
        from litellm import LlmProviders

        assert LlmProviders.UPSTAGE.value == "upstage"
        assert "upstage" in litellm.provider_list

    def test_upstage_json_config(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        upstage = JSONProviderRegistry.get("upstage")
        assert upstage is not None
        assert upstage.base_url == "https://api.upstage.ai/v1"
        assert upstage.api_key_env == "UPSTAGE_API_KEY"
        assert upstage.api_base_env == "UPSTAGE_API_BASE"

    def test_upstage_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "upstage" in openai_compatible_providers

    def test_prefixed_model_resolves_to_upstage_not_openai(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, _, api_base = get_llm_provider(
            model="upstage/solar-pro4",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "solar-pro4"
        assert provider == "upstage"
        assert api_base == "https://api.upstage.ai/v1"

    def test_explicit_api_base_and_key_win(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        _, provider, api_key, api_base = get_llm_provider(
            model="upstage/solar-pro4",
            custom_llm_provider=None,
            api_base="https://upstage.internal.example/v1",
            api_key="sk-test",
        )

        assert provider == "upstage"
        assert api_base == "https://upstage.internal.example/v1"
        assert api_key == "sk-test"

    def test_api_base_autodetects_upstage(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("UPSTAGE_API_KEY", "up-env-key")

        _, provider, api_key, api_base = get_llm_provider(
            model="solar-pro4",
            custom_llm_provider=None,
            api_base="https://api.upstage.ai/v1",
            api_key=None,
        )

        assert provider == "upstage"
        assert api_base == "https://api.upstage.ai/v1"
        assert api_key == "up-env-key"

    def test_autodetected_api_base_keeps_the_caller_api_key(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("UPSTAGE_API_KEY", "up-env-key")

        _, provider, api_key, _ = get_llm_provider(
            model="solar-pro4",
            custom_llm_provider=None,
            api_base="https://api.upstage.ai/v1",
            api_key="up-caller-key",
        )

        assert provider == "upstage"
        assert api_key == "up-caller-key"

    def test_env_api_key_is_read_from_upstage_variable(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("UPSTAGE_API_KEY", "up-env-key")

        provider = JSONProviderRegistry.get("upstage")
        assert provider is not None

        api_base, api_key = create_config_class(provider)()._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://api.upstage.ai/v1"
        assert api_key == "up-env-key"


class TestUpstageRequestShape:
    """A wrong host, a wrong env var, or a leaked `upstage/` prefix fails here and nowhere else."""

    @pytest.mark.respx()
    def test_chat_completion_targets_the_upstage_endpoint(self, respx_mock, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("UPSTAGE_API_KEY", "up-env-key")

        respx_mock.post("https://api.upstage.ai/v1/chat/completions").respond(
            json=CHAT_COMPLETION_RESPONSE, status_code=200
        )

        response = completion(
            model="upstage/solar-pro4",
            messages=[{"role": "user", "content": "Hello!"}],
        )

        assert len(respx_mock.calls) == 1
        request = respx_mock.calls[0].request

        assert str(request.url) == "https://api.upstage.ai/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer up-env-key"

        body = json.loads(request.content)
        assert body["model"] == "solar-pro4"
        assert body["messages"] == [{"role": "user", "content": "Hello!"}]

        assert response.choices[0].message.content == "Hello from Solar"

    @pytest.mark.respx()
    def test_streaming_reaches_upstage_and_reassembles(self, respx_mock, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("UPSTAGE_API_KEY", "up-env-key")

        respx_mock.post("https://api.upstage.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=STREAMING_RESPONSE.encode(),
            )
        )

        chunks = list(
            completion(
                model="upstage/solar-pro4",
                messages=[{"role": "user", "content": "Hello!"}],
                stream=True,
            )
        )

        request = respx_mock.calls[0].request
        assert str(request.url) == "https://api.upstage.ai/v1/chat/completions"
        assert json.loads(request.content)["stream"] is True

        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Hello"


class TestUpstageParamHandling:
    def test_max_completion_tokens_is_not_rewritten(self):
        """Upstage takes `max_completion_tokens` natively, so the usual rewrite to `max_tokens` would break it."""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("upstage")
        assert provider is not None

        optional_params = create_config_class(provider)().map_openai_params(
            non_default_params={"max_completion_tokens": 256},
            optional_params={},
            model="solar-pro4",
            drop_params=False,
        )

        assert optional_params["max_completion_tokens"] == 256
        assert "max_tokens" not in optional_params


class TestUpstageEndpointSupportMatrix:
    @pytest.mark.parametrize(
        "matrix_path",
        [
            Path(__file__).parents[4] / "provider_endpoints_support.json",
            Path(litellm.__file__).parent / "provider_endpoints_support_backup.json",
        ],
    )
    def test_matrix_declares_chat_completions_only(self, matrix_path: Path):
        endpoints = json.loads(matrix_path.read_text())["providers"]["upstage"]["endpoints"]

        assert endpoints["chat_completions"] is True
        assert endpoints["responses"] is False
        assert endpoints["messages"] is False
        assert endpoints["embeddings"] is False
