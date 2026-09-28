"""
Tests for the Voxell Forge provider (JSON-configured, embeddings only).
"""

import json
from pathlib import Path
from typing import Final

import httpx
import pytest
import respx

import litellm

VOXELL_MODELS: Final = ("voxell/turbo", "voxell/pro", "voxell/ultra")


def _load(*path_parts: str) -> dict:
    json_path = Path(__file__).parents[4].joinpath(*path_parts)
    with open(json_path) as f:
        return json.load(f)


def _mock_voxell_embedding_route(respx_mock: respx.MockRouter, model: str) -> respx.Route:
    return respx_mock.post("https://api.voxell.ai/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "model": model,
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )
    )


class TestVoxellProviderConfig:
    def test_voxell_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "VOXELL")
        assert LlmProviders.VOXELL.value == "voxell"
        assert "voxell" in litellm.provider_list

    def test_voxell_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("voxell")

        voxell = JSONProviderRegistry.get("voxell")
        assert voxell is not None
        assert voxell.base_url == "https://api.voxell.ai/v1"
        assert voxell.api_key_env == "VOXELL_API_KEY"
        assert voxell.api_base_env == "VOXELL_API_BASE"
        assert voxell.supported_endpoints == ["/v1/embeddings"]

    def test_supports_embeddings_reads_supported_endpoints(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.supports_embeddings("voxell") is True
        assert JSONProviderRegistry.supports_embeddings("scx-ai") is False
        assert JSONProviderRegistry.supports_embeddings("not-a-provider") is False

    def test_voxell_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_endpoints, openai_compatible_providers

        assert "voxell" in openai_compatible_providers
        assert "https://api.voxell.ai/v1" in openai_compatible_endpoints

    def test_voxell_provider_resolution(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("VOXELL_API_KEY", "vx-env-key")
        monkeypatch.delenv("VOXELL_API_BASE", raising=False)

        model, provider, api_key, api_base = get_llm_provider(
            model="voxell/turbo",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "turbo"
        assert provider == "voxell"
        assert api_key == "vx-env-key"
        assert api_base == "https://api.voxell.ai/v1"

    def test_voxell_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="voxell/pro",
            custom_llm_provider=None,
            api_base="https://custom.voxell.example/v1",
            api_key="vx-test",
        )

        assert provider == "voxell"
        assert api_base == "https://custom.voxell.example/v1"
        assert api_key == "vx-test"

    def test_voxell_url_autodetection(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="turbo",
            custom_llm_provider=None,
            api_base="https://api.voxell.ai/v1",
            api_key=None,
        )
        assert provider == "voxell"
        assert api_base == "https://api.voxell.ai/v1"


class TestVoxellEmbeddingDispatch:
    def test_embedding_hits_voxell_endpoint(self, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LITELLM_DEFAULT_EMBEDDING_ENCODING_FORMAT", raising=False)
        monkeypatch.setenv("VOXELL_API_KEY", "vx-env-key")
        route: Final = _mock_voxell_embedding_route(respx_mock, "turbo")

        response: Final = litellm.embedding(model="voxell/turbo", input=["hello"])

        request: Final = route.calls.last.request
        body: Final = json.loads(request.read())
        assert request.headers["authorization"] == "Bearer vx-env-key"
        assert body["model"] == "turbo"
        assert body["input"] == ["hello"]
        assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
        assert response._hidden_params["custom_llm_provider"] == "voxell"

    def test_embedding_forwards_dimensions_and_encoding_format(self, respx_mock: respx.MockRouter):
        route: Final = _mock_voxell_embedding_route(respx_mock, "pro")

        litellm.embedding(
            model="voxell/pro",
            input=["hello"],
            api_key="vx-test",
            dimensions=512,
            encoding_format="float",
        )

        body: Final = json.loads(route.calls.last.request.read())
        assert body["dimensions"] == 512
        assert body["encoding_format"] == "float"

    def test_embedding_env_var_sets_default_encoding_format(
        self, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("LITELLM_DEFAULT_EMBEDDING_ENCODING_FORMAT", "float")
        route: Final = _mock_voxell_embedding_route(respx_mock, "turbo")

        litellm.embedding(model="voxell/turbo", input=["hello"], api_key="vx-test")

        assert json.loads(route.calls.last.request.read())["encoding_format"] == "float"

    @pytest.mark.asyncio
    async def test_aembedding_hits_voxell_endpoint(self, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        route: Final = _mock_voxell_embedding_route(respx_mock, "ultra")

        response: Final = await litellm.aembedding(model="voxell/ultra", input=["hello"], api_key="vx-test")

        assert route.called
        assert json.loads(route.calls.last.request.read())["model"] == "ultra"
        assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]


class TestVoxellModelMetadata:
    def test_voxell_models_registered_as_embedding_models(self):
        model_cost: Final = _load("model_prices_and_context_window.json")
        for model in VOXELL_MODELS:
            info = model_cost.get(model)
            assert info is not None, f"{model} missing from model_prices_and_context_window.json"
            assert info["litellm_provider"] == "voxell"
            assert info["mode"] == "embedding"
            assert info["output_vector_size"] > 0
            assert info["max_input_tokens"] == info["max_tokens"] > 0
            assert info["input_cost_per_token"] >= 0
            assert info["output_cost_per_token"] == 0.0

    def test_embedding_cost_derived_from_cost_map(self, respx_mock: respx.MockRouter):
        _mock_voxell_embedding_route(respx_mock, "pro")

        response: Final = litellm.embedding(model="voxell/pro", input=["hello"], api_key="vx-test")

        expected: Final = response.usage.prompt_tokens * litellm.model_cost["voxell/pro"]["input_cost_per_token"]
        assert response._hidden_params["response_cost"] == pytest.approx(expected)

    def test_voxell_models_synced_to_backup(self):
        model_cost: Final = _load("model_prices_and_context_window.json")
        backup: Final = _load("litellm", "model_prices_and_context_window_backup.json")
        for model in VOXELL_MODELS:
            assert model in backup, f"{model} missing from backup json"
            assert backup[model] == model_cost[model], f"{model} differs between root and backup json"

    def test_voxell_endpoint_support_matrix(self):
        for parts in (("provider_endpoints_support.json",), ("litellm", "provider_endpoints_support_backup.json")):
            endpoints = _load(*parts)["providers"]["voxell"]["endpoints"]
            assert endpoints["embeddings"] is True
            assert endpoints["chat_completions"] is False
            assert endpoints["responses"] is False
            assert endpoints["messages"] is False
