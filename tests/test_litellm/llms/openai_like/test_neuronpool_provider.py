"""NeuronPool OpenAI-compatible JSON provider."""

from pathlib import Path

import litellm


class TestNeuronpoolProviderConfig:
    def test_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("neuronpool")
        cfg = JSONProviderRegistry.get("neuronpool")
        assert cfg is not None
        assert cfg.base_url == "https://neuronpool.damnknee.workers.dev/v1"
        assert cfg.api_key_env == "NEURONPOOL_API_KEY"
        assert cfg.api_base_env == "NEURONPOOL_API_BASE"
        assert cfg.param_mappings.get("max_completion_tokens") == "max_tokens"
        assert "/v1/chat/completions" in cfg.supported_endpoints
        assert "/v1/embeddings" in cfg.supported_endpoints

    def test_prefixed_model_resolves(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, _, api_base = get_llm_provider(
            model="neuronpool/gpt-oss-20b",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )
        assert model == "gpt-oss-20b"
        assert provider == "neuronpool"
        assert api_base == "https://neuronpool.damnknee.workers.dev/v1"

    def test_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        _, provider, api_key, api_base = get_llm_provider(
            model="neuronpool/neuronpool-tiny-chat",
            custom_llm_provider=None,
            api_base="http://127.0.0.1:8787/v1",
            api_key="sk-test",
        )
        assert provider == "neuronpool"
        assert api_base == "http://127.0.0.1:8787/v1"
        assert api_key == "sk-test"


class TestNeuronpoolSidecarPrices:
    CHAT_MODELS = (
        "neuronpool/llama-3.2-1b-instruct",
        "neuronpool/llama-3.1-8b-instruct",
        "neuronpool/qwen2.5-7b-instruct",
        "neuronpool/gemma-3-12b-it",
        "neuronpool/qwen3-30b-a3b",
        "neuronpool/gpt-oss-20b",
        "neuronpool/neuronpool-tiny-chat",
    )

    @staticmethod
    def _load_sidecar():
        path = Path(__file__).parents[4] / "neuronpool_model_prices.json"
        import json

        with open(path) as f:
            return json.load(f)

    def test_sidecar_chat_entries(self):
        prices = self._load_sidecar()
        for model in self.CHAT_MODELS:
            info = prices.get(model)
            assert info is not None, f"{model} missing from neuronpool_model_prices.json"
            assert info["litellm_provider"] == "neuronpool"
            assert info["mode"] == "chat"
            assert info["input_cost_per_token"] > 0
            assert info["output_cost_per_token"] > 0
            assert info["max_output_tokens"] == 4096

    def test_sidecar_embedding_entry(self):
        prices = self._load_sidecar()
        info = prices["neuronpool/nomic-embed-text"]
        assert info["litellm_provider"] == "neuronpool"
        assert info["mode"] == "embedding"
        assert info["input_cost_per_token"] > 0
        assert info["output_cost_per_token"] == 0
