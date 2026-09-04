"""
Tests for Kyma API provider configuration and integration.

Kyma API (https://kymaapi.com) is an OpenAI-compatible LLM gateway. Registered
here as a JSON-configured provider (litellm/llms/openai_like/providers.json).

No real network calls: config/registration assertions use JSONProviderRegistry
directly, and the routing/cost assertions use litellm's own `mock_response`
completion path (see litellm/main.py) rather than patching internals -- the
previous submission (PR #25543) patched `litellm.main.completion`, which is
never called because `litellm.completion` is bound at import time via
`from .main import *`, so the mock silently never fired.
"""

import json
import os
from pathlib import Path

import pytest

import litellm


@pytest.fixture(autouse=True, scope="module")
def _use_local_model_cost_map():
    """Force litellm's bundled backup instead of a live fetch from
    raw.githubusercontent.com/BerriAI/litellm/main. Without this, cost/model-info
    lookups race against whether `kyma` has reached the remote main branch's
    model_prices_and_context_window.json yet, rather than testing the file this
    PR ships. Same idiom used elsewhere in this suite, e.g.
    tests/litellm_utils_tests/test_utils.py::test_supports_reasoning and
    tests/_vcr_conftest_common.py.
    """
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"  # test-quality-ok: kyma not on remote map yet
    litellm.model_cost = litellm.get_model_cost_map(url="")  # test-quality-ok: reload after env set


class TestKymaProviderConfig:
    def test_kyma_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "KYMA")
        assert LlmProviders.KYMA.value == "kyma"
        assert "kyma" in litellm.provider_list

    def test_kyma_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("kyma")

        kyma = JSONProviderRegistry.get("kyma")
        assert kyma is not None
        assert kyma.base_url == "https://kymaapi.com/v1"
        assert kyma.api_key_env == "KYMA_API_KEY"

    def test_kyma_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "kyma" in openai_compatible_providers

    def test_kyma_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="kyma/deepseek-v3",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "deepseek-v3"
        assert provider == "kyma"
        assert api_base == "https://kymaapi.com/v1"

    def test_kyma_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="kyma/deepseek-v3",
            custom_llm_provider=None,
            api_base="https://custom.kymaapi.com/v1",
            api_key="sk-test",
        )

        assert provider == "kyma"
        assert api_base == "https://custom.kymaapi.com/v1"
        assert api_key == "sk-test"

    def test_kyma_url_autodetection(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="deepseek-v3",
            custom_llm_provider=None,
            api_base="https://kymaapi.com/v1",
            api_key=None,
        )
        assert provider == "kyma"
        assert api_base == "https://kymaapi.com/v1"

    def test_kyma_provider_config_manager(self):
        from litellm import LlmProviders
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_chat_config(model="deepseek-v3", provider=LlmProviders.KYMA)

        assert config is not None
        assert config.custom_llm_provider == "kyma"

    def test_kyma_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "kyma-chat",
                    "litellm_params": {
                        "model": "kyma/deepseek-v3",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "kyma-chat"


class TestKymaModelMetadata:
    # A small, deliberately mixed sample: a plain model, a vision model, a
    # model with prompt-caching enabled, and the one 2-tier step-pricing model.
    SAMPLE_MODELS = (
        "kyma/deepseek-v3",
        "kyma/claude-haiku-4-5",
        "kyma/qwen3.7-flash",
    )

    @staticmethod
    def _load(*path_parts):
        json_path = Path(__file__).parents[4].joinpath(*path_parts)
        with open(json_path) as f:
            return json.load(f)

    def test_kyma_models_registered_with_correct_metadata(self):
        model_cost = self._load("model_prices_and_context_window.json")
        for model in self.SAMPLE_MODELS:
            info = model_cost.get(model)
            assert info is not None, f"{model} missing from model_prices_and_context_window.json"
            assert info["litellm_provider"] == "kyma"
            assert info["mode"] == "chat"
            assert info["input_cost_per_token"] > 0
            assert info["output_cost_per_token"] > 0
            assert info["supports_function_calling"] is True
            assert info["supports_reasoning"] is True

    def test_kyma_prompt_caching_is_a_ninety_percent_discount(self):
        """Kyma's documented cache-read price is 10% of the input price
        (90% discount) -- see CLAUDE.md, Prompt Caching. Any model with
        supports_prompt_caching=True must carry the discounted rate, or
        LiteLLM's own cost calculator prices cache reads at $0."""
        model_cost = self._load("model_prices_and_context_window.json")
        checked = 0
        for model, info in model_cost.items():
            if not model.startswith("kyma/"):
                continue
            if info.get("supports_prompt_caching") is True:
                checked += 1
                assert "cache_read_input_token_cost" in info, f"{model} missing cache_read_input_token_cost"
                assert info["cache_read_input_token_cost"] == pytest.approx(info["input_cost_per_token"] * 0.1)
        assert checked > 0, "expected at least one kyma model with supports_prompt_caching=True"

    def test_kyma_two_tier_step_pricing(self):
        """qwen3.7-flash is Kyma's only 2-step model: the 256k-token rate
        must win over the 32k-token rate for a prompt past both thresholds."""
        info = litellm.get_model_info(model="kyma/qwen3.7-flash")

        assert info["input_cost_per_token_above_32k_tokens"] > info["input_cost_per_token"]
        assert info["input_cost_per_token_above_256k_tokens"] > info["input_cost_per_token_above_32k_tokens"]

        prompt_cost, _ = litellm.cost_per_token(
            model="kyma/qwen3.7-flash",
            prompt_tokens=300_000,
            completion_tokens=0,
        )
        # 300k tokens is past both the 32k and 256k thresholds -- the whole
        # prompt bills at the 256k rate, not the 32k rate (Kyma's tiers, like
        # ours, apply to the whole request once crossed -- not marginally).
        assert prompt_cost == pytest.approx(300_000 * info["input_cost_per_token_above_256k_tokens"])

    def test_kyma_models_synced_to_backup(self):
        model_cost = self._load("model_prices_and_context_window.json")
        backup = self._load("litellm", "model_prices_and_context_window_backup.json")
        kyma_models = [m for m in model_cost if m.startswith("kyma/")]
        assert len(kyma_models) == 70
        for model in kyma_models:
            assert model in backup, f"{model} missing from backup json"
            assert backup[model] == model_cost[model], f"{model} differs between root and backup json"


class TestKymaSupportedEndpoints:
    def test_supported_endpoints_matrix(self):
        matrix = self._load()
        endpoints = matrix["providers"]["kyma"]["endpoints"]
        assert endpoints["chat_completions"] is True
        # The JSON-configured-provider path only wires chat routing (see
        # litellm/llms/openai_like/json_loader.py); Kyma's own /v1/messages,
        # /v1/embeddings and /v1/rerank surfaces are not reachable through
        # this provider slug, so they are advertised false here rather than
        # overclaimed. Matches the libertai/scx-ai precedent.
        assert endpoints["embeddings"] is False

    def test_root_and_backup_endpoints_agree_for_kyma(self):
        root = self._load()
        backup_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
        backup = json.loads(backup_path.read_text())
        assert root["providers"]["kyma"] == backup["providers"]["kyma"]

    @staticmethod
    def _load():
        path = Path(__file__).parents[4] / "provider_endpoints_support.json"
        return json.loads(path.read_text())


class TestKymaRouting:
    @pytest.mark.asyncio
    async def test_router_spend_is_attributed_to_kyma_pricing(self):
        """Routed traffic is costed off the kyma entry, not an OpenAI one.
        Uses litellm's built-in `mock_response` -- no network call, and no
        risk of the mock being silently skipped (the bug that killed the
        previous Kyma submission)."""
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "kyma-chat",
                    "litellm_params": {"model": "kyma/deepseek-v3", "api_key": "sk-test"},
                }
            ]
        )

        response = await router.acompletion(
            model="kyma-chat",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="hello from kyma",
        )

        info = litellm.get_model_info(model="kyma/deepseek-v3")
        usage = response.usage
        expected = (
            usage.prompt_tokens * info["input_cost_per_token"] + usage.completion_tokens * info["output_cost_per_token"]
        )
        assert response._hidden_params["response_cost"] == pytest.approx(expected)
