import json
import os
from pathlib import Path

try:
    import pytest
except ImportError:
    pytest = None

import litellm
from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

REPO_ROOT = Path(__file__).resolve().parents[4]

BERGET_MODELS = [
    "Qwen/Qwen3.8-27B-FP8",
    "google/gemma-4-31B-it",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "moonshotai/Kimi-K3",
    "zai-org/GLM-5.3-Flash",
]


class TestBergetProvider:
    def test_provider_registered(self):
        assert JSONProviderRegistry.exists("berget")

    def test_provider_config(self):
        provider = JSONProviderRegistry.get("berget")
        assert provider.base_url == "https://api.berget.ai/v1"
        assert provider.api_key_env == "BERGET_API_KEY"
        assert provider.api_base_env == "BERGET_API_BASE"

    def test_dynamic_config_resolves_api_base(self):
        provider = JSONProviderRegistry.get("berget")
        config = create_config_class(provider)()
        api_base, _ = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://api.berget.ai/v1"

    def test_dynamic_config_custom_overrides(self):
        provider = JSONProviderRegistry.get("berget")
        config = create_config_class(provider)()
        api_base, api_key = config._get_openai_compatible_provider_info(
            "https://api-stage.berget.ai/v1", "test-key"
        )
        assert api_base == "https://api-stage.berget.ai/v1"
        assert api_key == "test-key"

    def test_provider_inference_resolves_prefix(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, _, api_base = get_llm_provider("berget/zai-org/GLM-5.3-Flash")
        assert provider == "berget"
        assert model == "zai-org/GLM-5.3-Flash"
        assert api_base == "https://api.berget.ai/v1"

    def test_no_param_mappings_required_for_berget(self):
        provider = JSONProviderRegistry.get("berget")
        assert not getattr(provider, "param_mappings", {})

    def test_provider_support_documented(self):
        support = json.loads(
            (REPO_ROOT / "provider_endpoints_support.json").read_text()
        )
        assert support["providers"]["berget"]["endpoints"]["chat_completions"] is True
        assert support["providers"]["berget"]["endpoints"]["responses"] is False

    @pytest.mark.skipif(pytest is None, reason="pytest not installed")
    @pytest.mark.parametrize("model_id", BERGET_MODELS)
    def test_pricing_entries_exist(self, model_id):
        entries = json.loads(
            (REPO_ROOT / "model_prices_and_context_window.json").read_text()
        )
        key = f"berget/{model_id}"
        assert key in entries, key
        entry = entries[key]
        assert entry["litellm_provider"] == "berget"
        assert entry["mode"] == "chat"
        assert entry["input_cost_per_token"] > 0
        assert entry["output_cost_per_token"] > 0
        assert entry["max_input_tokens"] > 0

    @pytest.mark.skipif(pytest is None, reason="pytest not installed")
    def test_runtime_backup_matches_root_cost_map(self):
        root = json.loads(
            (REPO_ROOT / "model_prices_and_context_window.json").read_text()
        )
        backup = json.loads(
            (REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json").read_text()
        )
        assert {k: v for k, v in root.items() if k.startswith("berget")} == {
            k: v for k, v in backup.items() if k.startswith("berget")
        }

    @pytest.mark.skipif(pytest is None, reason="pytest not installed")
    def test_litellm_packaged_backup_used_by_runtime_env(self):
        packaged = Path(litellm.__file__).parent
        if (packaged / "model_prices_and_context_window_backup.json").is_file():
            backup = json.loads(
                (packaged / "model_prices_and_context_window_backup.json").read_text()
            )
            root = json.loads(
                (REPO_ROOT / "model_prices_and_context_window.json").read_text()
            )
            assert {k: v for k, v in root.items() if k.startswith("berget")} == {
                k: v for k, v in backup.items() if k.startswith("berget")
            }
