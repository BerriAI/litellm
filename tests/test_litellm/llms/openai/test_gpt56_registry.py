import json
import os

import litellm
from litellm.utils import get_supported_openai_params

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
PRIMARY = os.path.join(ROOT, "model_prices_and_context_window.json")
BACKUP = os.path.join(ROOT, "litellm", "model_prices_and_context_window_backup.json")

MODELS = ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"]

EXPECTED = {
    "gpt-5.6-sol":   {"input_cost_per_token": 5e-06,  "output_cost_per_token": 3e-05,   "cache_read_input_token_cost": 5e-07},
    "gpt-5.6-terra": {"input_cost_per_token": 2e-06,  "output_cost_per_token": 1.2e-05, "cache_read_input_token_cost": 2e-07},
    "gpt-5.6-luna":  {"input_cost_per_token": 2e-07,  "output_cost_per_token": 1.2e-06, "cache_read_input_token_cost": 2e-08},
}


class TestGPT56Registry:
    def setup_method(self):
        with open(PRIMARY) as f:
            self.primary = json.load(f)
        with open(BACKUP) as f:
            self.backup = json.load(f)

    def test_models_in_primary(self):
        for model in MODELS:
            assert model in self.primary, f"{model} missing from primary registry"

    def test_models_in_backup(self):
        for model in MODELS:
            assert model in self.backup, f"{model} missing from backup registry"

    def test_registries_match(self):
        for model in MODELS:
            assert self.primary[model] == self.backup[model], \
                f"{model} differs between primary and backup registries"

    def test_pricing(self):
        for model, prices in EXPECTED.items():
            entry = self.primary[model]
            for field, value in prices.items():
                assert abs(entry[field] - value) < 1e-10, \
                    f"{model} {field}: expected {value}, got {entry[field]}"

    def test_context_window(self):
        for model in MODELS:
            entry = self.primary[model]
            assert entry["max_input_tokens"] == 922000
            assert entry["max_output_tokens"] == 128000

    def test_sol_tool_choice(self):
        params = get_supported_openai_params(
            model="gpt-5.6-sol",
            custom_llm_provider="openai",
        )
        assert "tool_choice" in params
