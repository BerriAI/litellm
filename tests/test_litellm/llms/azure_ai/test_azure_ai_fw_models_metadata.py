"""
Regression tests for Azure AI Foundry Fireworks (FW-*) model cost map entries.

Prices for Data Zone pay-per-token meters come from the Azure retail prices API
(product "Azure Fireworks Models"). Kimi K3 rates come from the Microsoft Foundry
announcement. Models without dedicated Azure meters use published Fireworks
serverless rates.
"""

import json
from importlib.resources import files

import pytest



@pytest.fixture(scope="module")
def use_local_model_cost_map():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    import litellm
    from litellm.utils import _invalidate_model_cost_lowercase_map

    original_model_cost = litellm.model_cost
    litellm.model_cost = json.loads(
        files("litellm")
        .joinpath("model_prices_and_context_window_backup.json")
        .read_text(encoding="utf-8")
    )
    litellm.get_model_info.cache_clear()
    _invalidate_model_cost_lowercase_map()
    try:
        yield litellm
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()
        _invalidate_model_cost_lowercase_map()
        monkeypatch.undo()




@pytest.mark.parametrize(
    "model_name,expected_prompt,expected_completion",
    [
        ("FW-Kimi-K2.6", 1.045, 4.4),
        ("FW-DeepSeek-V4-Pro", 1.925, 3.828),
        ("FW-GLM-5.2", 1.54, 4.84),
        ("FW-Kimi-K3", 3.3, 16.5),
        ("FW-MiniMax-M2.5", 0.33, 1.32),
        ("FW-Inkling", 1.0, 4.05),
        ("FW-Nemotron-3-Ultra-NVFP4", 0.6, 2.4),
        ("FW-Nemotron-Lightning-3.5-30B-A3B", 0.06, 0.22),
    ],
)
def test_azure_ai_fw_cost_per_token(
    use_local_model_cost_map, model_name, expected_prompt, expected_completion
):
    from litellm.llms.azure_ai.cost_calculator import cost_per_token
    from litellm.types.utils import Usage

    usage = Usage(
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        total_tokens=2_000_000,
    )

    prompt_cost, completion_cost = cost_per_token(model=model_name, usage=usage)

    assert prompt_cost == pytest.approx(expected_prompt)
    assert completion_cost == pytest.approx(expected_completion)




def test_azure_ai_fw_nemotron_lightning_supports_tool_choice(use_local_model_cost_map):
    from litellm.llms.azure_ai.chat.transformation import AzureAIStudioConfig

    supported_params = AzureAIStudioConfig().get_supported_openai_params("FW-Nemotron-Lightning-3.5-30B-A3B")

    assert "tool_choice" in supported_params


def test_azure_ai_fw_kimi_k26_case_insensitive_lookup(use_local_model_cost_map):
    upper = use_local_model_cost_map.get_model_info(model="azure_ai/FW-Kimi-K2.6")
    lower = use_local_model_cost_map.get_model_info(model="azure_ai/fw-kimi-k2.6")

    assert upper["input_cost_per_token"] == pytest.approx(lower["input_cost_per_token"])
    assert upper["output_cost_per_token"] == pytest.approx(lower["output_cost_per_token"])
