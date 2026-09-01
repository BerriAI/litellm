import json
from pathlib import Path

import pytest

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.fireworks_ai.cost_calculator import cost_per_token
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

FLASH_KEYS = (
    "fireworks_ai/glm-5p3-flash",
    "fireworks_ai/accounts/fireworks/models/glm-5p3-flash",
)
BASE_KEYS = (
    "fireworks_ai/glm-5p3",
    "fireworks_ai/accounts/fireworks/models/glm-5p3",
)


def _model_cost():
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(json_path) as f:
        return json.load(f)


@pytest.mark.parametrize("model", FLASH_KEYS)
def test_fireworks_glm_5p3_flash_model_info(model):
    info = _model_cost().get(model)
    assert info is not None, f"{model} not found in model_prices_and_context_window.json"

    assert info["litellm_provider"] == "fireworks_ai"
    assert info["mode"] == "chat"

    # https://docs.fireworks.ai/serverless/pricing -- "$0.15 / $0.03 / $0.50" per 1M
    assert info["input_cost_per_token"] == 1.5e-07
    assert info["cache_read_input_token_cost"] == 3e-08
    assert info["output_cost_per_token"] == 5e-07

    assert info["max_input_tokens"] == 1048576
    assert info["max_output_tokens"] == 131072
    assert info["max_tokens"] == 131072

    assert info["supports_function_calling"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_tool_choice"] is True
    # GLM 5.3 Flash is multimodal, unlike GLM 5.3 -- matches zai/glm-5.3-flash
    # and friendliai/zai-org/GLM-5.3-Flash.
    assert info["supports_vision"] is True


@pytest.mark.parametrize("model", BASE_KEYS)
def test_fireworks_glm_5p3_model_info(model):
    """The short alias must carry the same rates as the accounts/... form."""
    info = _model_cost().get(model)
    assert info is not None, f"{model} not found in model_prices_and_context_window.json"

    assert info["litellm_provider"] == "fireworks_ai"
    assert info["mode"] == "chat"
    assert info["input_cost_per_token"] == 1.4e-06
    assert info["cache_read_input_token_cost"] == 2.6e-07
    assert info["output_cost_per_token"] == 4.4e-06
    assert info["max_input_tokens"] == 1048576
    assert info["supports_vision"] is False


def test_fireworks_glm_5p3_aliases_match_canonical_entries():
    model_cost = _model_cost()
    assert (
        model_cost["fireworks_ai/glm-5p3"]
        == model_cost["fireworks_ai/accounts/fireworks/models/glm-5p3"]
    )
    assert (
        model_cost["fireworks_ai/glm-5p3-flash"]
        == model_cost["fireworks_ai/accounts/fireworks/models/glm-5p3-flash"]
    )


@pytest.mark.parametrize("model", FLASH_KEYS + BASE_KEYS)
def test_fireworks_glm_5p3_backup_matches_main(model):
    """Ensure the bundled model cost map stays in sync with the canonical file."""
    repo_root = Path(__file__).parents[2]
    with open(repo_root / "model_prices_and_context_window.json") as f:
        main_cost = json.load(f)
    with open(repo_root / "litellm" / "model_prices_and_context_window_backup.json") as f:
        backup_cost = json.load(f)

    assert backup_cost.get(model) == main_cost.get(
        model
    ), f"{model} differs between main and backup model cost maps"


@pytest.mark.parametrize("model", FLASH_KEYS)
def test_fireworks_glm_5p3_flash_provider_routing(model):
    routed_model, provider, _, _ = get_llm_provider(model=model)
    assert routed_model == model.split("/", 1)[1]
    assert provider == "fireworks_ai"


def test_fireworks_glm_5p3_flash_prices_cached_tokens_at_cache_read_rate(
    local_model_cost_map,
):
    """Without a cost-map entry this model billed at $0; with one, cached input
    tokens must be charged at cache_read_input_token_cost, not the full input rate.
    """
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=100,
        total_tokens=1100,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=800),
    )
    prompt_cost, completion_cost = cost_per_token(
        model="accounts/fireworks/models/glm-5p3-flash", usage=usage
    )

    # 200 uncached * 1.5e-07 + 800 cached * 3e-08
    assert prompt_cost == pytest.approx(200 * 1.5e-07 + 800 * 3e-08)
    assert completion_cost == pytest.approx(100 * 5e-07)

    # and the cached tokens really are discounted vs the full input rate
    assert prompt_cost < 1000 * 1.5e-07
