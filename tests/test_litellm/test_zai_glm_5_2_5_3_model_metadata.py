import json
from pathlib import Path

import pytest

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.types.utils import PromptTokensDetailsWrapper, Usage
from litellm.utils import supports_prompt_caching, supports_reasoning

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

GLM_MODELS = ("zai/glm-5.2", "zai/glm-5.3")

INPUT_COST = 1.4e-06
CACHED_INPUT_COST = 2.6e-07
OUTPUT_COST = 4.4e-06


def _load(path):
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def local_model_cost_map(monkeypatch):
    """Force get_model_info to resolve against the in-repo cost map instead of the
    remote one fetched at import time, which still carries the pre-merge pricing."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


@pytest.mark.parametrize("model", GLM_MODELS)
def test_zai_glm_5_2_and_5_3_specs(model):
    info = _load(MAIN_PATH).get(model)
    assert info is not None, f"{model} missing from model_prices_and_context_window.json"

    assert info["litellm_provider"] == "zai"
    assert info["mode"] == "chat"

    assert info["input_cost_per_token"] == INPUT_COST
    assert info["output_cost_per_token"] == OUTPUT_COST
    assert info["cache_read_input_token_cost"] == CACHED_INPUT_COST
    assert info["cache_creation_input_token_cost"] == 0

    assert info["max_input_tokens"] == 1048576
    assert info["max_output_tokens"] == 128000

    assert info["supports_function_calling"] is True
    assert info["supports_prompt_caching"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_tool_choice"] is True

    routed_model, provider, _, _ = get_llm_provider(model=model)
    assert routed_model == model.split("/", 1)[1]
    assert provider == "zai"


@pytest.mark.parametrize("model", GLM_MODELS)
def test_zai_glm_capabilities_are_visible_to_callers(local_model_cost_map, model):
    """Z.ai advertises reasoning and context caching on both models, so the helpers
    every caller checks before sending a request must say so too."""
    assert supports_reasoning(model=model) is True
    assert supports_prompt_caching(model=model) is True

    info = litellm.get_model_info(model=model)
    assert info["max_input_tokens"] == 1048576
    assert info["max_output_tokens"] == 128000


@pytest.mark.parametrize("model", GLM_MODELS)
def test_cached_prompt_tokens_bill_at_the_cached_rate(local_model_cost_map, model):
    """A cache hit reports its reused tokens under prompt_tokens_details, and those
    tokens cost $0.26 per million rather than the full $1.40 input rate."""
    usage = Usage(
        prompt_tokens=21010,
        completion_tokens=100,
        total_tokens=21110,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=20992),
    )

    prompt_cost, completion_cost = litellm.cost_per_token(
        model=model, usage_object=usage, custom_llm_provider="zai"
    )

    assert prompt_cost == pytest.approx(18 * INPUT_COST + 20992 * CACHED_INPUT_COST)
    assert completion_cost == pytest.approx(100 * OUTPUT_COST)


@pytest.mark.parametrize("model", GLM_MODELS)
def test_backup_matches_main(model):
    """Ensure the bundled (backup) cost map stays in sync with the canonical file."""
    main_cost = _load(MAIN_PATH)
    backup_cost = _load(BACKUP_PATH)

    assert backup_cost.get(model) == main_cost.get(model), f"{model} differs between main and backup model cost maps"
