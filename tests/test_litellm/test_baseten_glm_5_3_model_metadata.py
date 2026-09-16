import json
from pathlib import Path

import pytest

import litellm
from litellm.utils import supports_function_calling, supports_prompt_caching

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

MODEL = "baseten/zai-org/GLM-5.3"

INPUT_COST = 1.4e-06
CACHED_INPUT_COST = 1.4e-07
OUTPUT_COST = 4.4e-06


def _load(path):
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def local_model_cost_map(monkeypatch):
    """Force get_model_info to resolve against the in-repo cost map instead of the
    remote one fetched at import time, which still carries the pre-merge registry."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


def test_baseten_glm_5_3_capabilities_are_visible_to_callers(local_model_cost_map):
    """The entry advertises prompt caching and tool calling, so the helpers every
    caller checks before sending a request must say so too."""
    assert supports_prompt_caching(model=MODEL) is True
    assert supports_function_calling(model=MODEL) is True

    info = litellm.get_model_info(model="zai-org/GLM-5.3", custom_llm_provider="baseten")
    assert info["max_input_tokens"] > 0
    assert info["max_output_tokens"] > 0


def test_backup_matches_main():
    """Ensure the bundled (backup) cost map stays in sync with the canonical file.

    Both keys are asserted present first: comparing two ``.get`` results alone passes
    just as happily when neither file has the entry at all, which is the exact state
    this test exists to catch.
    """
    main_cost = _load(MAIN_PATH)
    backup_cost = _load(BACKUP_PATH)

    assert MODEL in main_cost, f"{MODEL} missing from model_prices_and_context_window.json"
    assert MODEL in backup_cost, f"{MODEL} missing from model_prices_and_context_window_backup.json"
    assert backup_cost[MODEL] == main_cost[MODEL], f"{MODEL} differs between main and backup model cost maps"


def test_entry_advertises_only_what_the_baseten_path_accepts(local_model_cost_map):
    """The Baseten path rejects unsupported request parameters."""
    supported = litellm.get_supported_openai_params(model="zai-org/GLM-5.3", custom_llm_provider="baseten")
    assert supported is not None

    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.utils.get_optional_params(
            model="zai-org/GLM-5.3",
            custom_llm_provider="baseten",
            parallel_tool_calls=True,
            reasoning_effort="high",
            drop_params=False,
        )
