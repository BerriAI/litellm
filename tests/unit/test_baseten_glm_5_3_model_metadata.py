import json
from pathlib import Path

import pytest

import litellm

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"

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


def test_entry_is_in_cost_map():
    assert MODEL in _load(MAIN_PATH), f"{MODEL} missing from model_prices_and_context_window.json"


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
