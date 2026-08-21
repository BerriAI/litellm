import json
from pathlib import Path

import pytest

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

MEDIUM_3_5_MODELS = (
    "mistral/mistral-medium-3-5",
    "mistral/mistral-medium-2604",
    "mistral/mistral-medium-latest",
)

SYNCED_MODELS = MEDIUM_3_5_MODELS + (
    "mistral/mistral-medium-2508",
    "mistral/mistral-medium-3-1-2508",
)


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


@pytest.mark.parametrize("model", MEDIUM_3_5_MODELS)
def test_medium_3_5_specs(model):
    info = _load(MAIN_PATH).get(model)
    assert info is not None, f"{model} missing from model_prices_and_context_window.json"

    assert info["litellm_provider"] == "mistral"
    assert info["mode"] == "chat"

    assert info["input_cost_per_token"] == 1.5e-06
    assert info["output_cost_per_token"] == 7.5e-06

    assert info["max_input_tokens"] == 262144
    assert info["max_output_tokens"] == 262144
    assert info["max_tokens"] == 262144

    assert info["supports_reasoning"] is True
    assert info["supports_vision"] is True
    assert info["supports_function_calling"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_assistant_prefill"] is True

    routed_model, provider, _, _ = get_llm_provider(model=model)
    assert routed_model == model.split("/", 1)[1]
    assert provider == "mistral"


def test_mistral_medium_latest_resolves_to_medium_3_5(local_model_cost_map):
    """LIT-3883: the -latest alias was retargeted to Medium 3.5; get_model_info must
    return the 3.5 pricing/context/reasoning, not the stale Medium 3.1 values."""
    info = litellm.get_model_info(model="mistral/mistral-medium-latest")

    assert info["input_cost_per_token"] == 1.5e-06
    assert info["output_cost_per_token"] == 7.5e-06
    assert info["max_input_tokens"] == 262144
    assert info["supports_reasoning"] is True


def test_mistral_medium_2508_keeps_medium_3_1_specs():
    """The date-pinned 2508 alias is Medium 3.1 and must not inherit 3.5 pricing."""
    info = _load(MAIN_PATH).get("mistral/mistral-medium-2508")
    assert info is not None, "mistral/mistral-medium-2508 missing from cost map"

    assert info["input_cost_per_token"] == 4e-07
    assert info["output_cost_per_token"] == 2e-06
    assert info["max_input_tokens"] == 131072
    assert info.get("supports_reasoning") is not True


@pytest.mark.parametrize("model", SYNCED_MODELS)
def test_backup_matches_main(model):
    """Ensure the bundled (backup) cost map stays in sync with the canonical file."""
    main_cost = _load(MAIN_PATH)
    backup_cost = _load(BACKUP_PATH)

    assert backup_cost.get(model) == main_cost.get(model), f"{model} differs between main and backup model cost maps"
