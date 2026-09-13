import json
from pathlib import Path

import pytest

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.types.utils import PromptTokensDetailsWrapper, Usage
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


def test_baseten_glm_5_3_specs():
    info = _load(MAIN_PATH).get(MODEL)
    assert info is not None, f"{MODEL} missing from model_prices_and_context_window.json"

    assert info["litellm_provider"] == "baseten"
    assert info["mode"] == "chat"

    assert info["input_cost_per_token"] == INPUT_COST
    assert info["output_cost_per_token"] == OUTPUT_COST
    assert info["cache_read_input_token_cost"] == CACHED_INPUT_COST

    assert info["max_input_tokens"] == 1048576
    assert info["max_output_tokens"] == 262144
    assert info["max_tokens"] == 262144

    assert info["supports_function_calling"] is True
    assert info["supports_prompt_caching"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_vision"] is True
    assert info["supported_modalities"] == ["text", "image"]
    assert info["supported_output_modalities"] == ["text"]

    routed_model, provider, _, _ = get_llm_provider(model=MODEL)
    assert routed_model == "zai-org/GLM-5.3"
    assert provider == "baseten"


def test_baseten_glm_5_3_capabilities_are_visible_to_callers(local_model_cost_map):
    """The entry advertises prompt caching and tool calling, so the helpers every
    caller checks before sending a request must say so too."""
    assert supports_prompt_caching(model=MODEL) is True
    assert supports_function_calling(model=MODEL) is True

    info = litellm.get_model_info(model="zai-org/GLM-5.3", custom_llm_provider="baseten")
    assert info["max_input_tokens"] == 1048576
    assert info["max_output_tokens"] == 262144


def test_cached_prompt_tokens_bill_at_the_cached_rate(local_model_cost_map):
    """A cache hit reports its reused tokens under prompt_tokens_details, and those
    tokens cost a tenth of the input rate, not the full rate and not nothing."""
    usage = Usage(
        prompt_tokens=21010,
        completion_tokens=100,
        total_tokens=21110,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=20992),
    )

    prompt_cost, completion_cost = litellm.cost_per_token(
        model=MODEL, usage_object=usage, custom_llm_provider="baseten"
    )

    assert prompt_cost == pytest.approx(18 * INPUT_COST + 20992 * CACHED_INPUT_COST)
    assert completion_cost == pytest.approx(100 * OUTPUT_COST)


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
    """The entry must not claim a capability whose request parameter BasetenConfig
    refuses.

    ``BasetenConfig.get_supported_openai_params`` returns one hardcoded list for every
    Baseten model, and it carries neither ``parallel_tool_calls`` nor
    ``reasoning_effort``. Baseten's own Model API does take ``reasoning_effort``, but
    litellm's Baseten path drops it (``drop_params=True``) or raises
    ``UnsupportedParamsError`` (``drop_params=False``), so declaring
    ``supports_parallel_function_calling``, ``supports_reasoning`` or
    ``reasoning_effort_levels`` here would advertise a level the gateway then refuses to
    send. Wiring those params through the Baseten config is separate work; until it
    lands, the registry stays honest.
    """
    supported = litellm.get_supported_openai_params(model="zai-org/GLM-5.3", custom_llm_provider="baseten")
    assert supported is not None

    entry = _load(MAIN_PATH)[MODEL]

    capability_to_param = {
        "supports_function_calling": "tools",
        "supports_tool_choice": "tool_choice",
        "supports_response_schema": "response_format",
        "supports_parallel_function_calling": "parallel_tool_calls",
        "supports_reasoning": "reasoning_effort",
    }
    for capability, param in capability_to_param.items():
        if entry.get(capability):
            assert param in supported, f"{MODEL} advertises {capability} but baseten drops/rejects {param}"

    assert "reasoning_effort_levels" not in entry, (
        "reasoning_effort_levels advertises accepted reasoning_effort values, which the Baseten path does not accept"
    )
    assert "thinking_always_on" not in entry, (
        "thinking_always_on is only read by AnthropicModelInfo._is_always_on_thinking_model, "
        "which no Baseten route reaches"
    )

    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.utils.get_optional_params(
            model="zai-org/GLM-5.3",
            custom_llm_provider="baseten",
            parallel_tool_calls=True,
            reasoning_effort="high",
            drop_params=False,
        )
