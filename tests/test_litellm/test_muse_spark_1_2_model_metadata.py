import json
from pathlib import Path

import pytest

import litellm
from litellm.cost_calculator import cost_per_token
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import StandardBuiltInToolCostTracking

MUSE_SPARK_STANDARD = "meta/muse-spark-1.2"
MUSE_SPARK_CONTRIBUTOR = "meta/muse-spark-1.2-contributor"
WEB_SEARCH_COST_PER_QUERY = 0.0025

PRICING = (
    (MUSE_SPARK_STANDARD, 1.25e-06, 1.5e-07, 4.25e-06),
    (MUSE_SPARK_CONTRIBUTOR, 1e-07, 2e-09, 2e-07),
)


def _load_cost_map(filename: str = "model_prices_and_context_window.json") -> dict:
    with open(Path(__file__).parents[2] / filename) as f:
        return json.load(f)



@pytest.mark.parametrize("model, input_cost, cached_cost, output_cost", PRICING)
def test_muse_spark_1_2_model_info(model: str, input_cost: float, cached_cost: float, output_cost: float):
    info = _load_cost_map().get(model)
    assert info is not None, f"{model} not found in model_prices_and_context_window.json"

    assert info["litellm_provider"] == "meta"
    assert info["mode"] == "chat"

    assert info["input_cost_per_token"] == input_cost
    assert info["output_cost_per_token"] == output_cost
    assert info["cache_read_input_token_cost"] == cached_cost

    assert info["max_input_tokens"] == 1048576
    assert info["max_output_tokens"] == 131072
    assert info["max_tokens"] == 131072

    assert info["supports_function_calling"] is True
    assert info["supports_parallel_function_calling"] is True
    assert info["supports_prompt_caching"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_vision"] is True
    assert info["supports_pdf_input"] is True
    assert info["supports_web_search"] is True
    assert info["supports_minimal_reasoning_effort"] is True
    assert info["supports_xhigh_reasoning_effort"] is True

    assert info["supported_endpoints"] == ["/v1/chat/completions", "/v1/responses", "/v1/messages"]
    assert info["supported_modalities"] == ["text", "image", "video"]
    assert info["supported_output_modalities"] == ["text"]

    assert info["search_context_cost_per_query"] == {
        "search_context_size_high": WEB_SEARCH_COST_PER_QUERY,
        "search_context_size_low": WEB_SEARCH_COST_PER_QUERY,
        "search_context_size_medium": WEB_SEARCH_COST_PER_QUERY,
    }


@pytest.mark.parametrize("model, input_cost, cached_cost, output_cost", PRICING)
def test_muse_spark_1_2_cost_per_token(
    local_model_cost_map, model: str, input_cost: float, cached_cost: float, output_cost: float
):
    prompt_cost, completion_cost = cost_per_token(model=model, prompt_tokens=1000, completion_tokens=500)

    assert prompt_cost == pytest.approx(1000 * input_cost)
    assert completion_cost == pytest.approx(500 * output_cost)


@pytest.mark.parametrize("model", (MUSE_SPARK_STANDARD, MUSE_SPARK_CONTRIBUTOR))
def test_muse_spark_1_2_routes_to_meta_model_api(model: str):
    routed_model, provider, _, api_base = get_llm_provider(model=model, api_key="sk-test")

    assert routed_model == model.split("/", 1)[1]
    assert provider == "meta"
    assert api_base == "https://api.meta.ai/v1"


@pytest.mark.parametrize("model", (MUSE_SPARK_STANDARD, MUSE_SPARK_CONTRIBUTOR))
def test_muse_spark_1_2_web_search_cost_per_query(local_model_cost_map, model: str):
    info = litellm.get_model_info(model=model)

    assert StandardBuiltInToolCostTracking.get_cost_for_web_search(model_info=info) == WEB_SEARCH_COST_PER_QUERY


@pytest.mark.parametrize("model", (MUSE_SPARK_STANDARD, MUSE_SPARK_CONTRIBUTOR))
def test_muse_spark_1_2_backup_matches_main(model: str):
    """Ensure the bundled model cost map stays in sync with the canonical file."""
    main_cost = _load_cost_map()
    backup_cost = _load_cost_map("litellm/model_prices_and_context_window_backup.json")

    assert backup_cost.get(model) == main_cost.get(model), f"{model} differs between main and backup model cost maps"


def test_muse_spark_contributor_tier_is_cheaper_than_standard():
    cost_map = _load_cost_map()
    standard = cost_map[MUSE_SPARK_STANDARD]
    contributor = cost_map[MUSE_SPARK_CONTRIBUTOR]

    for field in ("input_cost_per_token", "output_cost_per_token", "cache_read_input_token_cost"):
        assert contributor[field] < standard[field], f"contributor {field} should undercut the standard tier"
