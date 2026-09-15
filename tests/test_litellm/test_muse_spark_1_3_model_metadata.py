import json
from pathlib import Path

import pytest

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import StandardBuiltInToolCostTracking

MUSE_SPARK_STANDARD = "meta/muse-spark-1.3"
MUSE_SPARK_CONTRIBUTOR = "meta/muse-spark-1.3-contributor"
WEB_SEARCH_COST_PER_QUERY = 0.0025

PRICING = (
    (MUSE_SPARK_STANDARD, 1.25e-06, 1.5e-07, 4.25e-06),
    (MUSE_SPARK_CONTRIBUTOR, 1e-07, 2e-09, 2e-07),
)


def _load_cost_map(filename: str = "model_prices_and_context_window.json") -> dict:
    with open(Path(__file__).parents[2] / filename) as f:
        return json.load(f)


@pytest.mark.parametrize("model", (MUSE_SPARK_STANDARD, MUSE_SPARK_CONTRIBUTOR))
def test_muse_spark_1_3_routes_to_meta_model_api(model: str):
    routed_model, provider, _, api_base = get_llm_provider(model=model, api_key="sk-test")

    assert routed_model == model.split("/", 1)[1]
    assert provider == "meta"
    assert api_base == "https://api.meta.ai/v1"


@pytest.mark.parametrize("model", (MUSE_SPARK_STANDARD, MUSE_SPARK_CONTRIBUTOR))
def test_muse_spark_1_3_web_search_cost_per_query(local_model_cost_map, model: str):
    info = litellm.get_model_info(model=model)

    assert StandardBuiltInToolCostTracking.get_cost_for_web_search(model_info=info) == WEB_SEARCH_COST_PER_QUERY


def test_muse_spark_contributor_tier_is_cheaper_than_standard():
    cost_map = _load_cost_map()
    standard = cost_map[MUSE_SPARK_STANDARD]
    contributor = cost_map[MUSE_SPARK_CONTRIBUTOR]

    for field in ("input_cost_per_token", "output_cost_per_token", "cache_read_input_token_cost"):
        assert contributor[field] < standard[field], f"contributor {field} should undercut the standard tier"
