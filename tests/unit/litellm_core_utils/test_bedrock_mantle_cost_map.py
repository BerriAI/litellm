from typing import Final

import litellm
from litellm.litellm_core_utils.bedrock_mantle_cost_map import expand_bedrock_mantle_views
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap, _finalize_model_cost_map
from litellm.llms.bedrock_mantle.common_utils import mantle_base_segment, mantle_supports_responses


def _runtime_row(input_cost: float) -> dict[str, object]:
    return {
        "litellm_provider": "bedrock_converse",
        "mode": "chat",
        "input_cost_per_token": input_cost,
        "output_cost_per_token": 3e-05,
        "max_input_tokens": 1_000_000,
        "bedrock_mantle": {
            "openai.gpt-x": {
                "mode": "responses",
                "max_input_tokens": 1_050_000,
                "use_openai_responses_path": True,
                "supported_endpoints": ["/v1/responses"],
                "search_context_cost_per_query": {"search_context_size_low": 0.01},
                "input_cost_per_token": 9.0,
            },
            "us-gov-west-1/openai.gpt-x": {"mode": "chat"},
        },
    }


def test_mantle_view_is_priced_from_the_runtime_row_and_follows_its_price_changes():
    for input_cost in (5.5e-06, 6.6e-06):
        expanded = expand_bedrock_mantle_views({"us.openai.gpt-x": _runtime_row(input_cost)})
        view = expanded["bedrock_mantle/openai.gpt-x"]
        assert view["input_cost_per_token"] == input_cost
        assert view["output_cost_per_token"] == 3e-05
        assert view["litellm_provider"] == "bedrock_mantle"


def test_mantle_view_keeps_surface_fields_and_block_only_prices():
    expanded: Final = expand_bedrock_mantle_views({"us.openai.gpt-x": _runtime_row(5.5e-06)})
    view: Final = expanded["bedrock_mantle/openai.gpt-x"]
    assert view["mode"] == "responses"
    assert view["max_input_tokens"] == 1_050_000
    assert view["use_openai_responses_path"] is True
    assert view["search_context_cost_per_query"] == {"search_context_size_low": 0.01}
    assert expanded["bedrock_mantle/us-gov-west-1/openai.gpt-x"]["input_cost_per_token"] == 5.5e-06
    assert expanded["us.openai.gpt-x"]["max_input_tokens"] == 1_000_000
    assert "bedrock_mantle" not in expanded["us.openai.gpt-x"]


def test_an_explicit_row_wins_over_a_derived_view():
    explicit: Final = {"litellm_provider": "bedrock_mantle", "input_cost_per_token": 1.0}
    expanded: Final = expand_bedrock_mantle_views(
        {"us.openai.gpt-x": _runtime_row(5.5e-06), "bedrock_mantle/openai.gpt-x": explicit}
    )
    assert expanded["bedrock_mantle/openai.gpt-x"] == explicit


def test_a_malformed_block_derives_nothing_and_is_dropped_from_the_host():
    row: Final = {"litellm_provider": "bedrock_converse", "bedrock_mantle": ["openai.gpt-x"]}
    assert expand_bedrock_mantle_views({"us.openai.gpt-x": row}) == {
        "us.openai.gpt-x": {"litellm_provider": "bedrock_converse"}
    }


def test_loaded_cost_map_serves_mantle_routing_and_model_lists_from_runtime_blocks():
    raw: Final = GetModelCostMap.load_local_model_cost_map()
    hosts: Final = {key: row for key, row in raw.items() if "bedrock_mantle" in row}
    assert hosts
    loaded: Final = _finalize_model_cost_map(GetModelCostMap.load_local_model_cost_map())
    mantle_models: Final = litellm.models_by_provider["bedrock_mantle"]
    for host_key, host in hosts.items():
        for mantle_id, surface in host["bedrock_mantle"].items():
            view = loaded[f"bedrock_mantle/{mantle_id}"]
            assert {k: v for k, v in view.items() if "cost" in k and k in host} == {
                k: v for k, v in host.items() if "cost" in k
            }, host_key
            assert mantle_supports_responses(mantle_id, loaded) is (
                "/v1/responses" in surface.get("supported_endpoints", []) or surface.get("mode") == "responses"
            )
            assert mantle_base_segment(mantle_id, loaded) == (
                "openai/v1" if surface.get("use_openai_responses_path") is True else "v1"
            )
            assert f"bedrock_mantle/{mantle_id}" in mantle_models
        assert "bedrock_mantle" not in loaded[host_key]
