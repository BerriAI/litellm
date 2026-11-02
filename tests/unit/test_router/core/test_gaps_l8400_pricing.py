"""
Gaps in the built-in pricing a custom-priced deployment inherits when the core registers it in
``litellm.model_cost`` (router_python_core.py ``_inherit_builtin_cache_pricing``, ``_cost_map_backend_model``,
``_inherit_builtin_service_tier_pricing``, ``_inherit_builtin_base_rates_for_off_peak``,
``_inherit_builtin_tiered_output_rate``), and the deployment checks of ``_create_deployment``
(access windows, PTU configuration). Expected rates are read from the built-in cost map, not hardcoded.
"""

from collections.abc import Mapping
from typing import Final

import pytest

import litellm
from tests.unit.test_router.core.registry_helpers import MakeCore, unique_id
from tests.unit.test_router.core.scripted_host import deployment

OFF_PEAK: Final = {"hours_utc": "01:00-02:00", "input_cost_per_token": 1e-9}


def registered(make_core: MakeCore, model: str, **params: object) -> Mapping[str, object]:  # kwargs-ok: litellm_params
    """The ``litellm.model_cost`` entry registered under a fresh deployment id."""
    model_id: Final = unique_id("l8400")
    make_core(model_list=[deployment("g", model_id, model=model, **params)])
    return litellm.model_cost[model_id]


def builtin(key: str) -> Mapping[str, object]:
    return litellm.model_cost[key]


# --- service-tier rates ---------------------------------------------------------------------------


def test_custom_input_pricing_inherits_the_backend_service_tier_rates(make_core: MakeCore) -> None:
    entry: Final = registered(make_core, "openai/gpt-4o", input_cost_per_token=1e-3)

    for field in (
        "input_cost_per_token_priority",
        "output_cost_per_token_priority",
        "cache_read_input_token_cost_priority",
    ):
        assert entry[field] == builtin("gpt-4o")[field]
    assert entry.get("input_cost_per_token_batches") is None  # batch rates are not a service tier


def test_a_user_set_service_tier_rate_wins_over_the_inherited_one(make_core: MakeCore) -> None:
    entry: Final = registered(make_core, "openai/gpt-4o", input_cost_per_token=1e-3, input_cost_per_token_priority=9e-3)

    assert entry["input_cost_per_token_priority"] == 9e-3
    assert entry["output_cost_per_token_priority"] == builtin("gpt-4o")["output_cost_per_token_priority"]


def test_output_only_custom_pricing_inherits_service_tiers_but_not_cache_pricing(make_core: MakeCore) -> None:
    entry: Final = registered(make_core, "openai/gpt-4o", output_cost_per_token=1e-3)

    assert entry["input_cost_per_token_priority"] == builtin("gpt-4o")["input_cost_per_token_priority"]
    assert entry.get("cache_read_input_token_cost") is None


def test_a_backend_missing_from_the_cost_map_contributes_nothing(make_core: MakeCore) -> None:
    entry: Final = registered(make_core, "openai/l8400-not-a-real-model", input_cost_per_token=1e-3)

    assert entry["input_cost_per_token"] == 1e-3
    assert entry.get("cache_read_input_token_cost") is None
    assert entry.get("input_cost_per_token_priority") is None


# --- which backend model the rates come from ------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [{"model_info": {"base_model": "gpt-5"}}, {"base_model": "gpt-5"}],
    ids=["model-info-base-model", "litellm-params-base-model"],
)
def test_the_base_model_names_the_backend_the_rates_are_inherited_from(
    make_core: MakeCore, params: dict[str, object]
) -> None:
    entry: Final = registered(make_core, "openai/my-fine-tune", input_cost_per_token=1e-3, **params)

    assert entry["cache_read_input_token_cost"] == builtin("gpt-5")["cache_read_input_token_cost"]
    assert entry["input_cost_per_token_flex"] == builtin("gpt-5")["input_cost_per_token_flex"]


# --- explicit cache-read rate -----------------------------------------------------------------------


def test_explicit_cache_read_rate_is_inherited_only_without_a_user_cache_read_rate(make_core: MakeCore) -> None:
    backend: Final = builtin("dashscope/qwen3.7-max")
    inherited: Final = registered(make_core, "dashscope/qwen3.7-max", input_cost_per_token=1e-3)
    user_set: Final = registered(
        make_core, "dashscope/qwen3.7-max", input_cost_per_token=1e-3, cache_read_input_token_cost=1e-8
    )

    assert inherited["explicit_cache_read_input_token_cost"] == backend["explicit_cache_read_input_token_cost"]
    assert inherited["cache_read_input_token_cost"] == backend["cache_read_input_token_cost"]
    assert user_set.get("explicit_cache_read_input_token_cost") is None
    assert user_set["cache_read_input_token_cost"] == 1e-8


# --- off-peak only ----------------------------------------------------------------------------------


def test_an_off_peak_only_deployment_copies_the_whole_backend_entry(make_core: MakeCore) -> None:
    backend: Final = builtin("deepseek-v4-flash")
    entry: Final = registered(make_core, "deepseek/deepseek-v4-flash", model_info={"off_peak_pricing": OFF_PEAK})

    assert entry["off_peak_pricing"] == OFF_PEAK
    for field in ("input_cost_per_token", "output_cost_per_token", "cache_read_input_token_cost", "max_input_tokens"):
        assert entry[field] == backend[field]
    assert entry["source"] == backend["source"]


def test_an_off_peak_deployment_keeps_its_own_fields(make_core: MakeCore) -> None:
    entry: Final = registered(
        make_core, "deepseek/deepseek-v4-flash", model_info={"off_peak_pricing": OFF_PEAK, "max_input_tokens": 7}
    )

    assert entry["max_input_tokens"] == 7
    assert entry["input_cost_per_token"] == builtin("deepseek-v4-flash")["input_cost_per_token"]


def test_an_off_peak_deployment_with_a_base_rate_copies_nothing_more(make_core: MakeCore) -> None:
    entry: Final = registered(
        make_core,
        "deepseek/deepseek-v4-flash",
        model_info={"off_peak_pricing": OFF_PEAK},
        input_cost_per_token=5e-3,
    )

    assert entry["input_cost_per_token"] == 5e-3
    assert entry.get("output_cost_per_token") is None
    assert entry.get("max_input_tokens") is None


def test_off_peak_pricing_given_in_litellm_params_is_not_registered(make_core: MakeCore) -> None:
    """Oddity: only ``model_info.off_peak_pricing`` counts; the same key in litellm_params is dropped."""
    entry: Final = registered(make_core, "deepseek/deepseek-v4-flash", off_peak_pricing=OFF_PEAK)

    assert entry.get("off_peak_pricing") is None
    assert entry.get("input_cost_per_token") is None


# --- tiered pricing output rate ---------------------------------------------------------------------


def test_a_tier_table_without_output_rates_inherits_the_backend_flat_output_rate(make_core: MakeCore) -> None:
    entry: Final = registered(
        make_core, "openai/gpt-4o", tiered_pricing=[{"input_cost_per_token": 1e-9, "range": [0, 1000]}]
    )

    assert entry["output_cost_per_token"] == builtin("gpt-4o")["output_cost_per_token"]


@pytest.mark.parametrize(
    ("model", "params"),
    [
        ("dashscope/qwen-flash", {"tiered_pricing": [{"input_cost_per_token": 1e-9, "range": [0, 1000]}]}),
        (
            "openai/gpt-4o",
            {"tiered_pricing": [{"input_cost_per_token": 1e-9, "output_cost_per_token": 2e-9, "range": [0, 1000]}]},
        ),
        (
            "openai/gpt-4o",
            {"tiered_pricing": [{"input_cost_per_token": 1e-9, "range": [0, 1000]}], "output_cost_per_token": 3e-9},
        ),
    ],
    ids=["tiered-only-backend", "every-tier-has-an-output-rate", "user-output-rate"],
)
def test_the_tiered_output_rate_is_not_inherited(make_core: MakeCore, model: str, params: dict[str, object]) -> None:
    entry: Final = registered(make_core, model, **params)

    assert entry.get("output_cost_per_token") == params.get("output_cost_per_token")


# --- _create_deployment checks ------------------------------------------------------------------------


def test_an_invalid_access_windows_config_raises_naming_the_model(make_core: MakeCore) -> None:
    with pytest.raises(ValueError, match=r"^model 'gpt': invalid model_info\.access_windows: "):
        make_core(model_list=[deployment("gpt", "a", model_info={"access_windows": "bad"})])


@pytest.mark.parametrize(
    "model_info",
    [
        {"access_windows": "bad"},
        {"ptu_shares": {"team-a": 1}},
    ],
    ids=["access-windows", "ptu-shares"],
)
def test_ignore_invalid_deployments_drops_a_deployment_with_an_invalid_config(
    make_core: MakeCore, model_info: dict[str, object]
) -> None:
    core: Final = make_core(
        model_list=[deployment("gpt", "a", model_info=model_info), deployment("gpt", "b")],
        ignore_invalid_deployments=True,
    )

    assert core.registry().names(kind="ids") == ["b"]


def test_ptu_shares_without_a_count_and_rate_raise(make_core: MakeCore) -> None:
    with pytest.raises(
        ValueError,
        match=r"^PTU configuration on model 'gpt' is invalid: ptu_count and cost_per_ptu_per_hour are required "
        r"when ptu_shares is set",
    ):
        make_core(model_list=[deployment("gpt", "a", model_info={"ptu_shares": {"team-a": 1}})])


def test_an_inverted_ptu_window_with_shares_raises(make_core: MakeCore) -> None:
    with pytest.raises(
        ValueError,
        match=r"^PTU configuration on model 'gpt' is invalid: ptu_effective_to must be after ptu_effective_from",
    ):
        make_core(
            model_list=[
                deployment(
                    "gpt",
                    "a",
                    model_info={
                        "ptu_effective_from": "2026-01-02T00:00:00Z",
                        "ptu_effective_to": "2026-01-01T00:00:00Z",
                        "ptu_shares": {"team-a": 1},
                        "ptu_count": 1,
                        "cost_per_ptu_per_hour": 1,
                    },
                )
            ]
        )


def test_a_db_sourced_deployment_skips_the_ptu_config_check(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("gpt", "a", model_info={"ptu_shares": {"team-a": 1}, "db_model": True})]
    )

    assert core.registry().names(kind="ids") == ["a"]
