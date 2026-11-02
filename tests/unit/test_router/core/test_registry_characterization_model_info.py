"""
What the registry knows about models: group info, deployment and router model info, configured
limits/mode/display name/service tiers/listing info, group usage this minute, remaining usage and
the rate limit headers ``set_response_headers`` attaches.
"""

import pytest

import litellm
from litellm.router_core import RouterCore
from litellm.types.router import Deployment, RouterCacheEnum
from litellm.utils import get_utc_datetime
from tests.unit.test_router.core.registry_helpers import MakeCore, typed, unique_id
from tests.unit.test_router.core.scripted_host import completion_response, deployment


def _usage_core(make_core: MakeCore) -> RouterCore:
    """gpt: a (tpm 100, rpm 10) and b (top-level tpm 1000, model_info tpm 50 / rpm 5); io: i (itpm 300, otpm 40)."""
    return make_core(
        model_list=[
            deployment("gpt", "a", model="openai/gpt-4o", tpm=100, rpm=10),
            {
                "model_name": "gpt",
                "litellm_params": {"model": "anthropic/claude-3-haiku-20240307", "api_key": "k"},
                "model_info": {"id": "b", "tpm": 50, "rpm": 5, "description": "desc b"},
                "tpm": 1000,
            },
            deployment("io", "i", model="openai/gpt-4o", itpm=300, otpm=40),
            deployment("plain", "p", model="openai/gpt-4o"),
        ],
        model_group_alias={"hid": {"model": "gpt", "hidden": True}, "al": "gpt"},
    )


def _minute() -> str:
    return get_utc_datetime().strftime("%H-%M")


async def _count(core: RouterCore, kind: RouterCacheEnum, deployment_id: str, model: str, value: int) -> None:
    key = kind.value.format(id=deployment_id, model=model, current_minute=_minute())
    await core.health().store.async_set_cache(key, value)


# -- model group info -----------------------------------------------------------------------------


def test_group_info_merges_providers_limits_and_description_across_deployments(make_core: MakeCore) -> None:
    info = _usage_core(make_core).registry().group("gpt")

    assert info is not None
    assert info.model_group == "gpt"
    assert info.providers == ["openai", "anthropic"]
    assert info.description == "desc b"
    assert info.mode == "chat"
    builtin = litellm.get_model_info("openai/gpt-4o")
    assert info.max_input_tokens == builtin["max_input_tokens"]
    assert info.input_cost_per_token == builtin["input_cost_per_token"]


def test_group_info_sums_tpm_and_rpm_with_top_level_values_before_model_info(make_core: MakeCore) -> None:
    info = _usage_core(make_core).registry().group("gpt")

    assert info is not None
    assert (info.tpm, info.rpm) == (1100, 15)


def test_group_info_sums_itpm_and_otpm(make_core: MakeCore) -> None:
    info = _usage_core(make_core).registry().group("io")

    assert info is not None
    assert (info.itpm, info.otpm, info.tpm, info.rpm) == (300, 40, None, None)


def test_group_info_through_an_alias_is_reported_under_the_alias(make_core: MakeCore) -> None:
    info = _usage_core(make_core).registry().group("al")

    assert info is not None
    assert (info.model_group, info.tpm) == ("al", 1100)


def test_group_info_of_a_hidden_alias_needs_include_hidden(make_core: MakeCore) -> None:
    registry = _usage_core(make_core).registry()

    assert registry.group("hid") is None
    shown = registry.group("hid", include_hidden=True)
    assert shown is not None
    assert shown.model_group == "hid"


def test_group_info_of_an_unknown_name_is_none(make_core: MakeCore) -> None:
    assert _usage_core(make_core).registry().group("nope") is None


def test_group_info_of_a_name_a_wildcard_serves_reads_the_requested_model(make_core: MakeCore) -> None:
    core = make_core(model_list=[deployment("openai/*", "w", model="openai/*")])

    info = core.registry().group("openai/gpt-4o-mini")

    assert info is not None
    assert (info.model_group, info.providers) == ("openai/gpt-4o-mini", ["openai"])
    assert info.max_input_tokens == litellm.get_model_info("openai/gpt-4o-mini")["max_input_tokens"]


def test_group_info_of_an_unmapped_model_uses_its_configured_mode_and_costs(make_core: MakeCore) -> None:
    core = make_core(
        model_list=[
            deployment(
                "unm",
                unique_id(),
                model="openai/totally-unmapped-registry-model",
                model_info={"mode": "embedding", "input_cost_per_token": "0.5"},
            )
        ]
    )

    info = core.registry().group("unm")

    assert info is not None
    assert (info.mode, info.input_cost_per_token, info.providers) == ("embedding", 0.5, ["openai"])
    assert info.max_input_tokens is None


def test_group_info_reports_configurable_clientside_auth_params(make_core: MakeCore) -> None:
    core = make_core(model_list=[deployment("gpt", "a", configurable_clientside_auth_params=["api_base"])])

    info = core.registry().group("gpt")

    assert info is not None
    assert info.configurable_clientside_auth_params == ["api_base"]


def test_cached_group_info_equals_group_info(make_core: MakeCore) -> None:
    registry = _usage_core(make_core).registry()

    assert registry.group("gpt", field="cached_info") == registry.group("gpt")


def test_cached_group_info_follows_a_deployment_added_later(make_core: MakeCore) -> None:
    core = _usage_core(make_core)
    before = core.registry().group("gpt", field="cached_info")
    assert before is not None

    core.add_deployment(typed("gpt", "c", tpm=1))  # pyright: ignore[reportAttributeAccessIssue]  # public on the core

    after = core.registry().group("gpt", field="cached_info")
    assert after is not None
    assert (before.tpm, after.tpm) == (1100, 1101)


def test_an_unknown_group_field_raises(make_core: MakeCore) -> None:
    with pytest.raises(ValueError, match="unknown group field 'bogus'"):
        _usage_core(make_core).registry().group("gpt", field="bogus")  # pyright: ignore[reportCallIssue, reportArgumentType]  # an invalid field on purpose


# -- configured values of a group -----------------------------------------------------------------


def _configured_core(make_core: MakeCore) -> RouterCore:
    return make_core(
        model_list=[
            deployment(
                "cfg",
                "c1",
                model="openai/gpt-4o",
                model_info={
                    "max_input_tokens": 1000,
                    "mode": "embedding",
                    "display_name": "Nice",
                    "service_tiers": ["flex"],
                },
            ),
            deployment(
                "cfg", "c2", model="openai/gpt-4o-mini", model_info={"max_input_tokens": 5000, "max_output_tokens": 77}
            ),
            deployment("blank", "b1", model_info={"mode": "  ", "display_name": ""}),
            deployment("bad", "x1", model_info={"max_input_tokens": "lots"}),
            deployment("openai/*", "w", model="openai/*"),
        ],
        model_group_alias={"cfg-alias": "cfg"},
    )


def test_token_limits_come_from_the_first_deployment_of_the_group(make_core: MakeCore) -> None:
    assert _configured_core(make_core).registry().group("cfg", field="token_limits") == (1000, None)


@pytest.mark.parametrize("name", ["nope", "openai/gpt-x", "cfg-alias"])
def test_token_limits_of_an_unknown_wildcard_or_alias_name_are_none(make_core: MakeCore, name: str) -> None:
    assert _configured_core(make_core).registry().group(name, field="token_limits") == (None, None)


def test_a_malformed_token_limit_reads_as_absent(make_core: MakeCore) -> None:
    assert _configured_core(make_core).registry().group("bad", field="token_limits") == (None, None)


@pytest.mark.parametrize(
    ("name", "mode", "display_name"),
    [("cfg", "embedding", "Nice"), ("blank", None, None), ("nope", None, None), ("cfg-alias", None, None)],
)
def test_mode_and_display_name_are_the_first_deployments_non_blank_values(
    make_core: MakeCore, name: str, mode: str | None, display_name: str | None
) -> None:
    registry = _configured_core(make_core).registry()

    assert registry.group(name, field="mode") == mode
    assert registry.group(name, field="display_name") == display_name


def test_listing_info_reports_the_widest_limits_and_each_backend_model_once(make_core: MakeCore) -> None:
    info = _configured_core(make_core).registry().group("cfg", field="listing_info")

    assert info is not None
    assert info.cost_map_keys == ("openai/gpt-4o", "openai/gpt-4o-mini")
    assert (info.max_input_tokens, info.max_output_tokens) == (5000, 77)


@pytest.mark.parametrize("name", ["nope", "openai/gpt-x", "cfg-alias"])
def test_listing_info_of_an_unknown_wildcard_or_alias_name_is_none(make_core: MakeCore, name: str) -> None:
    assert _configured_core(make_core).registry().group(name, field="listing_info") is None


def test_service_tiers_are_each_routable_deployments_value_in_order(make_core: MakeCore) -> None:
    registry = _configured_core(make_core).registry()

    assert registry.group("cfg", field="service_tiers") == (["flex"], None)
    assert registry.group("cfg-alias", field="service_tiers") == (["flex"], None)
    assert registry.group("nope", field="service_tiers") == ()


def test_service_tiers_leave_out_blocked_deployments_and_other_teams(make_core: MakeCore) -> None:
    core = make_core(
        model_list=[
            deployment("st", "s1", model_info={"service_tiers": ["a"], "blocked": True}),
            deployment("st", "s2", model_info={"service_tiers": ["b"]}),
            deployment("st", "s3", model_info={"service_tiers": ["c"], "team_id": "TA"}),
        ]
    )

    assert core.registry().group("st", field="service_tiers") == (["b"],)
    assert core.registry().group("st", field="service_tiers", team_id="TA") == (["b"], ["c"])
    assert core.registry().group("st", field="service_tiers", team_id="TB") == (["b"],)


# -- deployment model info --------------------------------------------------------------------------


def test_deployment_model_info_merges_custom_pricing_over_the_builtin_entry(make_core: MakeCore) -> None:
    model_id = unique_id()
    core = make_core(model_list=[deployment("gpt", model_id, model="openai/gpt-4o", input_cost_per_token=1e-3)])

    info = core.registry().deployment_model_info(model_id, "openai/gpt-4o")

    assert info is not None
    builtin = litellm.get_model_info("openai/gpt-4o")
    assert info["input_cost_per_token"] == 1e-3
    assert info["max_input_tokens"] == builtin["max_input_tokens"]


def test_deployment_model_info_without_custom_pricing_is_the_builtin_entry(make_core: MakeCore) -> None:
    model_id = unique_id()
    core = make_core(model_list=[deployment("gpt", model_id, model="openai/gpt-4o")])

    info = core.registry().deployment_model_info(model_id, "openai/gpt-4o")

    assert info is not None
    builtin = litellm.get_model_info("openai/gpt-4o")
    assert (info["input_cost_per_token"], info["output_cost_per_token"]) == (
        builtin["input_cost_per_token"],
        builtin["output_cost_per_token"],
    )


def test_deployment_model_info_of_an_unmapped_model_is_its_custom_entry(make_core: MakeCore) -> None:
    model_id = unique_id()
    core = make_core(
        model_list=[
            deployment(
                "unm",
                model_id,
                model="openai/unmapped-registry-thing",
                input_cost_per_token=5e-6,
                model_info={"max_input_tokens": 1000},
            )
        ]
    )

    info = core.registry().deployment_model_info(model_id, "openai/unmapped-registry-thing")

    assert info is not None
    assert (info["input_cost_per_token"], info["max_input_tokens"]) == (5e-6, 1000)


def test_deployment_model_info_of_an_unknown_id_and_model_is_none(make_core: MakeCore) -> None:
    core = make_core(model_list=[deployment("gpt", "a")])

    assert core.registry().deployment_model_info("missing-id", "nope/nothing-at-all") is None


def test_cached_deployment_model_info_equals_the_uncached_one(make_core: MakeCore) -> None:
    model_id = unique_id()
    core = make_core(model_list=[deployment("gpt", model_id, model="openai/gpt-4o", input_cost_per_token=1e-3)])

    assert core.registry().deployment_model_info(model_id, "openai/gpt-4o", cached=True) == (
        core.registry().deployment_model_info(model_id, "openai/gpt-4o")
    )


def test_discovered_model_info_is_empty_without_a_refresh(make_core: MakeCore) -> None:
    core = make_core(model_list=[deployment("gpt", "a")])

    assert dict(core.registry().discovered_model_info("a")) == {}
    assert dict(core.registry().discovered_model_info(None)) == {}


# -- router model info ------------------------------------------------------------------------------


def test_router_model_info_by_id_is_the_builtin_entry_with_the_deployments_model_info_on_top(
    make_core: MakeCore,
) -> None:
    model_id = unique_id()
    core = make_core(
        model_list=[
            deployment(
                "gpt", model_id, model="openai/gpt-4o", input_cost_per_token=1e-3, model_info={"max_input_tokens": 99}
            )
        ]
    )

    info = core.registry().router_model_info(None, "gpt", id=model_id)

    builtin = litellm.get_model_info("openai/gpt-4o")
    assert info["key"] == builtin["key"]
    assert info["max_input_tokens"] == 99
    assert info["max_output_tokens"] == builtin["max_output_tokens"]


def test_router_model_info_of_a_passed_deployment_dict(make_core: MakeCore) -> None:
    core = make_core(model_list=[deployment("gpt", "a")])

    info = core.registry().router_model_info(deployment("x", "x", model="openai/gpt-4o-mini"), "x")

    assert info["key"] == litellm.get_model_info("openai/gpt-4o-mini")["key"]


def test_router_model_info_reads_the_base_model(make_core: MakeCore) -> None:
    core = make_core(
        model_list=[
            deployment(
                "az", "z", model="azure/my-deployment", api_base="https://x", model_info={"base_model": "azure/gpt-4o"}
            )
        ]
    )

    info = core.registry().router_model_info(None, "az", id="z")

    assert info["max_input_tokens"] == litellm.get_model_info("azure/gpt-4o")["max_input_tokens"]


def test_router_model_info_of_a_wildcard_deployment_reads_the_requested_model(make_core: MakeCore) -> None:
    core = make_core(model_list=[deployment("openai/*", "w", model="openai/*")])

    info = core.registry().router_model_info(None, "openai/gpt-4o-mini", id="w")

    assert info["key"] == litellm.get_model_info("openai/gpt-4o-mini")["key"]


def test_router_model_info_for_an_unknown_id_and_no_deployment_raises(make_core: MakeCore) -> None:
    core = make_core(model_list=[deployment("gpt", "a")])

    with pytest.raises(ValueError, match="Deployment not found"):
        core.registry().router_model_info(None, "gpt", id="missing")


def test_router_model_info_of_a_deployment_without_litellm_params_raises(make_core: MakeCore) -> None:
    core = make_core(model_list=[deployment("gpt", "a")])

    with pytest.raises(ValueError, match="Deployment missing valid litellm_params"):
        core.registry().router_model_info({"model_name": "x", "litellm_params": {}, "model_info": {"id": "q"}}, "x")


def test_router_model_info_of_an_unmapped_model_is_the_zeroed_stub_construction_registered(make_core: MakeCore) -> None:
    """Construction registers every backend model in the cost map, so an unmapped one reads as free, not missing."""
    core = make_core(model_list=[deployment("gpt", "a", model="openai/not-a-known-registry-model")])

    info = core.registry().router_model_info(None, "gpt", id="a")

    assert info["key"] == "openai/not-a-known-registry-model"
    assert (info["input_cost_per_token"], info["output_cost_per_token"], info["max_input_tokens"]) == (0, 0, None)
    assert info["litellm_provider"] == "openai"


# -- usage this minute ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_group_usage_is_none_until_something_is_counted(make_core: MakeCore) -> None:
    core = _usage_core(make_core)

    assert await core.health().usage("gpt") == (None, None)
    assert await core.health().usage("nope") == (None, None)


@pytest.mark.asyncio
async def test_group_usage_sums_each_deployments_counters_for_this_minute(make_core: MakeCore) -> None:
    core = _usage_core(make_core)
    await _count(core, RouterCacheEnum.TPM, "a", "openai/gpt-4o", 30)
    await _count(core, RouterCacheEnum.TPM, "b", "anthropic/claude-3-haiku-20240307", 4)
    await _count(core, RouterCacheEnum.RPM, "b", "anthropic/claude-3-haiku-20240307", 2)

    assert await core.health().usage("gpt") == (34, 2)
    assert await core.health().usage("al") == (34, 2)


@pytest.mark.asyncio
async def test_io_token_usage_sums_input_and_output_counters(make_core: MakeCore) -> None:
    core = _usage_core(make_core)
    assert await core.get_model_group_io_token_usage("io") == (None, None)  # pyright: ignore[reportAttributeAccessIssue]  # public on the core
    await _count(core, RouterCacheEnum.ITPM, "i", "openai/gpt-4o", 7)

    assert await core.get_model_group_io_token_usage("io") == (7, None)  # pyright: ignore[reportAttributeAccessIssue]  # public on the core


@pytest.mark.asyncio
async def test_remaining_usage_is_each_limit_minus_this_minutes_usage(make_core: MakeCore) -> None:
    core = _usage_core(make_core)
    await _count(core, RouterCacheEnum.TPM, "a", "openai/gpt-4o", 30)
    await _count(core, RouterCacheEnum.RPM, "b", "anthropic/claude-3-haiku-20240307", 2)

    assert await core.health().usage("gpt", kind="remaining") == {
        "x-ratelimit-remaining-tokens": 1070,
        "x-ratelimit-limit-tokens": 1100,
        "x-ratelimit-remaining-requests": 13,
        "x-ratelimit-limit-requests": 15,
    }


@pytest.mark.asyncio
async def test_remaining_usage_of_an_io_limited_group_reports_input_and_output_token_headers(
    make_core: MakeCore,
) -> None:
    core = _usage_core(make_core)
    await _count(core, RouterCacheEnum.ITPM, "i", "openai/gpt-4o", 7)

    remaining = await core.health().usage("io", kind="remaining")

    assert remaining["x-ratelimit-limit-input-tokens"] == 300
    assert remaining["x-ratelimit-remaining-input-tokens"] == 293
    assert remaining["x-ratelimit-limit-output-tokens"] == 40
    assert remaining["x-ratelimit-remaining-output-tokens"] == 40
    assert "x-ratelimit-limit-tokens" not in remaining


@pytest.mark.asyncio
@pytest.mark.parametrize("group", ["plain", "nope"])
async def test_remaining_usage_of_a_group_without_limits_is_empty(make_core: MakeCore, group: str) -> None:
    assert await _usage_core(make_core).health().usage(group, kind="remaining") == {}


@pytest.mark.asyncio
async def test_an_unknown_usage_kind_raises(make_core: MakeCore) -> None:
    with pytest.raises(ValueError, match="unknown usage kind 'bogus'"):
        await _usage_core(make_core).health().usage("gpt", kind="bogus")  # pyright: ignore[reportCallIssue, reportArgumentType]  # an invalid kind on purpose


# -- set_response_headers ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_headers_name_the_group_and_its_remaining_limits(make_core: MakeCore) -> None:
    core = _usage_core(make_core)
    await _count(core, RouterCacheEnum.TPM, "a", "openai/gpt-4o", 30)
    response = completion_response()

    returned = await core.set_response_headers(response, model_group="gpt")  # pyright: ignore[reportAttributeAccessIssue]  # public on the core

    assert returned is response
    assert response._hidden_params["additional_headers"] == {
        "x-litellm-model-group": "gpt",
        "x-ratelimit-remaining-tokens": 1070,
        "x-ratelimit-limit-tokens": 1100,
        "x-ratelimit-remaining-requests": 15,
        "x-ratelimit-limit-requests": 15,
    }


@pytest.mark.asyncio
async def test_response_headers_without_a_group_carry_only_a_none_group(make_core: MakeCore) -> None:
    returned = await _usage_core(make_core).set_response_headers(completion_response(), model_group=None)  # pyright: ignore[reportAttributeAccessIssue]  # public on the core

    assert returned._hidden_params["additional_headers"] == {"x-litellm-model-group": None}


@pytest.mark.asyncio
async def test_response_headers_of_a_group_without_limits_carry_only_the_group(make_core: MakeCore) -> None:
    returned = await _usage_core(make_core).set_response_headers(completion_response(), model_group="plain")  # pyright: ignore[reportAttributeAccessIssue]  # public on the core

    assert returned._hidden_params["additional_headers"] == {"x-litellm-model-group": "plain"}


@pytest.mark.asyncio
async def test_response_headers_of_no_response_are_none(make_core: MakeCore) -> None:
    assert await _usage_core(make_core).set_response_headers(None, model_group="gpt") is None  # pyright: ignore[reportAttributeAccessIssue]  # public on the core


@pytest.mark.asyncio
async def test_response_headers_stamp_the_model_id_of_the_deployment_in_the_request(make_core: MakeCore) -> None:
    returned = await _usage_core(make_core).set_response_headers(  # pyright: ignore[reportAttributeAccessIssue]  # public on the core
        completion_response(), model_group="plain", request_kwargs={"metadata": {"model_info": {"id": "p"}}}
    )

    assert returned._hidden_params["model_id"] == "p"


# -- budgets per deployment -------------------------------------------------------------------------


def test_no_deployment_budget_means_no_budget_limiter(make_core: MakeCore) -> None:
    core = make_core(model_list=[deployment("gpt", "a")])

    assert core.registry().plugins("budget_limiter") is None


@pytest.mark.asyncio
async def test_adding_a_deployment_with_a_budget_installs_the_budget_limiter(make_core: MakeCore) -> None:
    core = make_core(model_list=[deployment("gpt", "a")])

    core.add_deployment(  # pyright: ignore[reportAttributeAccessIssue]  # public on the core
        Deployment(**deployment("gpt", "budgeted", max_budget=1.0, budget_duration="1d"))  # pyright: ignore[reportArgumentType]  # a model_list entry
    )

    limiter = core.registry().plugins("budget_limiter")
    assert limiter is not None
    assert type(limiter).__name__ == "RouterBudgetLimiting"


def test_a_team_without_ptu_shares_has_no_ptu_ceiling(make_core: MakeCore) -> None:
    core = make_core(model_list=[deployment("gpt", "a")])

    assert core.health().team_ptu_ceiling("TA", "gpt") is None
