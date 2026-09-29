"""Tests for per-team PTU shares: who a shared deployment is served to and what a share is worth."""

from typing import Final

from litellm.llms.azure.ptu_capacity import AZURE_PTU_CAPACITY
from litellm.router_utils.ptu_shares import (
    PTUTeamCeiling,
    filter_ptu_shared_deployments,
    model_group_deployments,
    model_group_ptu_capacity,
    ptu_capacity_warning,
    team_ptu_ceiling,
)

_GPT41: Final = AZURE_PTU_CAPACITY["gpt-4.1"]
_GPT4O: Final = AZURE_PTU_CAPACITY["gpt-4o"]
_GPT6SOL: Final = AZURE_PTU_CAPACITY["gpt-6-sol"]
_SHARES: Final = {"team-a": 30, "team-b": 20}


def _shared(model: str = "azure/gpt-4.1", shares: object = _SHARES, deployment_id: str = "shared") -> dict:
    return {
        "model_name": "gpt-4.1-ptu",
        "litellm_params": {"model": model},
        "model_info": {
            "id": deployment_id,
            "ptu_count": 50,
            "cost_per_ptu_per_hour": 1.0,
            "ptu_effective_from": "2026-01-01T00:00:00Z",
            "ptu_shares": shares,
        },
    }


def _single_team(model: str = "azure/gpt-4.1") -> dict:
    return {
        "model_name": "gpt-4.1-ptu",
        "litellm_params": {"model": model},
        "model_info": {
            "id": "single",
            "team_id": "team-a",
            "ptu_count": 50,
            "cost_per_ptu_per_hour": 1.0,
            "ptu_effective_from": "2026-01-01T00:00:00Z",
        },
    }


_OPEN: Final = {"model_name": "gpt-4.1-ptu", "litellm_params": {"model": "azure/gpt-4.1"}, "model_info": {"id": "open"}}


def _unaliased_ceiling(deployments: list[dict], team_id: str, requested_model: str) -> PTUTeamCeiling | None:
    return team_ptu_ceiling(deployments, deployments, team_id, requested_model)


def test_a_team_holding_a_share_keeps_the_shared_deployment():
    result: Final = filter_ptu_shared_deployments([_shared(), _OPEN], "team-a")
    assert [d["model_info"]["id"] for d in result.deployments] == ["shared", "open"]
    assert result.withheld is False


def test_a_team_without_a_share_only_sees_the_unshared_deployments():
    result: Final = filter_ptu_shared_deployments([_shared(), _OPEN], "team-c")
    assert [d["model_info"]["id"] for d in result.deployments] == ["open"]
    assert result.withheld is True


def test_a_caller_with_no_team_holds_no_share():
    for team_id in (None, ""):
        result = filter_ptu_shared_deployments([_shared()], team_id)
        assert result.deployments == ()
        assert result.withheld is True


def test_a_single_team_deployment_and_a_malformed_share_map_are_not_filtered_here():
    """A ``team_id`` deployment is scoped by the router's team filter, and an unusable
    ``ptu_shares`` is refused at registration, so neither is withheld by the share filter."""
    result: Final = filter_ptu_shared_deployments([_single_team(), _shared(shares={"team-a": 0})], "team-z")
    assert [d["model_info"]["id"] for d in result.deployments] == ["single", "shared"]
    assert result.withheld is False


def test_a_share_converts_to_the_models_input_tpm_per_ptu():
    ceiling: Final = _unaliased_ceiling([_shared()], "team-a", "gpt-4.1-ptu")
    assert ceiling == PTUTeamCeiling(
        model_group="gpt-4.1-ptu",
        tpm_limit=30 * _GPT41.input_tpm_per_ptu,
        output_to_input_ratio=_GPT41.output_to_input_ratio,
        cached_input_ratio=_GPT41.cached_input_ratio,
    )


def test_shares_across_deployments_add_up_and_the_larger_output_ratio_wins():
    gpt4o: Final = _shared(model="azure/gpt-4o", shares={"team-a": 10}, deployment_id="shared-4o")
    ceiling: Final = _unaliased_ceiling([_shared(), gpt4o, _OPEN], "team-a", "gpt-4.1-ptu")
    assert ceiling is not None
    assert ceiling.tpm_limit == 30 * _GPT41.input_tpm_per_ptu + 10 * _GPT4O.input_tpm_per_ptu
    assert ceiling.output_to_input_ratio == max(_GPT41.output_to_input_ratio, _GPT4O.output_to_input_ratio)


def test_the_larger_cached_input_ratio_wins_across_deployments():
    """A team sharing two models is weighted by the one that charges more for cache reads,
    whichever order the deployments come in."""
    gpt6sol: Final = _shared(model="azure/gpt-6-sol", shares={"team-a": 10}, deployment_id="shared-6")
    ceiling: Final = _unaliased_ceiling([_shared(), gpt6sol], "team-a", "gpt-4.1-ptu")
    assert ceiling is not None
    assert _GPT41.cached_input_ratio < _GPT6SOL.cached_input_ratio
    assert ceiling.cached_input_ratio == _GPT6SOL.cached_input_ratio


def test_a_group_is_served_by_name_or_by_a_team_scoped_deployments_public_name():
    """A deployment registered for one team is renamed to a unique internal name and keeps
    the name callers use in ``team_public_model_name``."""
    team_scoped: Final = {
        "model_name": "gpt-4.1-ptu-3f9c1b",
        "litellm_params": {"model": "azure/gpt-4.1"},
        "model_info": {"id": "team-scoped", "team_id": "team-a", "team_public_model_name": "gpt-4.1-ptu"},
    }
    other: Final = {"model_name": "other", "litellm_params": {"model": "azure/gpt-4o"}, "model_info": {"id": "other"}}
    deployments: Final = [team_scoped, _OPEN, other]
    served: Final = model_group_deployments(deployments, "gpt-4.1-ptu")
    assert [d["model_info"]["id"] for d in served] == ["team-scoped", "open"]
    assert model_group_deployments(deployments, "gpt-4.1-ptu-3f9c1b") == (team_scoped,)
    assert model_group_deployments(deployments, "missing") == ()


def test_no_share_or_no_sizing_row_sets_no_ceiling():
    assert _unaliased_ceiling([_shared()], "team-c", "gpt-4.1-ptu") is None
    assert _unaliased_ceiling([_shared(model="azure/unknown-deployment")], "team-a", "gpt-4.1-ptu") is None
    assert _unaliased_ceiling([_single_team(), _OPEN], "team-a", "gpt-4.1-ptu") is None


def test_naming_a_shared_deployment_by_id_or_provider_model_draws_on_its_groups_ceiling():
    """The router serves a deployment id or a provider model string when no group has that
    name, so those names share the group's ceiling instead of bypassing it."""
    payg: Final = {"model_name": "gpt-4.1-payg", "litellm_params": {"model": "azure/gpt-4.1"}, "model_info": {"id": "payg"}}
    deployments: Final = [payg, _shared(), _OPEN]
    by_group: Final = _unaliased_ceiling(deployments, "team-a", "gpt-4.1-ptu")
    assert by_group is not None
    assert by_group.model_group == "gpt-4.1-ptu"
    assert _unaliased_ceiling(deployments, "team-a", "shared") == by_group
    assert _unaliased_ceiling(deployments, "team-a", "azure/gpt-4.1") == by_group
    assert _unaliased_ceiling(deployments, "team-a", "payg") is None
    assert _unaliased_ceiling(deployments, "team-a", "missing") is None


def test_a_group_name_wins_over_a_deployment_id_it_collides_with():
    """The router routes a name that is both a group and a deployment id to the group."""
    colliding: Final = {
        "model_name": "shared",
        "litellm_params": {"model": "azure/gpt-4o"},
        "model_info": {"id": "colliding"},
    }
    assert _unaliased_ceiling([_shared(), colliding], "team-a", "shared") is None


def test_a_team_scoped_deployment_named_by_id_draws_on_its_public_groups_ceiling():
    team_scoped: Final = {
        "model_name": "gpt-4.1-ptu-3f9c1b",
        "litellm_params": {"model": "azure/gpt-4.1"},
        "model_info": {**_shared()["model_info"], "id": "team-scoped", "team_public_model_name": "gpt-4.1-ptu"},
    }
    ceiling: Final = _unaliased_ceiling([team_scoped], "team-a", "team-scoped")
    assert ceiling is not None
    assert ceiling.model_group == "gpt-4.1-ptu"
    assert ceiling == _unaliased_ceiling([team_scoped], "team-a", "gpt-4.1-ptu")


def test_alias_and_routing_group_copies_do_not_split_a_deployments_ceiling():
    """The router lists alias and routing-group copies of a deployment under their own names
    ahead of its own rows, so every name still resolves to the deployment's group."""
    shared: Final = _shared()
    listed: Final = [{**shared, "model_name": "ptu-alias"}, {**shared, "model_name": "ptu-routing-group"}, shared]
    by_group: Final = team_ptu_ceiling(listed, [shared], "team-a", "gpt-4.1-ptu")
    assert by_group is not None
    assert by_group.model_group == "gpt-4.1-ptu"
    for name in ("shared", "azure/gpt-4.1", "ptu-routing-group"):
        assert team_ptu_ceiling(listed, [shared], "team-a", name) == by_group


def test_a_provider_model_draws_on_the_group_where_the_team_holds_its_share():
    """Two groups share deployments of one provider model among different teams, and the router
    serves each team only the one it holds a share of."""
    east: Final = _shared(shares={"team-a": 30}, deployment_id="east")
    west: Final = {**_shared(shares={"team-b": 20}, deployment_id="west"), "model_name": "gpt-4.1-ptu-west"}
    by_provider_model: Final = _unaliased_ceiling([east, west], "team-b", "azure/gpt-4.1")
    assert by_provider_model is not None
    assert by_provider_model.model_group == "gpt-4.1-ptu-west"
    assert by_provider_model == _unaliased_ceiling([east, west], "team-b", "gpt-4.1-ptu-west")
    assert _unaliased_ceiling([east, west], "team-a", "azure/gpt-4.1") == _unaliased_ceiling(
        [east, west], "team-a", "gpt-4.1-ptu"
    )


def test_a_groups_capacity_comes_from_its_first_reserved_deployment_with_a_row():
    assert model_group_ptu_capacity([_OPEN, _single_team()]) is _GPT41
    assert model_group_ptu_capacity([_shared(model="azure/unknown"), _shared(model="azure/gpt-4o")]) is _GPT4O
    assert model_group_ptu_capacity([_OPEN]) is None
    assert model_group_ptu_capacity([]) is None


def test_a_reserved_deployment_without_a_sizing_row_is_warned_about_by_name():
    warning: Final = ptu_capacity_warning("gpt-4.1-ptu", _shared(model="azure/my-ptu-deployment"))
    assert warning is not None
    assert "gpt-4.1-ptu" in warning
    assert "base_model" in warning


def test_a_sized_reservation_and_an_unreserved_deployment_raise_no_warning():
    assert ptu_capacity_warning("gpt-4.1-ptu", _shared()) is None
    assert ptu_capacity_warning("gpt-4.1-ptu", _single_team()) is None
    unsized_open: Final = {**_OPEN, "litellm_params": {"model": "azure/my-ptu-deployment"}}
    assert ptu_capacity_warning("gpt-4.1-ptu", unsized_open) is None


def test_an_unsized_single_team_azure_reservation_is_warned_about_its_ptu_hours_only():
    warning: Final = ptu_capacity_warning("gpt-4.1-ptu", _single_team(model="azure/my-ptu-deployment"))
    assert warning is not None
    assert "PTU hours" in warning
    assert "ceiling" not in warning


def test_an_unsized_single_team_reservation_on_another_provider_is_not_warned_about():
    assert ptu_capacity_warning("claude-ptu", _single_team(model="anthropic/claude-sonnet-4-5")) is None


def test_a_bare_model_name_counts_as_azure_through_custom_llm_provider():
    deployment: Final = {
        **_single_team(model="my-ptu-deployment"),
        "litellm_params": {"model": "my-ptu-deployment", "custom_llm_provider": "azure"},
    }
    warning: Final = ptu_capacity_warning("gpt-4.1-ptu", deployment)
    assert warning is not None
    assert "PTU hours" in warning
