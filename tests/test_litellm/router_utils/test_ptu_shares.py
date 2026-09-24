"""Tests for per-team PTU shares: who a shared deployment is served to and what a share is worth."""

from typing import Final

from litellm.litellm_core_utils.azure_ptu_capacity import AZURE_PTU_CAPACITY
from litellm.router_utils.ptu_shares import (
    PTUTeamCeiling,
    filter_ptu_shared_deployments,
    model_group_ptu_capacity,
    ptu_capacity_warning,
    team_ptu_ceiling,
)

_GPT41: Final = AZURE_PTU_CAPACITY["gpt-4.1"]
_GPT4O: Final = AZURE_PTU_CAPACITY["gpt-4o"]
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
    ceiling: Final = team_ptu_ceiling([_shared()], "team-a")
    assert ceiling == PTUTeamCeiling(
        tpm_limit=30 * _GPT41.input_tpm_per_ptu, output_to_input_ratio=_GPT41.output_to_input_ratio
    )


def test_shares_across_deployments_add_up_and_the_larger_output_ratio_wins():
    gpt4o: Final = _shared(model="azure/gpt-4o", shares={"team-a": 10}, deployment_id="shared-4o")
    ceiling: Final = team_ptu_ceiling([_shared(), gpt4o, _OPEN], "team-a")
    assert ceiling is not None
    assert ceiling.tpm_limit == 30 * _GPT41.input_tpm_per_ptu + 10 * _GPT4O.input_tpm_per_ptu
    assert ceiling.output_to_input_ratio == max(_GPT41.output_to_input_ratio, _GPT4O.output_to_input_ratio)


def test_no_share_or_no_sizing_row_sets_no_ceiling():
    assert team_ptu_ceiling([_shared()], "team-c") is None
    assert team_ptu_ceiling([_shared(model="azure/unknown-deployment")], "team-a") is None
    assert team_ptu_ceiling([_single_team(), _OPEN], "team-a") is None


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
    assert ptu_capacity_warning("gpt-4.1-ptu", {**_OPEN, "litellm_params": {"model": "azure/my-ptu-deployment"}}) is None
