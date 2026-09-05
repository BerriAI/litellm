"""Tests for the shared PTU rules: which deployments accrue flat cost, and what that zeroes."""

import os
from datetime import date, datetime, timezone
from unittest.mock import patch

import pytest

from litellm.litellm_core_utils.ptu_pricing import (
    ptu_config_error,
    ptu_identity_error,
    CUSTOM_PRICING_FIELDS,
    PTU_EMPTIED_PRICING_FIELDS,
    PTU_ZEROED_PRICING_FIELDS,
    PTU_ZEROED_TABLE_FIELDS,
    SEARCH_CONTEXT_SIZES,
    ptu_terms,
    zeroed_ptu_pricing,
)
from litellm.types.router import ModelInfo

_VALID = {
    "team_id": "team-alpha",
    "ptu_count": 100,
    "cost_per_ptu_per_hour": 0.02,
    "ptu_effective_from": "2026-01-01T00:00:00Z",
}


def _with_flag(model_info, declared=None, enabled=True):
    with patch.dict(os.environ, {"LITELLM_ENABLE_PTU_COST_ATTRIBUTION": "True" if enabled else ""}, clear=False):
        return zeroed_ptu_pricing(model_info, declared or {})


def test_a_complete_reservation_is_accepted():
    terms = ptu_terms(_VALID)

    assert terms is not None
    assert terms.team_id == "team-alpha"
    assert terms.ptu_count == 100
    assert terms.effective_from == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert terms.effective_to is None


@pytest.mark.parametrize(
    "override",
    [
        {"team_id": None},
        {"team_id": ""},
        {"ptu_count": None},
        {"cost_per_ptu_per_hour": None},
        {"ptu_count": 0},
        {"ptu_count": -1},
        {"ptu_count": ModelInfo.MAX_PTU_COUNT + 1},
        {"cost_per_ptu_per_hour": -0.01},
        {"cost_per_ptu_per_hour": ModelInfo.MAX_COST_PER_PTU_PER_HOUR + 1},
        {"ptu_count": "not-a-number"},
        {"ptu_effective_from": None},
        {"ptu_effective_from": "not-a-date"},
        {"ptu_effective_to": "not-a-date"},
        {"ptu_effective_to": "2025-01-01T00:00:00Z"},
        {"ptu_effective_to": "2026-01-01T00:00:00Z"},
    ],
    ids=[
        "no team",
        "blank team",
        "no count",
        "no rate",
        "zero count",
        "negative count",
        "count over the cap",
        "negative rate",
        "rate over the cap",
        "count not a number",
        "no start",
        "unparseable start",
        "unparseable end",
        "end before start",
        "end equal to start",
    ],
)
def test_an_incomplete_reservation_accrues_nothing(override):
    """Anything the rollup declines to charge must also decline to be zeroed, or the
    deployment serves its traffic for free with nothing charged in its place."""
    assert ptu_terms({**_VALID, **override}) is None
    assert _with_flag({**_VALID, **override}) is None


def test_a_naive_start_is_read_as_utc():
    """config.yaml is hand-typed, and pydantic hands back a naive datetime for a date with
    no offset."""
    terms = ptu_terms({**_VALID, "ptu_effective_from": datetime(2026, 5, 1, 12, 0)})

    assert terms is not None
    assert terms.effective_from == datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)


def test_an_offset_start_is_converted_rather_than_relabelled():
    terms = ptu_terms({**_VALID, "ptu_effective_from": "2026-05-01T12:00:00-05:00"})

    assert terms is not None
    assert terms.effective_from == datetime(2026, 5, 1, 17, 0, tzinfo=timezone.utc)


def test_nothing_is_zeroed_while_the_feature_is_off():
    """No flat cost accrues with the flag off, so zeroing would serve the traffic free."""
    assert _with_flag(_VALID, enabled=False) is None


def test_the_standing_rates_are_all_zeroed():
    override = _with_flag(_VALID)

    assert override is not None
    assert [field for field in PTU_ZEROED_PRICING_FIELDS if override[field] != 0.0] == []


def test_tiered_pricing_is_emptied_rather_than_zeroed():
    """A tier outranks the flat rates written beside it, so a zero there would leave the
    cost map's tiers billing the traffic the reserved capacity already covers."""
    override = _with_flag(_VALID, declared={"tiered_pricing": [{"range": [0, 1000], "input_cost_per_token": 0.003}]})

    assert override is not None
    for field in PTU_EMPTIED_PRICING_FIELDS:
        assert override[field] == ()


def test_the_search_context_table_is_zeroed_in_place_on_every_deployment():
    """An absent table means the provider's own default rather than free, so it is written
    even when the deployment never declared one."""
    override = _with_flag(_VALID)

    assert override is not None
    for field in PTU_ZEROED_TABLE_FIELDS:
        assert dict(override[field]) == dict.fromkeys(SEARCH_CONTEXT_SIZES, 0.0)


def test_the_maps_grounding_rate_is_zeroed_on_every_deployment():
    """An absent rate falls back to the Maps default rather than free, so it is written
    even when the deployment never declared one."""
    override = _with_flag(_VALID)

    assert override is not None
    assert override["google_maps_grounding_cost_per_query"] == 0.0


def test_a_declared_table_does_not_become_a_scalar():
    """Zeroing it as a plain 0.0 would leave the provider's reader without a table to
    consult, which is the same as absent."""
    override = _with_flag(_VALID, declared={"search_context_cost_per_query": {"search_context_size_medium": 0.05}})

    assert override is not None
    assert dict(override["search_context_cost_per_query"]) == dict.fromkeys(SEARCH_CONTEXT_SIZES, 0.0)


def test_a_rate_the_deployment_declares_itself_is_zeroed_too():
    """The standing set covers the mirrored rates. Anything else the operator wrote would
    otherwise survive and bill the traffic the hourly charge already paid for."""
    extra = "input_cost_per_token_above_200k_tokens"
    assert extra in CUSTOM_PRICING_FIELDS
    assert extra not in PTU_ZEROED_PRICING_FIELDS

    override = _with_flag(_VALID, declared={extra: 9e-06})

    assert override is not None
    assert override[extra] == 0.0


def test_a_setting_that_is_not_a_charge_is_left_alone():
    """CustomPricingLiteLLMParams also carries configuration, and zeroing one of those
    would break the deployment rather than stop a charge."""
    override = _with_flag(_VALID, declared={"output_vector_size": 1536})

    assert override is not None
    assert "output_vector_size" not in override


# --- the rule both the endpoints and config.yaml registration enforce ---------------


def test_a_complete_reservation_has_no_error():
    assert ptu_config_error(_VALID) is None


def test_a_deployment_with_no_ptu_fields_is_not_a_ptu_deployment():
    """The gate must stay scoped to PTU configuration, or it would reject every ordinary
    deployment for lacking a team_id."""
    assert ptu_config_error({"team_id": "team-alpha"}) is None
    assert ptu_config_error({}) is None


@pytest.mark.parametrize(
    "override, expected",
    [
        ({"team_id": None}, "team_id is required when PTU fields are set (one model maps to one team)"),
        ({"team_id": ""}, "team_id is required when PTU fields are set (one model maps to one team)"),
        ({"cost_per_ptu_per_hour": None}, "ptu_count and cost_per_ptu_per_hour must be set together"),
        ({"ptu_count": None}, "ptu_count and cost_per_ptu_per_hour must be set together"),
        ({"ptu_effective_to": "2025-01-01T00:00:00Z"}, "ptu_effective_to must be after ptu_effective_from"),
    ],
    ids=["no team", "blank team", "count without rate", "rate without count", "inverted window"],
)
def test_an_incoherent_reservation_names_its_reason(override, expected):
    assert ptu_config_error({**_VALID, **override}) == expected


def test_a_missing_start_is_explained_rather_than_inferred():
    error = ptu_config_error({k: v for k, v in _VALID.items() if k != "ptu_effective_from"})

    assert error is not None
    assert error.startswith("ptu_effective_from is required when PTU fields are set")


def test_an_inverted_window_is_caught_before_the_count_and_rate_gate():
    """A patch that moves one end of the window carries no count or rate, so ordering has to
    be checked first or an inverted window reaches the row and the next load cannot parse it."""
    window_only = {
        "ptu_effective_from": "2026-01-01T00:00:00Z",
        "ptu_effective_to": "2025-01-01T00:00:00Z",
    }

    assert ptu_config_error(window_only) == "ptu_effective_to must be after ptu_effective_from"


# --- the identity a config.yaml reservation has to declare ---------------------------


def test_a_declared_unique_id_is_accepted():
    assert ptu_identity_error(declared_id="azure-ptu-eastus", taken=False) is None


@pytest.mark.parametrize("missing", [None, ""], ids=["absent", "blank"])
def test_a_reservation_without_an_id_is_refused(missing):
    error = ptu_identity_error(declared_id=missing, taken=False)

    assert error is not None
    assert error.startswith("model_info.id is required when PTU fields are set")


def test_the_refusal_names_the_id_the_deployment_already_uses():
    """An operator who invents a fresh name starts a second identity beside the charges
    already written, which is the duplicate this rule exists to prevent."""
    error = ptu_identity_error(declared_id=None, taken=False, current_id="0ba149287615")

    assert error is not None
    assert "0ba149287615" in error


def test_the_refusal_points_at_the_model_info_route_when_the_current_id_is_unknown():
    error = ptu_identity_error(declared_id=None, taken=False)

    assert error is not None
    assert "GET /model/info" in error


def test_an_id_declared_twice_is_refused():
    error = ptu_identity_error(declared_id="azure-ptu-eastus", taken=True)

    assert error is not None
    assert "declared on more than one deployment" in error


def test_the_deployment_is_named_when_the_caller_supplies_one():
    error = ptu_identity_error(declared_id=None, taken=False, model_name="azure-ptu")

    assert error is not None
    assert error.startswith("PTU configuration on model 'azure-ptu' is invalid:")


def test_a_bare_yaml_date_bound_is_read_as_that_day_opening():
    """An unquoted 2027-01-01 in config.yaml loads as a date, not a string. Discarding it
    took the whole deployment out of PTU handling, so it billed per token and accrued no
    flat cost while the provider invoiced the reservation hourly."""
    terms = ptu_terms({**_VALID, "ptu_effective_to": date(2027, 1, 1)})

    assert terms is not None
    assert terms.effective_to == datetime(2027, 1, 1, tzinfo=timezone.utc)


def test_a_bare_yaml_date_start_is_read_as_that_day_opening():
    terms = ptu_terms({**_VALID, "ptu_effective_from": date(2026, 5, 1)})

    assert terms is not None
    assert terms.effective_from == datetime(2026, 5, 1, tzinfo=timezone.utc)


def test_the_string_zero_is_a_declared_id():
    """0 is a perfectly stable id, and ModelInfo stores it as a string. Reading it as absent
    refused a deployment whose identity was never in doubt."""
    assert ptu_identity_error(declared_id="0", taken=False) is None


def test_an_empty_id_is_no_id():
    error = ptu_identity_error(declared_id="", taken=False)

    assert error is not None
    assert error.startswith("model_info.id is required")
