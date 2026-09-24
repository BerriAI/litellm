"""Tests for the Azure PTU sizing table and the conversions built on it."""

from typing import Final

import pytest

from litellm.litellm_core_utils.azure_ptu_capacity import (
    AZURE_PTU_CAPACITY,
    PTUCapacity,
    azure_ptu_capacity,
    deployment_ptu_capacity,
    normalized_tokens,
    ptu_hours,
)

_GPT41: Final = AZURE_PTU_CAPACITY["gpt-4.1"]
_ROW: Final = PTUCapacity(input_tpm_per_ptu=1_000, output_to_input_ratio=4.0)
_CACHED_ROW: Final = PTUCapacity(input_tpm_per_ptu=1_000, output_to_input_ratio=5.0, cached_input_ratio=0.1)


@pytest.mark.parametrize(
    "model",
    ["gpt-4.1", "azure/gpt-4.1", "azure/gpt-4.1-2025-04-14", "GPT-4.1", "azure/eastus/gpt-4.1-2025-04-14"],
)
def test_a_model_resolves_to_its_row_through_a_provider_prefix_and_a_dated_version(model):
    assert azure_ptu_capacity(model) is _GPT41


def test_a_deployment_name_that_is_not_a_model_has_no_row():
    assert azure_ptu_capacity("azure/my-ptu-deployment") is None
    assert azure_ptu_capacity("") is None


def test_a_dated_version_only_drops_a_full_date_suffix():
    """``gpt-4o-mini`` must not lose its ``-mini`` the way ``-2024-07-18`` is dropped."""
    assert azure_ptu_capacity("azure/gpt-4o-mini-2024-07-18") is AZURE_PTU_CAPACITY["gpt-4o-mini"]
    assert azure_ptu_capacity("gpt-4o-mini") is not AZURE_PTU_CAPACITY["gpt-4o"]


def test_every_row_serves_its_input_tpm_for_an_hour():
    for capacity in AZURE_PTU_CAPACITY.values():
        assert capacity.normalized_tokens_per_ptu_hour == capacity.input_tpm_per_ptu * 60
        assert capacity.input_tpm_per_ptu > 0
        assert capacity.output_to_input_ratio >= 1.0
        assert 0.0 <= capacity.cached_input_ratio < 1.0


def test_a_deployment_prefers_its_declared_base_model_over_its_deployment_name():
    deployment: Final = {
        "model_info": {"base_model": "azure/gpt-4.1"},
        "litellm_params": {"model": "azure/gpt-4o"},
    }
    assert deployment_ptu_capacity(deployment) is _GPT41


def test_a_deployment_falls_back_to_its_litellm_model_when_no_base_model_is_declared():
    assert deployment_ptu_capacity({"litellm_params": {"model": "azure/gpt-4o"}}) is AZURE_PTU_CAPACITY["gpt-4o"]
    assert deployment_ptu_capacity({"model_info": {"base_model": ""}, "litellm_params": {"model": "azure/gpt-4o"}}) is (
        AZURE_PTU_CAPACITY["gpt-4o"]
    )


def test_a_deployment_with_no_recognisable_model_has_no_row():
    assert deployment_ptu_capacity({"litellm_params": {"model": "azure/team-a-ptu"}}) is None
    assert deployment_ptu_capacity({"model_info": None, "litellm_params": None}) is None
    assert deployment_ptu_capacity({}) is None


def test_output_is_weighted_by_the_models_ratio_and_uncached_input_counts_in_full():
    assert normalized_tokens(_ROW, prompt_tokens=100, completion_tokens=10) == pytest.approx(140.0)


def test_cached_input_is_free_unless_the_row_prices_it():
    assert normalized_tokens(_ROW, prompt_tokens=100, completion_tokens=0, cache_read_tokens=60) == pytest.approx(40.0)
    assert normalized_tokens(_CACHED_ROW, prompt_tokens=100, completion_tokens=0, cache_read_tokens=60) == pytest.approx(
        46.0
    )


def test_cached_input_never_exceeds_the_prompt_and_negatives_count_as_zero():
    assert normalized_tokens(_ROW, prompt_tokens=100, completion_tokens=0, cache_read_tokens=250) == 0.0
    assert normalized_tokens(_ROW, prompt_tokens=-5, completion_tokens=-5, cache_read_tokens=-5) == 0.0


def test_one_ptu_hour_is_one_ptus_input_tpm_served_for_sixty_minutes():
    assert ptu_hours(_ROW, _ROW.input_tpm_per_ptu * 60) == pytest.approx(1.0)
    assert ptu_hours(_ROW, _ROW.input_tpm_per_ptu * 30) == pytest.approx(0.5)
    assert ptu_hours(_ROW, 0.0) == 0.0
