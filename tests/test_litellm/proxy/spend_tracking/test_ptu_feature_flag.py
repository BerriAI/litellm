"""Tests for the opt-in flag that gates PTU flat-cost attribution."""

import pytest

from litellm.proxy.spend_tracking.ptu_feature_flag import (
    PTU_COST_ATTRIBUTION_ENV_VAR,
    is_ptu_cost_attribution_enabled,
)


def test_disabled_when_env_var_is_unset(monkeypatch):
    monkeypatch.delenv(PTU_COST_ATTRIBUTION_ENV_VAR, raising=False)
    assert is_ptu_cost_attribution_enabled() is False


@pytest.mark.parametrize("value", ["true", "True", "TRUE", " true "])
def test_enabled_for_the_values_the_house_helper_recognises(monkeypatch, value):
    monkeypatch.setenv(PTU_COST_ATTRIBUTION_ENV_VAR, value)
    assert is_ptu_cost_attribution_enabled() is True


@pytest.mark.parametrize("value", ["false", "False", "0", "1", "", "yes", "off", "maybe"])
def test_disabled_for_everything_else(monkeypatch, value):
    monkeypatch.setenv(PTU_COST_ATTRIBUTION_ENV_VAR, value)
    assert is_ptu_cost_attribution_enabled() is False


def test_reads_the_env_var_on_every_call(monkeypatch):
    monkeypatch.delenv(PTU_COST_ATTRIBUTION_ENV_VAR, raising=False)
    assert is_ptu_cost_attribution_enabled() is False

    monkeypatch.setenv(PTU_COST_ATTRIBUTION_ENV_VAR, "true")
    assert is_ptu_cost_attribution_enabled() is True
