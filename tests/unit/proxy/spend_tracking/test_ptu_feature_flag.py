"""Tests for the opt-in flag that gates PTU flat-cost attribution."""

from collections.abc import Iterator

import httpx
import pytest

import litellm
from litellm.integrations.custom_secret_manager import CustomSecretManager
from litellm.litellm_core_utils import ptu_pricing
from litellm.proxy.spend_tracking.ptu_feature_flag import (
    PTU_COST_ATTRIBUTION_ENV_VAR,
    is_ptu_cost_attribution_enabled,
)
from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem


class _SecretManagerHolding(CustomSecretManager):
    def __init__(self, secrets: dict[str, str]) -> None:
        super().__init__(secret_manager_name="flag_holder")
        self.secrets = secrets
        self.reads = 0

    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: dict | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> str | None:
        return self.sync_read_secret(secret_name, optional_params, timeout)

    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: dict | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> str | None:
        self.reads += 1
        return self.secrets.get(secret_name)


@pytest.fixture
def secret_manager(monkeypatch: pytest.MonkeyPatch) -> Iterator[_SecretManagerHolding]:
    manager = _SecretManagerHolding({PTU_COST_ATTRIBUTION_ENV_VAR: "true"})
    monkeypatch.delenv(PTU_COST_ATTRIBUTION_ENV_VAR, raising=False)
    monkeypatch.setattr(litellm, "secret_manager_client", manager)
    monkeypatch.setattr(litellm, "_key_management_system", KeyManagementSystem.CUSTOM)
    monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings(access_mode="read_only"))
    ptu_pricing._enabled_in_secret_manager.cache_clear()
    yield manager
    ptu_pricing._enabled_in_secret_manager.cache_clear()


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


def test_an_env_value_wins_without_asking_the_secret_manager(
    monkeypatch: pytest.MonkeyPatch, secret_manager: _SecretManagerHolding
) -> None:
    monkeypatch.setenv(PTU_COST_ATTRIBUTION_ENV_VAR, "false")
    assert is_ptu_cost_attribution_enabled() is False
    assert secret_manager.reads == 0


def test_a_flag_kept_in_a_secret_manager_still_turns_attribution_on(secret_manager: _SecretManagerHolding) -> None:
    assert is_ptu_cost_attribution_enabled() is True


def test_the_secret_manager_is_asked_once_for_many_requests(secret_manager: _SecretManagerHolding) -> None:
    assert all(is_ptu_cost_attribution_enabled() for _ in range(50))
    assert secret_manager.reads == 1


def test_a_secret_manager_without_the_flag_leaves_it_off(secret_manager: _SecretManagerHolding) -> None:
    secret_manager.secrets.clear()
    assert is_ptu_cost_attribution_enabled() is False
