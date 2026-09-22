"""Markerless harness checks for the secret manager backend registry: every
registered backend has a lane config that boots the proxy against that same
backend with the settings the tests assume, so drift fails without a live stack."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
import yaml
from pydantic import BaseModel

from e2e_config import SECRET_MANAGER_OPT_IN_ENV
from secret_backends import BACKENDS, selected_backend
from secret_store import SecretBackend
from test_secret_manager_e2e import VIRTUAL_KEY_PREFIX

E2E_ROOT: Final = Path(__file__).resolve().parent.parent


class KeyManagementSettings(BaseModel):
    access_mode: str
    store_virtual_keys: bool
    prefix_for_stored_virtual_keys: str


class GeneralSettings(BaseModel):
    key_management_system: str
    key_management_settings: KeyManagementSettings


class LaneConfig(BaseModel):
    general_settings: GeneralSettings


def _lane_config(backend: SecretBackend) -> GeneralSettings:
    return LaneConfig.model_validate(yaml.safe_load((E2E_ROOT / backend.proxy_config).read_text())).general_settings


@pytest.mark.parametrize("backend", BACKENDS.values(), ids=BACKENDS.keys())
def test_lane_config_boots_the_proxy_against_its_backend(backend: SecretBackend) -> None:
    settings: Final = _lane_config(backend)

    assert settings.key_management_system == backend.system
    assert settings.key_management_settings.access_mode == "read_and_write"
    assert settings.key_management_settings.store_virtual_keys
    assert settings.key_management_settings.prefix_for_stored_virtual_keys == VIRTUAL_KEY_PREFIX


@pytest.mark.parametrize("system", ["", "not_a_secret_manager"])
def test_unknown_backend_fails_naming_the_known_ones(monkeypatch: pytest.MonkeyPatch, system: str) -> None:
    monkeypatch.setenv(SECRET_MANAGER_OPT_IN_ENV, system)

    with pytest.raises(pytest.fail.Exception, match="hashicorp_vault"):
        selected_backend()


def test_named_backend_is_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SECRET_MANAGER_OPT_IN_ENV, " hashicorp_vault ")

    assert selected_backend() is BACKENDS["hashicorp_vault"]
