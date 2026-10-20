from pathlib import Path
from typing import Final

from pydantic import BaseModel, TypeAdapter

import litellm
from litellm.integrations.custom_logger import CustomLogger


class DashboardField(BaseModel):
    type: str
    required: bool


class DashboardCallbackConfig(BaseModel):
    id: str
    displayName: str
    logo: str
    supports_key_team_logging: bool
    dynamic_params: dict[str, DashboardField]


def _zerobus_config() -> DashboardCallbackConfig:
    path: Final = Path(litellm.__file__).parent / "integrations" / "callback_configs.json"
    configs: Final = TypeAdapter(tuple[DashboardCallbackConfig, ...]).validate_json(path.read_text())
    return next(config for config in configs if config.id == "zerobus")


def test_zerobus_appears_in_the_dashboard_callback_dropdown():
    """The dropdown is served from callback_configs.json, so an entry only in the dashboard source is invisible."""
    entry = _zerobus_config()

    assert entry.displayName == "Databricks Zerobus"
    assert entry.supports_key_team_logging is False
    assert entry.dynamic_params["ZEROBUS_CLIENT_SECRET"].type == "password"
    assert all(field.required is True for field in entry.dynamic_params.values())


def test_the_dropdown_logo_asset_exists():
    """A logo the dashboard cannot resolve degrades silently to a letter tile."""
    logo = _zerobus_config().logo
    repo_root = Path(litellm.__file__).parent.parent
    asset = repo_root / "ui" / "litellm-dashboard" / "public" / "assets" / "logos" / logo

    assert asset.is_file()


def test_the_dropdown_fields_are_the_env_vars_the_logger_reads():
    """Naming the fields as stored means the edit form prefills saved values instead of showing blanks."""
    fields = tuple(_zerobus_config().dynamic_params)

    assert fields == tuple(CustomLogger.get_callback_env_vars("zerobus"))
