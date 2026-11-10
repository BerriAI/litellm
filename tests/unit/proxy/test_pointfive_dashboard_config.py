import json
from pathlib import Path

import litellm
from litellm.integrations.custom_logger import CustomLogger


def _dashboard_configs() -> tuple[dict, ...]:
    path = Path(litellm.__file__).parent / "integrations" / "callback_configs.json"
    return tuple(json.loads(path.read_text()))


def _pointfive_config() -> dict:
    return next(config for config in _dashboard_configs() if config["id"] == "pointfive")


def test_pointfive_appears_in_the_dashboard_callback_dropdown():
    """The dropdown is served from callback_configs.json, so an entry only in the dashboard source is invisible."""
    entry = _pointfive_config()

    assert entry["displayName"] == "PointFive"
    assert entry["supports_key_team_logging"] is False
    assert entry["dynamic_params"]["POINTFIVE_API_KEY"]["required"] is True
    assert entry["dynamic_params"]["POINTFIVE_API_KEY"]["type"] == "password"
    assert entry["dynamic_params"]["POINTFIVE_API_URL"]["required"] is False


def test_the_dropdown_logo_asset_exists():
    """A logo the dashboard cannot resolve degrades silently to a letter tile."""
    logo = _pointfive_config()["logo"]
    repo_root = Path(litellm.__file__).parent.parent
    asset = repo_root / "ui" / "litellm-dashboard" / "public" / "assets" / "logos" / logo

    assert logo == "pointfive.png"
    assert asset.is_file()


def test_the_dropdown_fields_are_the_env_vars_the_logger_reads():
    """
    The field names are the environment variables verbatim.

    The proxy would uppercase them either way, but naming them as stored means the edit
    form finds the saved values and prefills them instead of showing blanks.
    """
    fields = tuple(_pointfive_config()["dynamic_params"])

    assert fields == tuple(CustomLogger.get_callback_env_vars("pointfive"))
