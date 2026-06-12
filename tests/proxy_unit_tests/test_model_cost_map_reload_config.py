"""
Tests for model_cost_map_reload_config configuration via config.yaml.

This tests that:
1. The proxy can read model_cost_map_reload_config from config.yaml (general_settings)
2. Database config takes precedence over YAML config (runtime overrides)
3. A warning is logged when DB config overrides YAML config
4. force_reload from YAML is ignored (only works via API)
5. When config comes from YAML, no DB write happens after reload
6. When config comes from DB, DB write happens to clear force_reload
"""

import os
import sys
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath("../.."))

import litellm.proxy.proxy_server as proxy_server
from litellm.proxy.proxy_server import ProxyConfig


class FakeConfigRecord:
    """Mock Prisma config record."""

    def __init__(self, param_value):
        self.param_value = param_value


class FakeConfigRepository:
    """Mock Prisma ConfigRepository."""

    def __init__(self, prisma_client=None):
        self.table = MagicMock()
        self.table.upsert = AsyncMock()
        self.table.delete = AsyncMock()
        self.table.find_unique = AsyncMock()


@pytest.fixture(autouse=True)
def reset_global_flags():
    """Reset module-level flags between tests to prevent interference."""
    proxy_server._yaml_override_warned = False
    yield


@pytest.mark.asyncio
async def test_get_model_cost_map_reload_config_returns_tuple():
    """The helper should return (config, source) as a tuple."""
    proxy_config = ProxyConfig()
    proxy_server.general_settings = {}
    mock_prisma = MagicMock()

    with patch(
        "litellm.proxy.proxy_server.get_config_param",
        new=AsyncMock(return_value=None),
    ):
        config, source = await proxy_config._get_model_cost_map_reload_config(
            mock_prisma
        )

    assert config is None
    assert source is None


@pytest.mark.asyncio
async def test_get_model_cost_map_reload_config_from_db_takes_precedence():
    """
    When both DB and YAML config exist, DB config should take precedence.
    A warning should be logged that YAML is being ignored.
    """
    proxy_config = ProxyConfig()

    # Set up YAML config (lower precedence)
    proxy_server.general_settings = {
        "model_cost_map_reload_config": {
            "interval_hours": 48,
            "force_reload": False,
        }
    }

    # Set up DB config (higher precedence)
    db_config = {"interval_hours": 24, "force_reload": True}
    mock_prisma = MagicMock()

    with patch(
        "litellm.proxy.proxy_server.get_config_param",
        new=AsyncMock(return_value=FakeConfigRecord(db_config)),
    ):
        with patch.object(proxy_server.verbose_proxy_logger, "warning") as mock_warning:
            config, source = await proxy_config._get_model_cost_map_reload_config(
                mock_prisma
            )

    assert config is not None
    assert config["interval_hours"] == 24  # DB value, not YAML's 48
    assert config["force_reload"] is True
    assert source == "db"
    # Warning should be logged that YAML is being ignored
    mock_warning.assert_called_once()
    assert "database" in mock_warning.call_args[0][0]
    assert "config.yaml" in mock_warning.call_args[0][0]


@pytest.mark.asyncio
async def test_get_model_cost_map_reload_config_from_yaml_when_db_empty():
    """
    When DB config is None, YAML config should be used as fallback.
    """
    proxy_config = ProxyConfig()

    yaml_config = {"interval_hours": 12, "force_reload": False}
    proxy_server.general_settings = {"model_cost_map_reload_config": yaml_config}

    mock_prisma = MagicMock()

    with patch(
        "litellm.proxy.proxy_server.get_config_param",
        new=AsyncMock(return_value=None),
    ):
        config, source = await proxy_config._get_model_cost_map_reload_config(
            mock_prisma
        )

    assert config is not None
    assert config["interval_hours"] == 12
    assert config["force_reload"] is False
    assert source == "yaml"


@pytest.mark.asyncio
async def test_get_model_cost_map_reload_config_none_when_both_empty():
    """
    When neither DB nor YAML config is set, should return (None, None).
    """
    proxy_config = ProxyConfig()
    proxy_server.general_settings = {}  # No YAML config

    mock_prisma = MagicMock()

    with patch(
        "litellm.proxy.proxy_server.get_config_param",
        new=AsyncMock(return_value=None),
    ):
        config, source = await proxy_config._get_model_cost_map_reload_config(
            mock_prisma
        )

    assert config is None
    assert source is None


@pytest.mark.asyncio
async def test_get_model_cost_map_reload_config_yaml_string_value():
    """
    YAML config value can be a JSON string that needs parsing.
    """
    proxy_config = ProxyConfig()

    yaml_config = json.dumps({"interval_hours": 6, "force_reload": False})
    proxy_server.general_settings = {"model_cost_map_reload_config": yaml_config}

    mock_prisma = MagicMock()

    with patch(
        "litellm.proxy.proxy_server.get_config_param",
        new=AsyncMock(return_value=None),
    ):
        config, source = await proxy_config._get_model_cost_map_reload_config(
            mock_prisma
        )

    assert config is not None
    assert config["interval_hours"] == 6
    assert source == "yaml"


@pytest.mark.asyncio
async def test_check_and_reload_no_db_write_when_config_from_yaml():
    """
    When reload is triggered by YAML config, the method should NOT try to
    write back to the database (since there's no DB record to update).
    """
    proxy_config = ProxyConfig()
    proxy_server.general_settings = {
        "model_cost_map_reload_config": {
            "interval_hours": 1,  # 1 hour, will trigger since no last reload
            "force_reload": False,
        }
    }
    proxy_server.last_model_cost_map_reload = None  # No previous reload

    mock_prisma = MagicMock()
    fake_repo = FakeConfigRepository(mock_prisma)

    with patch(
        "litellm.proxy.proxy_server.get_config_param",
        new=AsyncMock(return_value=None),  # No DB record at all
    ):
        with patch(
            "litellm.proxy.proxy_server.ConfigRepository",
            return_value=fake_repo,
        ):
            with patch(
                "litellm.litellm_core_utils.get_model_cost_map.get_model_cost_map",
                return_value={"gpt-4o": {"input_cost_per_token": 0.001}},
            ):
                with patch.object(proxy_server, "_invalidate_model_cost_lowercase_map"):
                    with patch("litellm.proxy.proxy_server.litellm.add_known_models"):
                        await proxy_config._check_and_reload_model_cost_map(mock_prisma)

    # upsert should NOT have been called since config came from YAML
    fake_repo.table.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_check_and_reload_ignores_force_reload_from_yaml():
    """
    force_reload: true in YAML should be ignored with a warning, to prevent
    infinite reloads every 10 seconds (the flag is never cleared in YAML).
    """
    proxy_config = ProxyConfig()
    proxy_server.general_settings = {
        "model_cost_map_reload_config": {
            "interval_hours": None,  # No interval, only force_reload
            "force_reload": True,
        }
    }
    proxy_server.last_model_cost_map_reload = None

    mock_prisma = MagicMock()

    with patch(
        "litellm.proxy.proxy_server.get_config_param",
        new=AsyncMock(return_value=None),
    ):
        with patch.object(proxy_server.verbose_proxy_logger, "warning") as mock_warning:
            with patch(
                "litellm.litellm_core_utils.get_model_cost_map.get_model_cost_map",
            ) as mock_get_cost_map:
                with patch.object(proxy_server, "_invalidate_model_cost_lowercase_map"):
                    with patch("litellm.proxy.proxy_server.litellm.add_known_models"):
                        await proxy_config._check_and_reload_model_cost_map(mock_prisma)

    # Warning should be logged about force_reload being ignored
    mock_warning.assert_called_once()
    assert "force_reload" in mock_warning.call_args[0][0]
    assert "config.yaml" in mock_warning.call_args[0][0]

    # No reload should have happened because interval is None and force_reload
    # from YAML is ignored
    mock_get_cost_map.assert_not_called()


@pytest.mark.asyncio
async def test_check_and_reload_db_write_when_config_from_db():
    """
    When reload is triggered by DB config, the method SHOULD write back to
    the database to clear the force_reload flag.
    """
    proxy_config = ProxyConfig()
    proxy_server.general_settings = {}  # No YAML config
    proxy_server.last_model_cost_map_reload = None

    db_config = {"interval_hours": 1, "force_reload": True}
    mock_prisma = MagicMock()
    fake_repo = FakeConfigRepository(mock_prisma)

    with patch(
        "litellm.proxy.proxy_server.get_config_param",
        new=AsyncMock(return_value=FakeConfigRecord(db_config)),
    ):
        with patch(
            "litellm.proxy.proxy_server.ConfigRepository",
            return_value=fake_repo,
        ):
            with patch(
                "litellm.litellm_core_utils.get_model_cost_map.get_model_cost_map",
                return_value={"gpt-4o": {"input_cost_per_token": 0.001}},
            ):
                with patch.object(proxy_server, "_invalidate_model_cost_lowercase_map"):
                    with patch("litellm.proxy.proxy_server.litellm.add_known_models"):
                        await proxy_config._check_and_reload_model_cost_map(mock_prisma)

    # upsert SHOULD have been called to clear force_reload in the DB
    fake_repo.table.upsert.assert_awaited_once()
