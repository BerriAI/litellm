"""
Tests for batch_load_config - batching LiteLLM_Config queries into one find_many.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.db.litellm_config_cache import (
    POLLING_PARAM_NAMES,
    batch_load_config,
)


def _make_config_record(param_name: str, param_value: dict):
    """Helper to create a mock LiteLLM_Config record."""
    return SimpleNamespace(param_name=param_name, param_value=param_value)


def _make_mock_prisma_client(records):
    """Helper to create a mock PrismaClient with find_many returning records."""
    mock_client = MagicMock()
    mock_client.db.litellm_config.find_many = AsyncMock(return_value=records)
    return mock_client


@pytest.mark.asyncio
async def test_batch_load_fetches_all_config_in_one_query():
    """batch_load_config should call find_many once with all param names."""
    records = [
        _make_config_record("general_settings", {"key": "value"}),
        _make_config_record("litellm_settings", {"setting": True}),
    ]
    mock_client = _make_mock_prisma_client(records)

    result = await batch_load_config(mock_client)

    mock_client.db.litellm_config.find_many.assert_called_once_with(
        where={"param_name": {"in": POLLING_PARAM_NAMES}}
    )
    assert len(result) == 2


@pytest.mark.asyncio
async def test_batch_load_returns_dict_keyed_by_param_name():
    """Result should be a dict keyed by param_name."""
    records = [
        _make_config_record("general_settings", {"alerting": ["slack"]}),
        _make_config_record("model_cost_map_reload_config", {"interval_hours": 24}),
    ]
    mock_client = _make_mock_prisma_client(records)

    result = await batch_load_config(mock_client)

    assert result["general_settings"].param_value == {"alerting": ["slack"]}
    assert result["model_cost_map_reload_config"].param_value == {"interval_hours": 24}


@pytest.mark.asyncio
async def test_batch_load_returns_empty_dict_for_no_records():
    """If no config records exist, should return empty dict."""
    mock_client = _make_mock_prisma_client([])

    result = await batch_load_config(mock_client)

    assert result == {}


@pytest.mark.asyncio
async def test_batch_load_missing_key_returns_none_on_get():
    """dict.get() on a missing key should return None."""
    records = [
        _make_config_record("general_settings", {"key": "value"}),
    ]
    mock_client = _make_mock_prisma_client(records)

    result = await batch_load_config(mock_client)

    assert result.get("nonexistent_key") is None
    assert result.get("general_settings") is not None


@pytest.mark.asyncio
async def test_batch_load_contains_all_polling_param_names_when_all_exist():
    """When all param_names exist in DB, all should be in the result."""
    records = [_make_config_record(name, {"data": name}) for name in POLLING_PARAM_NAMES]
    mock_client = _make_mock_prisma_client(records)

    result = await batch_load_config(mock_client)

    for name in POLLING_PARAM_NAMES:
        assert name in result
        assert result[name].param_value == {"data": name}
