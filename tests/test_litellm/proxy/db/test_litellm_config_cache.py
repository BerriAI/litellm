"""
Tests for batch_load_config and batch_load_non_llm_objects.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.db.litellm_config_cache import (
    _NON_LLM_TABLES,
    POLLING_PARAM_NAMES,
    _DBRecord,
    _wrap_records,
    batch_load_config,
    batch_load_non_llm_objects,
)


def _make_config_record(param_name: str, param_value: dict):
    """Helper to create a mock LiteLLM_Config record."""
    return SimpleNamespace(param_name=param_name, param_value=param_value)


def _make_mock_prisma_client(records):
    """Helper to create a mock PrismaClient with find_many returning records."""
    mock_client = MagicMock()
    mock_client.db.litellm_config.find_many = AsyncMock(return_value=records)
    return mock_client


# ─── batch_load_config tests ───


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


# ─── _DBRecord tests ───


def test_db_record_attribute_access():
    """_DBRecord should support attribute access like Prisma models."""
    record = _DBRecord(name="test", value=42)
    assert record.name == "test"
    assert record.value == 42


def test_db_record_model_dump():
    """_DBRecord.model_dump() should return a dict of all fields."""
    record = _DBRecord(name="test", value=42, nested={"a": 1})
    dumped = record.model_dump()
    assert dumped == {"name": "test", "value": 42, "nested": {"a": 1}}


def test_wrap_records():
    """_wrap_records should convert list of dicts to list of _DBRecord."""
    raw = [{"id": "1", "name": "a"}, {"id": "2", "name": "b"}]
    wrapped = _wrap_records(raw)
    assert len(wrapped) == 2
    assert wrapped[0].id == "1"
    assert wrapped[1].name == "b"
    assert wrapped[0].model_dump() == {"id": "1", "name": "a"}


# ─── batch_load_non_llm_objects tests ───


@pytest.mark.asyncio
async def test_batch_load_non_llm_objects_single_query():
    """batch_load_non_llm_objects should call query_raw once."""
    # Simulate query_raw returning one row per table
    raw_rows = [
        {"_tbl": "guardrails", "data": json.dumps([{"guardrail_id": "g1", "name": "test"}])},
        {"_tbl": "policies", "data": json.dumps([])},
        {"_tbl": "agents", "data": json.dumps([{"agent_id": "a1"}])},
    ]
    mock_client = MagicMock()
    mock_client.db.query_raw = AsyncMock(return_value=raw_rows)

    result = await batch_load_non_llm_objects(mock_client)

    mock_client.db.query_raw.assert_called_once()
    assert len(result["guardrails"]) == 1
    assert result["guardrails"][0].guardrail_id == "g1"
    assert result["policies"] == []
    assert result["agents"][0].agent_id == "a1"


@pytest.mark.asyncio
async def test_batch_load_non_llm_objects_all_tables():
    """Result should contain keys for all tables returned by query."""
    raw_rows = [
        {"_tbl": key, "data": json.dumps([])}
        for key, _ in _NON_LLM_TABLES
    ]
    mock_client = MagicMock()
    mock_client.db.query_raw = AsyncMock(return_value=raw_rows)

    result = await batch_load_non_llm_objects(mock_client)

    for key, _ in _NON_LLM_TABLES:
        assert key in result


@pytest.mark.asyncio
async def test_batch_load_non_llm_objects_records_have_model_dump():
    """Records should support model_dump() for Prisma compatibility."""
    raw_rows = [
        {"_tbl": "prompts", "data": json.dumps([{"prompt_id": "p1", "content": "hello"}])},
    ]
    mock_client = MagicMock()
    mock_client.db.query_raw = AsyncMock(return_value=raw_rows)

    result = await batch_load_non_llm_objects(mock_client)

    prompt = result["prompts"][0]
    assert prompt.prompt_id == "p1"
    dumped = prompt.model_dump()
    assert dumped["prompt_id"] == "p1"
    assert dumped["content"] == "hello"
