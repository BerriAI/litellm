"""Pin the LiteLLM_Config cached-read layer.

Symbols pinned here:
  - ``_ConfigRow``
  - ``_config_cache_key``
  - ``_pack_config_row``
  - ``_unpack_config_row``
  - ``get_config_param``
  - ``invalidate_config_param``
  - ``prefetch_config_params``
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm.proxy.utils as utils_mod
from litellm.proxy.utils import (
    _config_cache_key,
    _ConfigRow,
    _pack_config_row,
    _unpack_config_row,
    get_config_param,
    invalidate_config_param,
    prefetch_config_params,
)


@pytest.fixture(autouse=True)
def _swap_config_cache(
    monkeypatch: pytest.MonkeyPatch, mock_dual_cache: Any
) -> Any:
    """Replace the module-level cache so tests see a clean store per run."""
    monkeypatch.setattr(utils_mod, "litellm_config_cache", mock_dual_cache)
    return mock_dual_cache


def test_config_cache_key_uses_documented_prefix() -> None:
    actual = {
        "key": _config_cache_key("max_budget"),
        "another": _config_cache_key("disable_spend_updates"),
        "prefix": _config_cache_key("x").split(":")[0],
    }
    assert actual == {
        "key": "litellm_config:param:max_budget",
        "another": "litellm_config:param:disable_spend_updates",
        "prefix": "litellm_config",
    }


def test_config_cache_key_error_propagates_from_bad_format() -> None:
    class _Boom:
        def __format__(self, _spec: str) -> str:
            raise ValueError("format failure")

    with pytest.raises(ValueError, match="format failure"):
        _config_cache_key(_Boom())  # type: ignore[arg-type]


def test_config_row_dataclass_shape() -> None:
    row = _ConfigRow(param_name="alpha", param_value={"k": 1})
    assert {
        "param_name": row.param_name,
        "param_value": row.param_value,
        "slots": _ConfigRow.__slots__,
    } == {
        "param_name": "alpha",
        "param_value": {"k": 1},
        "slots": ("param_name", "param_value"),
    }


def test_config_row_rejects_unknown_attribute() -> None:
    row = _ConfigRow("a", 1)
    with pytest.raises(AttributeError):
        row.something_else = 2  # type: ignore[attr-defined]


def test_pack_config_row_returns_dict_for_caching() -> None:
    row = SimpleNamespace(param_name="zeta", param_value=[1, 2, 3])
    actual = _pack_config_row(row)
    expanded = {**actual, "is_dict": isinstance(actual, dict)}
    assert expanded == {
        "param_name": "zeta",
        "param_value": [1, 2, 3],
        "is_dict": True,
    }


def test_pack_config_row_error_on_missing_attribute() -> None:
    bad = SimpleNamespace(param_name="only_name")
    with pytest.raises(AttributeError):
        _pack_config_row(bad)


def test_unpack_config_row_round_trips_dict() -> None:
    packed = {"param_name": "alpha", "param_value": "abc"}
    unpacked = _unpack_config_row(packed)
    assert isinstance(unpacked, _ConfigRow)
    actual = {
        "param_name": unpacked.param_name,
        "param_value": unpacked.param_value,
        "from_none": _unpack_config_row(None),
        "from_miss_sentinel": _unpack_config_row(utils_mod._CONFIG_CACHE_MISS),
        "from_other_type": _unpack_config_row(123),
    }
    assert actual == {
        "param_name": "alpha",
        "param_value": "abc",
        "from_none": None,
        "from_miss_sentinel": None,
        "from_other_type": None,
    }


def test_unpack_config_row_error_on_malformed_dict() -> None:
    with pytest.raises(KeyError):
        _unpack_config_row({"only_name": "x"})


@pytest.mark.asyncio
async def test_get_config_param_cache_hit_returns_unpacked_row(
    _swap_config_cache: Any,
) -> None:
    cache_key = _config_cache_key("p1")
    await _swap_config_cache.async_set_cache(
        cache_key, {"param_name": "p1", "param_value": {"x": 1}}
    )
    prisma = MagicMock()
    prisma.get_generic_data = AsyncMock()

    row = await get_config_param(prisma, "p1")
    actual = {
        "type": type(row).__name__,
        "param_name": row.param_name,
        "param_value": row.param_value,
        "db_not_touched": prisma.get_generic_data.await_count == 0,
    }
    assert actual == {
        "type": "_ConfigRow",
        "param_name": "p1",
        "param_value": {"x": 1},
        "db_not_touched": True,
    }


@pytest.mark.asyncio
async def test_get_config_param_cache_miss_fetches_from_db_and_caches(
    _swap_config_cache: Any,
) -> None:
    db_row = SimpleNamespace(param_name="p2", param_value={"y": 2})
    prisma = MagicMock()
    prisma.get_generic_data = AsyncMock(return_value=db_row)

    row = await get_config_param(prisma, "p2")
    cached = _swap_config_cache._store[_config_cache_key("p2")]
    actual = {
        "returned": row,
        "cached": cached,
        "db_called": prisma.get_generic_data.await_count,
        "db_args": prisma.get_generic_data.await_args.kwargs,
    }
    assert actual == {
        "returned": db_row,
        "cached": {"param_name": "p2", "param_value": {"y": 2}},
        "db_called": 1,
        "db_args": {"key": "param_name", "value": "p2", "table_name": "config"},
    }


@pytest.mark.asyncio
async def test_get_config_param_caches_negative_lookup_as_miss_sentinel(
    _swap_config_cache: Any,
) -> None:
    prisma = MagicMock()
    prisma.get_generic_data = AsyncMock(return_value=None)
    row = await get_config_param(prisma, "absent")
    assert row is None
    assert _swap_config_cache._store[_config_cache_key("absent")] == (
        utils_mod._CONFIG_CACHE_MISS
    )


@pytest.mark.asyncio
async def test_get_config_param_raises_when_db_raises() -> None:
    prisma = MagicMock()
    prisma.get_generic_data = AsyncMock(side_effect=RuntimeError("db down"))
    with pytest.raises(RuntimeError, match="db down"):
        await get_config_param(prisma, "p3")


@pytest.mark.asyncio
async def test_invalidate_config_param_evicts_from_cache(
    _swap_config_cache: Any,
) -> None:
    cache_key = _config_cache_key("p4")
    await _swap_config_cache.async_set_cache(cache_key, {"param_name": "p4", "param_value": 1})
    await invalidate_config_param("p4")
    actual = {
        "store_empty": _swap_config_cache._store == {},
        "delete_calls": _swap_config_cache.async_delete_cache.await_count,
        "delete_arg": _swap_config_cache.async_delete_cache.await_args.args[0],
    }
    assert actual == {
        "store_empty": True,
        "delete_calls": 1,
        "delete_arg": "litellm_config:param:p4",
    }


@pytest.mark.asyncio
async def test_invalidate_config_param_propagates_cache_error(
    _swap_config_cache: Any,
) -> None:
    _swap_config_cache.async_delete_cache = AsyncMock(
        side_effect=ConnectionError("redis down")
    )
    with pytest.raises(ConnectionError):
        await invalidate_config_param("p5")


@pytest.mark.asyncio
async def test_prefetch_config_params_populates_cache_for_each_name(
    _swap_config_cache: Any,
) -> None:
    rows: List[SimpleNamespace] = [
        SimpleNamespace(param_name="a", param_value={"av": 1}),
        SimpleNamespace(param_name="c", param_value=[3]),
    ]
    prisma = MagicMock()
    prisma.db.litellm_config.find_many = AsyncMock(return_value=rows)
    await prefetch_config_params(prisma, ["a", "b", "c"])
    actual = {
        "a": _swap_config_cache._store[_config_cache_key("a")],
        "b": _swap_config_cache._store[_config_cache_key("b")],
        "c": _swap_config_cache._store[_config_cache_key("c")],
    }
    assert actual == {
        "a": {"param_name": "a", "param_value": {"av": 1}},
        "b": utils_mod._CONFIG_CACHE_MISS,
        "c": {"param_name": "c", "param_value": [3]},
    }


@pytest.mark.asyncio
async def test_prefetch_config_params_empty_list_is_noop(
    _swap_config_cache: Any,
) -> None:
    prisma = MagicMock()
    prisma.db.litellm_config.find_many = AsyncMock(return_value=[])
    await prefetch_config_params(prisma, [])
    assert prisma.db.litellm_config.find_many.await_count == 0
    assert _swap_config_cache._store == {}


@pytest.mark.asyncio
async def test_prefetch_config_params_swallows_db_error_without_caching(
    _swap_config_cache: Any,
) -> None:
    prisma = MagicMock()
    prisma.db.litellm_config.find_many = AsyncMock(side_effect=RuntimeError("boom"))
    await prefetch_config_params(prisma, ["a", "b"])
    assert _swap_config_cache._store == {}
