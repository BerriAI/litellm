import datetime as real_datetime
import json
import os
import sys

import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.proxy._types import ProxyErrorTypes
from litellm.proxy.utils import ProxyLogging

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


from unittest.mock import MagicMock

from litellm.proxy.utils import get_custom_url, join_paths


def test_get_custom_url(monkeypatch):
    monkeypatch.setenv("SERVER_ROOT_PATH", "/litellm")
    custom_url = get_custom_url(request_base_url="http://0.0.0.0:4000", route="ui/")
    assert custom_url == "http://0.0.0.0:4000/litellm/ui/"


def test_proxy_only_error_true_for_llm_route():
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    assert proxy_logging_obj._is_proxy_only_llm_api_error(
        original_exception=Exception(),
        error_type=ProxyErrorTypes.auth_error,
        route="/v1/chat/completions",
    )


def test_proxy_only_error_true_for_info_route():
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    assert (
        proxy_logging_obj._is_proxy_only_llm_api_error(
            original_exception=Exception(),
            error_type=ProxyErrorTypes.auth_error,
            route="/key/info",
        )
        is True
    )


def test_proxy_only_error_false_for_non_llm_non_info_route():
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    assert (
        proxy_logging_obj._is_proxy_only_llm_api_error(
            original_exception=Exception(),
            error_type=ProxyErrorTypes.auth_error,
            route="/key/generate",
        )
        is False
    )


def test_proxy_only_error_false_for_other_error_type():
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    assert (
        proxy_logging_obj._is_proxy_only_llm_api_error(
            original_exception=Exception(),
            error_type=None,
            route="/v1/chat/completions",
        )
        is False
    )


def test_get_model_group_info_order():
    from litellm import Router
    from litellm.proxy.proxy_server import _get_model_group_info

    router = Router(
        model_list=[
            {
                "model_name": "openai/tts-1",
                "litellm_params": {
                    "model": "openai/tts-1",
                    "api_key": "sk-1234",
                },
            },
            {
                "model_name": "openai/gpt-3.5-turbo",
                "litellm_params": {
                    "model": "openai/gpt-3.5-turbo",
                    "api_key": "sk-1234",
                },
            },
        ]
    )
    model_list = _get_model_group_info(
        llm_router=router,
        all_models_str=["openai/tts-1", "openai/gpt-3.5-turbo"],
        model_group=None,
    )

    model_groups = [m.model_group for m in model_list]
    assert model_groups == ["openai/tts-1", "openai/gpt-3.5-turbo"]


def test_join_paths_no_duplication():
    """Test that join_paths doesn't duplicate route when base_path already ends with it"""
    result = join_paths(
        base_path="http://0.0.0.0:4000/my-custom-path/", route="/my-custom-path"
    )
    assert result == "http://0.0.0.0:4000/my-custom-path"


def test_join_paths_normal_join():
    """Test normal path joining"""
    result = join_paths(base_path="http://0.0.0.0:4000", route="/api/v1")
    assert result == "http://0.0.0.0:4000/api/v1"


def test_join_paths_with_trailing_slash():
    """Test path joining with trailing slash on base_path"""
    result = join_paths(base_path="http://0.0.0.0:4000/", route="api/v1")
    assert result == "http://0.0.0.0:4000/api/v1"


def test_join_paths_empty_base():
    """Test path joining with empty base_path"""
    result = join_paths(base_path="", route="api/v1")
    assert result == "/api/v1"


def test_join_paths_empty_route():
    """Test path joining with empty route"""
    result = join_paths(base_path="http://0.0.0.0:4000", route="")
    assert result == "http://0.0.0.0:4000"


def test_join_paths_both_empty():
    """Test path joining with both empty"""
    result = join_paths(base_path="", route="")
    assert result == "/"


def test_join_paths_nested_path():
    """Test path joining with nested paths"""
    result = join_paths(base_path="http://0.0.0.0:4000/v1", route="chat/completions")
    assert result == "http://0.0.0.0:4000/v1/chat/completions"


def _patch_today(monkeypatch, year, month, day):
    class PatchedDate(real_datetime.date):
        @classmethod
        def today(cls):
            return real_datetime.date(year, month, day)

    monkeypatch.setattr("litellm.proxy.utils.date", PatchedDate)


def test_get_projected_spend_over_limit_day_one(monkeypatch):
    from litellm.proxy.utils import _get_projected_spend_over_limit

    _patch_today(monkeypatch, 2026, 1, 1)
    result = _get_projected_spend_over_limit(100.0, 1.0)

    assert result is not None
    projected_spend, projected_exceeded_date = result
    assert projected_spend == 3100.0
    assert projected_exceeded_date == real_datetime.date(2026, 1, 1)


def test_get_projected_spend_over_limit_december(monkeypatch):
    from litellm.proxy.utils import _get_projected_spend_over_limit

    _patch_today(monkeypatch, 2026, 12, 15)
    result = _get_projected_spend_over_limit(100.0, 1.0)

    assert result is not None
    projected_spend, projected_exceeded_date = result
    assert projected_spend == pytest.approx(214.28571428571428)
    assert projected_exceeded_date == real_datetime.date(2026, 12, 15)


def test_get_projected_spend_over_limit_includes_current_spend(monkeypatch):
    from litellm.proxy.utils import _get_projected_spend_over_limit

    _patch_today(monkeypatch, 2026, 4, 11)
    result = _get_projected_spend_over_limit(100.0, 200.0)

    assert result is not None
    projected_spend, projected_exceeded_date = result
    assert projected_spend == 290.0
    assert projected_exceeded_date == real_datetime.date(2026, 4, 21)


@pytest.mark.asyncio
async def test_get_generic_data_retries_on_transport_error():
    """
    Test that get_generic_data retries once after a successful DB reconnect
    when a transport error (e.g. httpx.ReadError) occurs.
    """
    import httpx
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.utils import PrismaClient

    prisma_client = PrismaClient(
        database_url="postgresql://user:pass@localhost:5432/db",
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
    )

    mock_db = MagicMock()
    # First call raises a transport error, second call succeeds
    fake_result = MagicMock()
    fake_result.param_name = "general_settings"
    mock_find_first = AsyncMock(
        side_effect=[httpx.ReadError("connection reset"), fake_result]
    )
    mock_db.litellm_config.find_first = mock_find_first
    prisma_client.db = mock_db

    # Mock attempt_db_reconnect to succeed
    prisma_client.attempt_db_reconnect = AsyncMock(return_value=True)

    result = await prisma_client.get_generic_data(
        key="param_name",
        value="general_settings",
        table_name="config",
    )

    assert result == fake_result
    assert mock_find_first.call_count == 2
    prisma_client.attempt_db_reconnect.assert_awaited_once_with(
        reason="get_generic_data_transport_error",
    )


@pytest.mark.asyncio
async def test_get_generic_data_no_retry_on_non_transport_error():
    """
    Test that get_generic_data does NOT retry on non-transport errors
    (e.g. a PrismaError for invalid query).
    """
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.utils import PrismaClient

    prisma_client = PrismaClient(
        database_url="postgresql://user:pass@localhost:5432/db",
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
    )

    mock_db = MagicMock()
    mock_find_first = AsyncMock(side_effect=ValueError("some non-transport error"))
    mock_db.litellm_config.find_first = mock_find_first
    prisma_client.db = mock_db

    prisma_client.attempt_db_reconnect = AsyncMock(return_value=True)

    with pytest.raises(ValueError, match="some non-transport error"):
        await prisma_client.get_generic_data(
            key="param_name",
            value="general_settings",
            table_name="config",
        )

    # Should NOT have attempted reconnect for a non-transport error
    assert mock_find_first.call_count == 1
    prisma_client.attempt_db_reconnect.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_generic_data_raises_after_failed_reconnect():
    """
    Test that get_generic_data raises the original error when reconnect fails.
    """
    import httpx
    from unittest.mock import AsyncMock

    from litellm.proxy.utils import PrismaClient

    prisma_client = PrismaClient(
        database_url="postgresql://user:pass@localhost:5432/db",
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
    )

    mock_db = MagicMock()
    mock_find_first = AsyncMock(side_effect=httpx.ReadError("connection reset"))
    mock_db.litellm_config.find_first = mock_find_first
    prisma_client.db = mock_db

    # Mock attempt_db_reconnect to fail
    prisma_client.attempt_db_reconnect = AsyncMock(return_value=False)

    with pytest.raises(httpx.ReadError):
        await prisma_client.get_generic_data(
            key="param_name",
            value="general_settings",
            table_name="config",
        )

    # Should have tried reconnect once but not retried the query
    assert mock_find_first.call_count == 1
    prisma_client.attempt_db_reconnect.assert_awaited_once()
