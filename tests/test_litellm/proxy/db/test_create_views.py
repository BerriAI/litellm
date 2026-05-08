"""
Tests for create_missing_views exception handling fix.

Verifies that real DB errors (auth failures, connection errors, etc.)
are re-raised instead of being silently swallowed, while genuine
"view not found" errors still trigger view creation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, call


@pytest.mark.asyncio
async def test_create_views_reraises_connection_error():
    """should re-raise exceptions that are NOT 'does not exist' errors (e.g. connection errors)."""
    from litellm.proxy.db.create_views import create_missing_views

    mock_db = MagicMock()
    mock_db.query_raw = AsyncMock(
        side_effect=Exception("connection refused: unable to connect to database")
    )
    mock_db.execute_raw = AsyncMock()

    with pytest.raises(Exception, match="connection refused"):
        await create_missing_views(mock_db)

    mock_db.execute_raw.assert_not_called()


@pytest.mark.asyncio
async def test_create_views_reraises_permission_error():
    """should re-raise permission denied errors, not treat them as missing views."""
    from litellm.proxy.db.create_views import create_missing_views

    mock_db = MagicMock()
    mock_db.query_raw = AsyncMock(
        side_effect=Exception(
            "permission denied for table LiteLLM_VerificationTokenView"
        )
    )
    mock_db.execute_raw = AsyncMock()

    with pytest.raises(Exception, match="permission denied"):
        await create_missing_views(mock_db)

    mock_db.execute_raw.assert_not_called()


@pytest.mark.asyncio
async def test_create_views_creates_view_on_does_not_exist():
    """should call execute_raw to create view when error contains 'does not exist'."""
    from litellm.proxy.db.create_views import create_missing_views

    mock_db = MagicMock()
    mock_db.query_raw = AsyncMock(
        side_effect=[
            Exception('relation "LiteLLM_VerificationTokenView" does not exist'),
            None,  # MonthlyGlobalSpend exists
            None,  # Last30dKeysBySpend exists
            None,  # Last30dModelsBySpend exists
            None,  # MonthlyGlobalSpendPerKey exists
            None,  # MonthlyGlobalSpendPerUserPerKey exists
            None,  # DailyTagSpend exists
            None,  # Last30dTopEndUsersSpend exists
        ]
    )
    mock_db.execute_raw = AsyncMock(return_value=None)

    await create_missing_views(mock_db)

    mock_db.execute_raw.assert_called_once()
    created_sql = mock_db.execute_raw.call_args[0][0]
    assert 'CREATE VIEW "LiteLLM_VerificationTokenView"' in created_sql


@pytest.mark.asyncio
async def test_create_views_creates_view_on_undefined_error():
    """should treat 'undefined' errors as 'view not found' and attempt creation."""
    from litellm.proxy.db.create_views import create_missing_views

    mock_db = MagicMock()
    mock_db.query_raw = AsyncMock(
        side_effect=[
            Exception("undefined table LiteLLM_VerificationTokenView"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]
    )
    mock_db.execute_raw = AsyncMock(return_value=None)

    await create_missing_views(mock_db)

    mock_db.execute_raw.assert_called_once()


@pytest.mark.asyncio
async def test_create_views_skips_creation_when_view_exists():
    """should not call execute_raw when all views already exist."""
    from litellm.proxy.db.create_views import create_missing_views

    mock_db = MagicMock()
    mock_db.query_raw = AsyncMock(return_value=[{"?column?": 1}])
    mock_db.execute_raw = AsyncMock()

    await create_missing_views(mock_db)

    mock_db.execute_raw.assert_not_called()


@pytest.mark.asyncio
async def test_create_views_reraises_undefined_function_error():
    """should re-raise 'undefined function' errors — bare 'undefined' is too broad
    and would previously misclassify DB function errors as missing-view signals."""
    from litellm.proxy.db.create_views import create_missing_views

    mock_db = MagicMock()
    mock_db.query_raw = AsyncMock(
        side_effect=Exception("ERROR: undefined function pg_get_viewdef()")
    )
    mock_db.execute_raw = AsyncMock()

    with pytest.raises(Exception, match="undefined function"):
        await create_missing_views(mock_db)

    mock_db.execute_raw.assert_not_called()


@pytest.mark.asyncio
async def test_create_views_creates_view_on_undefined_table_error():
    """should treat 'undefined table' as a missing-view signal and attempt creation."""
    from litellm.proxy.db.create_views import create_missing_views

    mock_db = MagicMock()
    mock_db.query_raw = AsyncMock(
        side_effect=[
            Exception('undefined table "LiteLLM_VerificationTokenView"'),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]
    )
    mock_db.execute_raw = AsyncMock(return_value=None)

    await create_missing_views(mock_db)

    mock_db.execute_raw.assert_called_once()
