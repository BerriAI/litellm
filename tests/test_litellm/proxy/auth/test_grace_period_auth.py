import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.auth.auth_checks import _check_previous_token_grace_period


class TestGracePeriodAuthentication:
    """Test the _check_previous_token_grace_period function."""

    @pytest.mark.asyncio
    async def test_authenticate_previous_token_within_grace_period(self):
        """
        Test that a previous token is authenticated when within the grace period.
        """
        # Setup
        mock_prisma_client = AsyncMock()
        now = datetime.now(timezone.utc)
        grace_expiry = now + timedelta(minutes=15)
        
        mock_key_record = MagicMock()
        mock_key_record.token = "new-token-hash"
        mock_key_record.previous_token = "old-token-hash"
        mock_key_record.previous_token_expires = grace_expiry
        
        mock_prisma_client.db.litellm_verificationtoken.find_first.return_value = mock_key_record
        
        result = await _check_previous_token_grace_period(
            hashed_token="old-token-hash",
            prisma_client=mock_prisma_client,
        )
        
        assert result is not None
        assert result == mock_key_record
        
        # Verify the query was correct
        mock_prisma_client.db.litellm_verificationtoken.find_first.assert_called_once()
        call_args = mock_prisma_client.db.litellm_verificationtoken.find_first.call_args
        where_clause = call_args[1]["where"]
        assert where_clause["previous_token"] == "old-token-hash"
        assert "previous_token_expires" in where_clause

    @pytest.mark.asyncio
    async def test_reject_previous_token_after_grace_period_expires(self):
        """Test that a previous token is rejected when the grace period has expired."""
        mock_prisma_client = AsyncMock()
        mock_prisma_client.db.litellm_verificationtoken.find_first.return_value = None
        
        result = await _check_previous_token_grace_period(
            hashed_token="old-token-hash",
            prisma_client=mock_prisma_client,
        )
        
        assert result is None

    @pytest.mark.asyncio
    async def test_return_none_when_no_matching_previous_token(self):
        """Test that None is returned when the token doesn't match any previous_token."""
        mock_prisma_client = AsyncMock()
        mock_prisma_client.db.litellm_verificationtoken.find_first.return_value = None
        
        result = await _check_previous_token_grace_period(
            hashed_token="unknown-token-hash",
            prisma_client=mock_prisma_client,
        )
        
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_database_exception_gracefully(self):
        """Test that database exceptions are handled gracefully and return None."""
        mock_prisma_client = AsyncMock()
        mock_prisma_client.db.litellm_verificationtoken.find_first.side_effect = Exception(
            "Database connection error"
        )
        
        result = await _check_previous_token_grace_period(
            hashed_token="some-token-hash",
            prisma_client=mock_prisma_client,
        )
        
        assert result is None

    @pytest.mark.asyncio
    async def test_query_with_correct_time_comparison(self):
        """Test that the grace period query uses >= (gte) for time comparison."""
        mock_prisma_client = AsyncMock()
        mock_prisma_client.db.litellm_verificationtoken.find_first.return_value = None
        
        with patch('litellm.proxy.auth.auth_checks.datetime') as mock_datetime:
            fixed_now = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
            mock_datetime.now.return_value = fixed_now
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            
            await _check_previous_token_grace_period(
                hashed_token="test-token",
                prisma_client=mock_prisma_client,
            )
        
        call_args = mock_prisma_client.db.litellm_verificationtoken.find_first.call_args
        where_clause = call_args[1]["where"]
        
        # Verify the query uses "gte" (greater than or equal) for expiry check
        assert where_clause["previous_token_expires"] == {"gte": fixed_now}
