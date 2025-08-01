import json
import os
import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path


from litellm.proxy.db.prisma_client import PrismaWrapper, should_update_prisma_schema


def test_should_update_prisma_schema(monkeypatch):
    # CASE 1: Environment variable behavior
    # When DISABLE_SCHEMA_UPDATE is not set -> should update
    monkeypatch.setenv("DISABLE_SCHEMA_UPDATE", None)
    assert should_update_prisma_schema() == True

    # When DISABLE_SCHEMA_UPDATE="true" -> should not update
    monkeypatch.setenv("DISABLE_SCHEMA_UPDATE", "true")
    assert should_update_prisma_schema() == False

    # When DISABLE_SCHEMA_UPDATE="false" -> should update
    monkeypatch.setenv("DISABLE_SCHEMA_UPDATE", "false")
    assert should_update_prisma_schema() == True

    # CASE 2: Explicit parameter behavior (overrides env var)
    monkeypatch.setenv("DISABLE_SCHEMA_UPDATE", None)
    assert should_update_prisma_schema(True) == False  # Param True -> should not update

    monkeypatch.setenv("DISABLE_SCHEMA_UPDATE", None)  # Set env var opposite to param
    assert should_update_prisma_schema(False) == True  # Param False -> should update


@pytest.mark.asyncio
async def test_recreate_prisma_client_successful_disconnect():
    """
    Test that recreate_prisma_client works normally when disconnect succeeds.
    """
    # Mock the original prisma client
    mock_prisma = AsyncMock()
    
    # Create a PrismaWrapper instance
    wrapper = PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=False)
    
    # Configure disconnect to succeed
    mock_prisma.disconnect.return_value = None
    
    # Mock the Prisma class constructor
    with patch("prisma.Prisma") as mock_prisma_class:
        mock_new_prisma = AsyncMock()
        mock_prisma_class.return_value = mock_new_prisma
        
        # Call the method
        await wrapper.recreate_prisma_client("postgresql://new:new@localhost:5432/new")
        
        # Verify that disconnect was called
        mock_prisma.disconnect.assert_called_once()

        # Verify that a new Prisma client was created and connected
        mock_prisma_class.assert_called_once()
        mock_new_prisma.connect.assert_called_once()
        
        # Verify that the new client replaced the original
        assert wrapper._original_prisma == mock_new_prisma
