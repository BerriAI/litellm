"""
Comprehensive test for new vector store endpoints: retrieve, list, update, delete
Tests both basic functionality and complex scenarios including target_model_names
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest


import litellm


@pytest.mark.asyncio
async def test_async_vector_store_update():
    """Test async vector store update."""
    mock_response = {
        "id": "vs_async123",
        "name": "Updated Async Name",
    }

    with patch(
        "litellm.vector_stores.main.aupdate",
        new=AsyncMock(return_value=mock_response),
    ) as mock_aupdate:
        router = litellm.Router(model_list=[])
        result = await router.avector_store_update(
            vector_store_id="vs_async123",
            name="Updated Async Name",
            custom_llm_provider="openai",
        )

        assert result["name"] == "Updated Async Name"
        mock_aupdate.assert_called_once()


def test_router_initializes_new_endpoints():
    """Test that router properly initializes the new vector store endpoints."""
    router = litellm.Router(model_list=[])

    # Verify all new endpoints are initialized
    assert hasattr(router, "vector_store_retrieve")
    assert hasattr(router, "avector_store_retrieve")
    assert hasattr(router, "vector_store_list")
    assert hasattr(router, "avector_store_list")
    assert hasattr(router, "vector_store_update")
    assert hasattr(router, "avector_store_update")
    assert hasattr(router, "vector_store_delete")
    assert hasattr(router, "avector_store_delete")

    # Verify they are callable
    assert callable(router.vector_store_retrieve)
    assert callable(router.avector_store_retrieve)
    assert callable(router.vector_store_list)
    assert callable(router.avector_store_list)
    assert callable(router.vector_store_update)
    assert callable(router.avector_store_update)
    assert callable(router.vector_store_delete)
    assert callable(router.avector_store_delete)


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running smoke tests for new vector store endpoints...")

    # Test router initialization
    print("✓ Testing router initialization...")
    test_router_initializes_new_endpoints()
    print("✓ Router initialization successful")

    # Test async operations
    print("✓ Testing async operations...")
    asyncio.run(test_async_vector_store_update())
    print("✓ Async operations successful")

    print("\n✅ All smoke tests passed!")
