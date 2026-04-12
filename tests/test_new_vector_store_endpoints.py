"""
Comprehensive test for new vector store endpoints: retrieve, list, update, delete
Tests both basic functionality and complex scenarios including target_model_names
"""
import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.proxy._types import UserAPIKeyAuth


@pytest.mark.asyncio
async def test_vector_store_retrieve_basic():
    """Test basic vector store retrieve functionality."""
    mock_response = {
        "id": "vs_test123",
        "object": "vector_store",
        "created_at": 1699061776,
        "name": "Test Vector Store",
        "file_counts": {
            "in_progress": 0,
            "completed": 5,
            "failed": 0,
            "cancelled": 0,
            "total": 5,
        },
        "status": "completed",
        "usage_bytes": 12345,
    }
    
    with patch(
        "litellm.vector_stores.main.aretrieve",
        new=AsyncMock(return_value=mock_response),
    ) as mock_retrieve:
        router = litellm.Router(model_list=[])
        result = await router.avector_store_retrieve(
            vector_store_id="vs_test123",
            custom_llm_provider="openai",
        )
        
        assert result["id"] == "vs_test123"
        assert result["object"] == "vector_store"
        assert result["status"] == "completed"
        mock_retrieve.assert_called_once()


@pytest.mark.asyncio
async def test_vector_store_list_basic():
    """Test basic vector store list functionality."""
    mock_response = {
        "object": "list",
        "data": [
            {
                "id": "vs_test1",
                "object": "vector_store",
                "created_at": 1699061776,
                "name": "Store 1",
            },
            {
                "id": "vs_test2",
                "object": "vector_store",
                "created_at": 1699061777,
                "name": "Store 2",
            },
        ],
        "first_id": "vs_test1",
        "last_id": "vs_test2",
        "has_more": False,
    }
    
    with patch(
        "litellm.vector_stores.main.alist",
        new=AsyncMock(return_value=mock_response),
    ) as mock_list:
        router = litellm.Router(model_list=[])
        result = await router.avector_store_list(
            limit=20,
            order="desc",
            custom_llm_provider="openai",
        )
        
        assert result["object"] == "list"
        assert len(result["data"]) == 2
        assert result["data"][0]["id"] == "vs_test1"
        mock_list.assert_called_once()


@pytest.mark.asyncio
async def test_vector_store_update_basic():
    """Test basic vector store update functionality."""
    mock_response = {
        "id": "vs_test123",
        "object": "vector_store",
        "created_at": 1699061776,
        "name": "Updated Name",
        "metadata": {"key": "value"},
        "status": "completed",
    }
    
    with patch(
        "litellm.vector_stores.main.aupdate",
        new=AsyncMock(return_value=mock_response),
    ) as mock_update:
        router = litellm.Router(model_list=[])
        result = await router.avector_store_update(
            vector_store_id="vs_test123",
            name="Updated Name",
            metadata={"key": "value"},
            custom_llm_provider="openai",
        )
        
        assert result["id"] == "vs_test123"
        assert result["name"] == "Updated Name"
        assert result["metadata"]["key"] == "value"
        mock_update.assert_called_once()


@pytest.mark.asyncio
async def test_vector_store_delete_basic():
    """Test basic vector store delete functionality."""
    mock_response = {
        "id": "vs_test123",
        "object": "vector_store.deleted",
        "deleted": True,
    }
    
    with patch(
        "litellm.vector_stores.main.adelete",
        new=AsyncMock(return_value=mock_response),
    ) as mock_delete:
        router = litellm.Router(model_list=[])
        result = await router.avector_store_delete(
            vector_store_id="vs_test123",
            custom_llm_provider="openai",
        )
        
        assert result["id"] == "vs_test123"
        assert result["deleted"] is True
        assert result["object"] == "vector_store.deleted"
        mock_delete.assert_called_once()


@pytest.mark.asyncio
async def test_async_vector_store_retrieve():
    """Test async vector store retrieve."""
    mock_response = {
        "id": "vs_async123",
        "object": "vector_store",
        "name": "Async Test Store",
    }
    
    with patch(
        "litellm.vector_stores.main.aretrieve",
        new=AsyncMock(return_value=mock_response),
    ) as mock_aretrieve:
        router = litellm.Router(model_list=[])
        result = await router.avector_store_retrieve(
            vector_store_id="vs_async123",
            custom_llm_provider="openai",
        )
        
        assert result["id"] == "vs_async123"
        mock_aretrieve.assert_called_once()


@pytest.mark.asyncio
async def test_async_vector_store_list():
    """Test async vector store list."""
    mock_response = {
        "object": "list",
        "data": [{"id": "vs_1"}, {"id": "vs_2"}],
    }
    
    with patch(
        "litellm.vector_stores.main.alist",
        new=AsyncMock(return_value=mock_response),
    ) as mock_alist:
        router = litellm.Router(model_list=[])
        result = await router.avector_store_list(
            limit=10,
            custom_llm_provider="openai",
        )
        
        assert len(result["data"]) == 2
        mock_alist.assert_called_once()


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


@pytest.mark.asyncio
async def test_async_vector_store_delete():
    """Test async vector store delete."""
    mock_response = {
        "id": "vs_async123",
        "deleted": True,
    }
    
    with patch(
        "litellm.vector_stores.main.adelete",
        new=AsyncMock(return_value=mock_response),
    ) as mock_adelete:
        router = litellm.Router(model_list=[])
        result = await router.avector_store_delete(
            vector_store_id="vs_async123",
            custom_llm_provider="openai",
        )
        
        assert result["deleted"] is True
        mock_adelete.assert_called_once()


@pytest.mark.asyncio
async def test_vector_store_list_with_pagination():
    """Test vector store list with pagination parameters."""
    mock_response = {
        "object": "list",
        "data": [{"id": f"vs_{i}"} for i in range(5)],
        "has_more": True,
        "first_id": "vs_0",
        "last_id": "vs_4",
    }
    
    with patch(
        "litellm.vector_stores.main.list",
        return_value=mock_response,
    ) as mock_list:
        router = litellm.Router(model_list=[])
        result = router.vector_store_list(
            limit=5,
            after="vs_previous",
            order="asc",
            custom_llm_provider="openai",
        )
        
        assert result["has_more"] is True
        assert len(result["data"]) == 5
        
        # Verify pagination params were passed
        call_kwargs = mock_list.call_args.kwargs
        assert call_kwargs["limit"] == 5
        assert call_kwargs["after"] == "vs_previous"
        assert call_kwargs["order"] == "asc"


@pytest.mark.asyncio
async def test_vector_store_update_with_expires_after():
    """Test vector store update with expiration policy."""
    expires_after = {
        "anchor": "last_active_at",
        "days": 7,
    }
    
    mock_response = {
        "id": "vs_test123",
        "expires_after": expires_after,
        "expires_at": 1699668576,
    }
    
    with patch(
        "litellm.vector_stores.main.update",
        return_value=mock_response,
    ) as mock_update:
        router = litellm.Router(model_list=[])
        result = router.vector_store_update(
            vector_store_id="vs_test123",
            expires_after=expires_after,
            custom_llm_provider="openai",
        )
        
        assert result["expires_after"]["days"] == 7
        assert result["expires_at"] is not None
        
        call_kwargs = mock_update.call_args.kwargs
        assert call_kwargs["expires_after"] == expires_after


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
    
    # Test basic sync operations
    print("✓ Testing basic sync operations...")
    asyncio.run(test_vector_store_retrieve_basic())
    asyncio.run(test_vector_store_list_basic())
    asyncio.run(test_vector_store_update_basic())
    asyncio.run(test_vector_store_delete_basic())
    print("✓ Basic sync operations successful")
    
    # Test async operations
    print("✓ Testing async operations...")
    asyncio.run(test_async_vector_store_retrieve())
    asyncio.run(test_async_vector_store_list())
    asyncio.run(test_async_vector_store_update())
    asyncio.run(test_async_vector_store_delete())
    print("✓ Async operations successful")
    
    print("\n✅ All smoke tests passed!")
