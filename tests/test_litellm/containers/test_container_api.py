import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.containers.main import (
    acreate_container,
    adelete_container,
    alist_containers,
    aretrieve_container,
    create_container,
    delete_container,
    list_containers,
    retrieve_container,
)
from litellm.main import base_llm_http_handler
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.llms.openai.containers.transformation import OpenAIContainerConfig
from litellm.router import Router
from litellm.types.containers.main import (
    ContainerListResponse,
    ContainerObject,
    DeleteContainerResult,
)


@pytest.fixture(autouse=True)
def clear_client_cache():
    """
    Clear the HTTP client cache before each test to ensure mocks are used.
    This prevents cached real clients from being reused across tests.
    """
    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if cache is not None:
        cache.flush_cache()
    yield
    # Clear again after test to avoid polluting other tests
    if cache is not None:
        cache.flush_cache()


class TestContainerAPI:
    """Test suite for container API functionality."""

    def test_create_container_basic(self):
        """Test basic container creation functionality."""
        # Mock the container creation response
        mock_response = ContainerObject(
            id="cntr_123456",
            object="container",
            created_at=1747857508,
            status="running",
            expires_after={"anchor": "last_active_at", "minutes": 20},
            last_active_at=1747857508,
            name="Test Container"
        )
        
        with patch.object(base_llm_http_handler, 'container_create_handler', return_value=mock_response):
            response = create_container(
                name="Test Container",
                custom_llm_provider="openai"
            )
            
            assert isinstance(response, ContainerObject)
            assert response.id == "cntr_123456"
            assert response.name == "Test Container"
            assert response.status == "running"
            assert response.object == "container"

    def test_create_container_with_expires_after(self):
        """Test container creation with expires_after parameter."""
        mock_response = ContainerObject(
            id="cntr_789",
            object="container", 
            created_at=1747857508,
            status="running",
            expires_after={"anchor": "last_active_at", "minutes": 30},
            last_active_at=1747857508,
            name="Expiring Container"
        )
        
        with patch.object(base_llm_http_handler, 'container_create_handler', return_value=mock_response):
            response = create_container(
                name="Expiring Container",
                expires_after={"anchor": "last_active_at", "minutes": 30},
                custom_llm_provider="openai"
            )
            
            assert response.expires_after.minutes == 30
            assert response.expires_after.anchor == "last_active_at"

    def test_create_container_with_file_ids(self):
        """Test container creation with file_ids parameter."""
        mock_response = ContainerObject(
            id="cntr_file_test",
            object="container",
            created_at=1747857508,
            status="running",
            expires_after={"anchor": "last_active_at", "minutes": 20},
            last_active_at=1747857508,
            name="Container with Files"
        )
        
        with patch.object(base_llm_http_handler, 'container_create_handler', return_value=mock_response):
            response = create_container(
                name="Container with Files",
                file_ids=["file_123", "file_456"],
                custom_llm_provider="openai"
            )
            
            assert response.name == "Container with Files"

    @pytest.mark.asyncio
    async def test_acreate_container_basic(self):
        """Test basic async container creation functionality."""
        mock_response = ContainerObject(
            id="cntr_async_123",
            object="container",
            created_at=1747857508,
            status="running", 
            expires_after={"anchor": "last_active_at", "minutes": 20},
            last_active_at=1747857508,
            name="Async Test Container"
        )
        
        with patch.object(base_llm_http_handler, 'container_create_handler', return_value=mock_response):
            response = await acreate_container(
                name="Async Test Container",
                custom_llm_provider="openai"
            )
            
            assert isinstance(response, ContainerObject)
            assert response.id == "cntr_async_123"
            assert response.name == "Async Test Container"


    @pytest.mark.asyncio
    async def test_alist_containers_basic(self):
        """Test basic async container listing functionality."""
        mock_response = ContainerListResponse(
            object="list",
            data=[
                ContainerObject(
                    id="cntr_async_list",
                    object="container",
                    created_at=1747857508,
                    status="running",
                    expires_after={"anchor": "last_active_at", "minutes": 20},
                    last_active_at=1747857508,
                    name="Async List Container"
                )
            ],
            first_id="cntr_async_list",
            last_id="cntr_async_list",
            has_more=False
        )
        
        with patch.object(base_llm_http_handler, 'container_list_handler', return_value=mock_response):
            response = await alist_containers(
                custom_llm_provider="openai"
            )
            
            assert isinstance(response, ContainerListResponse)
            assert len(response.data) == 1

    @pytest.mark.parametrize(
        "container_id,container_name,status,provider",
        [
            ("cntr_retrieve_test", "Retrieved Container", "running", "openai"),
            ("cntr_different_id", "Another Container", "stopped", "openai"),
        ],
    )
    def test_retrieve_container_basic(self, container_id, container_name, status, provider):
        """Test basic container retrieval functionality.
        
        This test verifies that:
        1. retrieve_container correctly calls the handler with the container_id
        2. The response is properly deserialized into a ContainerObject
        3. All fields are correctly mapped from the handler response
        4. The function works with different container states and IDs
        """
        # Arrange: Create mock response with test parameters
        mock_response = ContainerObject(
            id=container_id,
            object="container",
            created_at=1747857508,
            status=status,
            expires_after={"anchor": "last_active_at", "minutes": 20},
            last_active_at=1747857508,
            name=container_name
        )
        
        with patch.object(base_llm_http_handler, 'container_retrieve_handler', return_value=mock_response) as mock_method:
            # Act: Call retrieve_container
            response = retrieve_container(
                container_id=container_id,
                custom_llm_provider=provider
            )
            
            # Assert: Verify the handler was called correctly
            mock_method.assert_called_once()
            call_kwargs = mock_method.call_args.kwargs
            assert call_kwargs["container_id"] == container_id
            
            # Assert: Verify response structure and content
            assert isinstance(response, ContainerObject)
            assert response.id == container_id
            assert response.name == container_name
            assert response.status == status
            assert response.object == "container"
            assert response.expires_after.minutes == 20
            assert response.expires_after.anchor == "last_active_at"

    @pytest.mark.asyncio
    async def test_aretrieve_container_basic(self):
        """Test basic async container retrieval functionality."""
        container_id = "cntr_async_retrieve"
        mock_response = ContainerObject(
            id=container_id,
            object="container",
            created_at=1747857508,
            status="running",
            expires_after={"anchor": "last_active_at", "minutes": 20},
            last_active_at=1747857508,
            name="Async Retrieved Container"
        )
        
        with patch.object(base_llm_http_handler, 'container_retrieve_handler', return_value=mock_response):
            response = await aretrieve_container(
                container_id=container_id,
                custom_llm_provider="openai"
            )
            
            assert isinstance(response, ContainerObject)
            assert response.id == container_id

    def test_delete_container_basic(self):
        """Test basic container deletion functionality."""
        container_id = "cntr_delete_test"
        mock_response = DeleteContainerResult(
            id=container_id,
            object="container.deleted",
            deleted=True
        )
        
        with patch.object(base_llm_http_handler, 'container_delete_handler', return_value=mock_response):
            response = delete_container(
                container_id=container_id,
                custom_llm_provider="openai"
            )
            
            assert isinstance(response, DeleteContainerResult)
            assert response.id == container_id
            assert response.deleted == True
            assert response.object == "container.deleted"

    @pytest.mark.asyncio
    async def test_adelete_container_basic(self):
        """Test basic async container deletion functionality."""
        container_id = "cntr_async_delete"
        mock_response = DeleteContainerResult(
            id=container_id,
            object="container.deleted",
            deleted=True
        )
        
        with patch.object(base_llm_http_handler, 'container_delete_handler', return_value=mock_response):
            response = await adelete_container(
                container_id=container_id,
                custom_llm_provider="openai"
            )
            
            assert isinstance(response, DeleteContainerResult)
            assert response.id == container_id
            assert response.deleted == True

    def test_create_container_error_handling(self):
        """Test error handling in container creation."""
        with patch.object(base_llm_http_handler, 'container_create_handler', side_effect=Exception("API Error")):
            with pytest.raises(Exception):
                create_container(
                    name="Error Test Container",
                    custom_llm_provider="openai"
                )

    def test_container_provider_config_retrieval(self):
        """Test that provider config is retrieved correctly."""
        mock_response = ContainerObject(
            id="cntr_config_test",
            object="container",
            created_at=1747857508,
            status="running",
            expires_after={"anchor": "last_active_at", "minutes": 20},
            last_active_at=1747857508,
            name="Config Test"
        )
        
        with patch('litellm.containers.main.ProviderConfigManager') as mock_config_manager:
            mock_config_manager.get_provider_container_config.return_value = OpenAIContainerConfig()
            
            with patch.object(base_llm_http_handler, 'container_create_handler', return_value=mock_response):
                response = create_container(
                    name="Config Test",
                    custom_llm_provider="openai"
                )
                
                # Verify provider config was requested
                mock_config_manager.get_provider_container_config.assert_called_once()
                assert response.name == "Config Test"

    @pytest.mark.asyncio
    async def test_router_acreate_container_without_model(self):
        """
        Test that router.acreate_container works without a model configured.
        Ensures container operations bypass model deployment lookup.
        """
        router = Router(model_list=[])

        mock_response = ContainerObject(
            id="cntr_test",
            object="container",
            created_at=1747857508,
            status="running",
            expires_after={"anchor": "last_active_at", "minutes": 20},
            last_active_at=1747857508,
            name="Test Container"
        )

        # Mock async_container_create_handler since router.acreate_container
        # uses _is_async=True which calls the async handler
        with patch.object(
            base_llm_http_handler,
            'async_container_create_handler',
            new_callable=AsyncMock,
            return_value=mock_response
        ):
            result = await router.acreate_container(
                name="Test Container",
                custom_llm_provider="openai"
            )

            assert result.id == "cntr_test"
            assert result.name == "Test Container"
