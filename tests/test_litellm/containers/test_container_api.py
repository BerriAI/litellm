import asyncio
import json
import os
import sys
from unittest.mock import MagicMock, patch

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
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.llms.openai.containers.transformation import OpenAIContainerConfig
from litellm.router import Router
from litellm.types.containers.main import (
    ContainerListResponse,
    ContainerObject,
    DeleteContainerResult,
)


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
        
        with patch('litellm.containers.main.base_llm_http_handler') as mock_handler:
            mock_handler.container_create_handler.return_value = mock_response
            
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
        
        with patch('litellm.containers.main.base_llm_http_handler') as mock_handler:
            mock_handler.container_create_handler.return_value = mock_response
            
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
        
        with patch('litellm.containers.main.base_llm_http_handler') as mock_handler:
            mock_handler.container_create_handler.return_value = mock_response
            
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
        
        with patch('litellm.containers.main.base_llm_http_handler') as mock_handler:
            mock_handler.container_create_handler.return_value = mock_response
            
            response = await acreate_container(
                name="Async Test Container",
                custom_llm_provider="openai"
            )
            
            assert isinstance(response, ContainerObject)
            assert response.id == "cntr_async_123"
            assert response.name == "Async Test Container"

    def test_list_containers_basic(self):
        """Test basic container listing functionality."""
        mock_response = ContainerListResponse(
            object="list",
            data=[
                ContainerObject(
                    id="cntr_1",
                    object="container",
                    created_at=1747857508,
                    status="running",
                    expires_after={"anchor": "last_active_at", "minutes": 20},
                    last_active_at=1747857508,
                    name="Container 1"
                ),
                ContainerObject(
                    id="cntr_2", 
                    object="container",
                    created_at=1747857600,
                    status="running",
                    expires_after={"anchor": "last_active_at", "minutes": 15},
                    last_active_at=1747857600,
                    name="Container 2"
                )
            ],
            first_id="cntr_1",
            last_id="cntr_2",
            has_more=False
        )
        
        with patch('litellm.containers.main.base_llm_http_handler') as mock_handler:
            mock_handler.container_list_handler.return_value = mock_response
            
            response = list_containers(
                custom_llm_provider="openai"
            )
            
            assert isinstance(response, ContainerListResponse)
            assert len(response.data) == 2
            assert response.data[0].id == "cntr_1"
            assert response.data[1].id == "cntr_2"
            assert response.has_more == False

    def test_list_containers_with_params(self):
        """Test container listing with parameters."""
        mock_response = ContainerListResponse(
            object="list",
            data=[
                ContainerObject(
                    id="cntr_limited",
                    object="container",
                    created_at=1747857508,
                    status="running",
                    expires_after={"anchor": "last_active_at", "minutes": 20},
                    last_active_at=1747857508,
                    name="Limited Container"
                )
            ],
            first_id="cntr_limited",
            last_id="cntr_limited", 
            has_more=True
        )
        
        with patch('litellm.containers.main.base_llm_http_handler') as mock_handler:
            mock_handler.container_list_handler.return_value = mock_response
            
            response = list_containers(
                limit=1,
                order="desc",
                after="cntr_prev",
                custom_llm_provider="openai"
            )
            
            assert len(response.data) == 1
            assert response.has_more == True

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
        
        with patch('litellm.containers.main.base_llm_http_handler') as mock_handler:
            mock_handler.container_list_handler.return_value = mock_response
            
            response = await alist_containers(
                custom_llm_provider="openai"
            )
            
            assert isinstance(response, ContainerListResponse)
            assert len(response.data) == 1

    def test_retrieve_container_basic(self):
        """Test basic container retrieval functionality."""
        container_id = "cntr_retrieve_test"
        mock_response = ContainerObject(
            id=container_id,
            object="container",
            created_at=1747857508,
            status="running",
            expires_after={"anchor": "last_active_at", "minutes": 20},
            last_active_at=1747857508,
            name="Retrieved Container"
        )
        
        with patch('litellm.containers.main.base_llm_http_handler') as mock_handler:
            mock_handler.container_retrieve_handler.return_value = mock_response
            
            response = retrieve_container(
                container_id=container_id,
                custom_llm_provider="openai"
            )
            
            assert isinstance(response, ContainerObject)
            assert response.id == container_id
            assert response.name == "Retrieved Container"

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
        
        with patch('litellm.containers.main.base_llm_http_handler') as mock_handler:
            mock_handler.container_retrieve_handler.return_value = mock_response
            
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
        
        with patch('litellm.containers.main.base_llm_http_handler') as mock_handler:
            mock_handler.container_delete_handler.return_value = mock_response
            
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
        
        with patch('litellm.containers.main.base_llm_http_handler') as mock_handler:
            mock_handler.container_delete_handler.return_value = mock_response
            
            response = await adelete_container(
                container_id=container_id,
                custom_llm_provider="openai"
            )
            
            assert isinstance(response, DeleteContainerResult)
            assert response.id == container_id
            assert response.deleted == True

    def test_create_container_error_handling(self):
        """Test error handling in container creation."""
        with patch('litellm.containers.main.base_llm_http_handler') as mock_handler:
            mock_handler.container_create_handler.side_effect = Exception("API Error")
            
            with pytest.raises(Exception):
                create_container(
                    name="Error Test Container",
                    custom_llm_provider="openai"
                )

    def test_container_provider_config_retrieval(self):
        """Test that provider config is retrieved correctly."""
        with patch('litellm.containers.main.ProviderConfigManager') as mock_config_manager:
            mock_config_manager.get_provider_container_config.return_value = OpenAIContainerConfig()
            
            with patch('litellm.containers.main.base_llm_http_handler') as mock_handler:
                mock_response = ContainerObject(
                    id="cntr_config_test",
                    object="container",
                    created_at=1747857508,
                    status="running",
                    expires_after={"anchor": "last_active_at", "minutes": 20},
                    last_active_at=1747857508,
                    name="Config Test"
                )
                mock_handler.container_create_handler.return_value = mock_response
                
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

        with patch('litellm.containers.main.base_llm_http_handler') as mock_handler:
            mock_handler.container_create_handler.return_value = mock_response

            result = await router.acreate_container(
                name="Test Container",
                custom_llm_provider="openai"
            )

            assert result.id == "cntr_test"
            assert result.name == "Test Container"
