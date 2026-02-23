import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import httpx

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.types.containers.main import ContainerObject, ContainerListResponse, DeleteContainerResult
from litellm.containers.main import (
    create_container, acreate_container,
    list_containers, alist_containers,
    retrieve_container, aretrieve_container,
    delete_container, adelete_container
)


class TestContainerIntegration:
    """Integration tests for the complete container API flow."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock environment variable for API key
        os.environ["OPENAI_API_KEY"] = "sk-test123"

    def teardown_method(self):
        """Clean up after tests."""
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

    @patch('litellm.llms.custom_httpx.llm_http_handler.HTTPHandler')
    def test_container_create_full_flow(self, mock_http_handler):
        """Test the complete container creation flow with mocked HTTP."""
        # Setup mock HTTP response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "cntr_integration_test",
            "object": "container",
            "created_at": 1747857508,
            "status": "running",
            "expires_after": {"anchor": "last_active_at", "minutes": 20},
            "last_active_at": 1747857508,
            "name": "Integration Test Container"
        }
        mock_response.status_code = 200
        
        # Mock the HTTP handler
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_http_handler.return_value = mock_client
        
        with patch('litellm.llms.custom_httpx.llm_http_handler._get_httpx_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            # Execute
            response = create_container(
                name="Integration Test Container",
                expires_after={"anchor": "last_active_at", "minutes": 20},
                custom_llm_provider="openai"
            )
            
            # Verify
            assert isinstance(response, ContainerObject)
            assert response.id == "cntr_integration_test"
            assert response.name == "Integration Test Container"
            assert response.status == "running"

    @patch('litellm.llms.custom_httpx.llm_http_handler.HTTPHandler')
    def test_container_list_full_flow(self, mock_http_handler):
        """Test the complete container listing flow with mocked HTTP."""
        # Setup mock HTTP response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "id": "cntr_list_1",
                    "object": "container",
                    "created_at": 1747857508,
                    "status": "running",
                    "expires_after": {"anchor": "last_active_at", "minutes": 20},
                    "last_active_at": 1747857508,
                    "name": "List Container 1"
                },
                {
                    "id": "cntr_list_2",
                    "object": "container",
                    "created_at": 1747857600,
                    "status": "running",
                    "expires_after": {"anchor": "last_active_at", "minutes": 15},
                    "last_active_at": 1747857600,
                    "name": "List Container 2"
                }
            ],
            "first_id": "cntr_list_1",
            "last_id": "cntr_list_2",
            "has_more": False
        }
        mock_response.status_code = 200
        
        # Mock the HTTP handler
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_http_handler.return_value = mock_client
        
        with patch('litellm.llms.custom_httpx.llm_http_handler._get_httpx_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            # Execute
            response = list_containers(
                limit=10,
                order="desc",
                custom_llm_provider="openai"
            )
            
            # Verify
            assert isinstance(response, ContainerListResponse)
            assert len(response.data) == 2
            assert response.data[0].id == "cntr_list_1"
            assert response.data[1].id == "cntr_list_2"
            assert response.has_more == False

    @patch('litellm.llms.custom_httpx.llm_http_handler.HTTPHandler')
    def test_container_retrieve_full_flow(self, mock_http_handler):
        """Test the complete container retrieval flow with mocked HTTP."""
        container_id = "cntr_retrieve_integration"
        
        # Setup mock HTTP response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": container_id,
            "object": "container",
            "created_at": 1747857508,
            "status": "running",
            "expires_after": {"anchor": "last_active_at", "minutes": 20},
            "last_active_at": 1747857508,
            "name": "Retrieved Integration Container"
        }
        mock_response.status_code = 200
        
        # Mock the HTTP handler
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_http_handler.return_value = mock_client
        
        with patch('litellm.llms.custom_httpx.llm_http_handler._get_httpx_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            # Execute
            response = retrieve_container(
                container_id=container_id,
                custom_llm_provider="openai"
            )
            
            # Verify
            assert isinstance(response, ContainerObject)
            assert response.id == container_id
            assert response.name == "Retrieved Integration Container"

    @patch('litellm.llms.custom_httpx.llm_http_handler.HTTPHandler')
    def test_container_delete_full_flow(self, mock_http_handler):
        """Test the complete container deletion flow with mocked HTTP."""
        container_id = "cntr_delete_integration"
        
        # Setup mock HTTP response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": container_id,
            "object": "container.deleted",
            "deleted": True
        }
        mock_response.status_code = 200
        
        # Mock the HTTP handler
        mock_client = MagicMock()
        mock_client.delete.return_value = mock_response
        mock_http_handler.return_value = mock_client
        
        with patch('litellm.llms.custom_httpx.llm_http_handler._get_httpx_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            # Execute
            response = delete_container(
                container_id=container_id,
                custom_llm_provider="openai"
            )
            
            # Verify
            assert isinstance(response, DeleteContainerResult)
            assert response.id == container_id
            assert response.deleted == True
            assert response.object == "container.deleted"

    @pytest.mark.asyncio
    @patch('litellm.llms.custom_httpx.llm_http_handler.AsyncHTTPHandler')
    async def test_async_container_create_full_flow(self, mock_async_http_handler):
        """Test the complete async container creation flow with mocked HTTP."""
        # Setup mock HTTP response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "cntr_async_integration",
            "object": "container",
            "created_at": 1747857508,
            "status": "running",
            "expires_after": {"anchor": "last_active_at", "minutes": 30},
            "last_active_at": 1747857508,
            "name": "Async Integration Container"
        }
        mock_response.status_code = 200
        
        # Mock the async HTTP handler
        mock_client = MagicMock()
        
        async def mock_post(*args, **kwargs):
            return mock_response
            
        mock_client.post = mock_post
        mock_async_http_handler.return_value = mock_client
        
        with patch('litellm.llms.custom_httpx.llm_http_handler.get_async_httpx_client') as mock_get_async_client:
            mock_get_async_client.return_value = mock_client
            
            # Execute
            response = await acreate_container(
                name="Async Integration Container",
                expires_after={"anchor": "last_active_at", "minutes": 30},
                custom_llm_provider="openai"
            )
            
            # Verify
            assert isinstance(response, ContainerObject)
            assert response.id == "cntr_async_integration"
            assert response.name == "Async Integration Container"

    @pytest.mark.asyncio
    @patch('litellm.llms.custom_httpx.llm_http_handler.AsyncHTTPHandler')
    async def test_async_container_list_full_flow(self, mock_async_http_handler):
        """Test the complete async container listing flow with mocked HTTP."""
        # Setup mock HTTP response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "id": "cntr_async_list",
                    "object": "container",
                    "created_at": 1747857508,
                    "status": "running",
                    "expires_after": {"anchor": "last_active_at", "minutes": 25},
                    "last_active_at": 1747857508,
                    "name": "Async List Container"
                }
            ],
            "first_id": "cntr_async_list",
            "last_id": "cntr_async_list",
            "has_more": False
        }
        mock_response.status_code = 200
        
        # Mock the async HTTP handler
        mock_client = MagicMock()
        
        async def mock_get(*args, **kwargs):
            return mock_response
            
        mock_client.get = mock_get
        mock_async_http_handler.return_value = mock_client
        
        with patch('litellm.llms.custom_httpx.llm_http_handler.get_async_httpx_client') as mock_get_async_client:
            mock_get_async_client.return_value = mock_client
            
            # Execute
            response = await alist_containers(
                limit=5,
                custom_llm_provider="openai"
            )
            
            # Verify
            assert isinstance(response, ContainerListResponse)
            assert len(response.data) == 1
            assert response.data[0].id == "cntr_async_list"

    def test_container_workflow_simulation(self):
        """Test a complete workflow: create -> list -> retrieve -> delete."""
        container_id = "cntr_workflow_test"
        
        # Mock all HTTP responses
        create_response = MagicMock(spec=httpx.Response)
        create_response.json.return_value = {
            "id": container_id,
            "object": "container",
            "created_at": 1747857508,
            "status": "running",
            "expires_after": {"anchor": "last_active_at", "minutes": 20},
            "last_active_at": 1747857508,
            "name": "Workflow Test Container"
        }
        
        list_response = MagicMock(spec=httpx.Response)
        list_response.json.return_value = {
            "object": "list",
            "data": [create_response.json.return_value],
            "first_id": container_id,
            "last_id": container_id,  
            "has_more": False
        }
        
        retrieve_response = create_response  # Same as create
        
        delete_response = MagicMock(spec=httpx.Response)
        delete_response.json.return_value = {
            "id": container_id,
            "object": "container.deleted",
            "deleted": True
        }
        
        with patch('litellm.containers.main.base_llm_http_handler') as mock_handler:
            # Setup different responses for different operations
            mock_handler.container_create_handler.return_value = ContainerObject(**create_response.json.return_value)
            mock_handler.container_list_handler.return_value = ContainerListResponse(**list_response.json.return_value)
            mock_handler.container_retrieve_handler.return_value = ContainerObject(**retrieve_response.json.return_value)
            mock_handler.container_delete_handler.return_value = DeleteContainerResult(**delete_response.json.return_value)
            
            # Execute workflow
            # 1. Create container
            created = create_container(
                name="Workflow Test Container",
                custom_llm_provider="openai"
            )
            assert created.id == container_id
            
            # 2. List containers (should include our created one)
            containers = list_containers(custom_llm_provider="openai")
            assert len(containers.data) == 1
            assert containers.data[0].id == container_id
            
            # 3. Retrieve specific container
            retrieved = retrieve_container(
                container_id=container_id,
                custom_llm_provider="openai"
            )
            assert retrieved.id == container_id
            assert retrieved.name == "Workflow Test Container"
            
            # 4. Delete container
            deleted = delete_container(
                container_id=container_id,
                custom_llm_provider="openai"
            )
            assert deleted.id == container_id
            assert deleted.deleted == True

    def test_error_handling_integration(self):
        """Test error handling in the integration flow."""
        import importlib
        import litellm.containers.main as containers_main_module

        # Reload the module to ensure it has a fresh reference to base_llm_http_handler
        # after conftest reloads litellm
        importlib.reload(containers_main_module)

        # Re-import the function after reload
        from litellm.containers.main import create_container as create_container_fresh

        with patch('litellm.containers.main.base_llm_http_handler') as mock_handler:
            # Simulate an API error
            mock_handler.container_create_handler.side_effect = litellm.APIError(
                status_code=400,
                message="API Error occurred",
                llm_provider="openai",
                model=""
            )

            with pytest.raises(litellm.APIError):
                create_container_fresh(
                    name="Error Test Container",
                    custom_llm_provider="openai"
                )

    @pytest.mark.parametrize("provider", ["openai"])
    def test_provider_support(self, provider):
        """Test that the container API works with supported providers."""
        import importlib
        import litellm.containers.main as containers_main_module

        # Reload the module to ensure it has a fresh reference to base_llm_http_handler
        # after conftest reloads litellm (same pattern as test_error_handling_integration)
        importlib.reload(containers_main_module)

        from litellm.containers.main import create_container as create_container_fresh

        mock_response = ContainerObject(
            id="cntr_provider_test",
            object="container",
            created_at=1747857508,
            status="running",
            expires_after={"anchor": "last_active_at", "minutes": 20},
            last_active_at=1747857508,
            name="Provider Test Container"
        )
        
        with patch('litellm.containers.main.base_llm_http_handler') as mock_handler:
            mock_handler.container_create_handler.return_value = mock_response
            
            response = create_container_fresh(
                name="Provider Test Container",
                custom_llm_provider=provider
            )
            
            assert response.name == "Provider Test Container"
