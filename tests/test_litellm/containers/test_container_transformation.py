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
from litellm.llms.openai.containers.transformation import OpenAIContainerConfig
from litellm.llms.base_llm.containers.transformation import BaseContainerConfig
from litellm.types.containers.main import ContainerObject, ContainerListResponse, DeleteContainerResult
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging


class TestOpenAIContainerTransformation:
    """Test suite for OpenAI container transformation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = OpenAIContainerConfig()
        self.logging_obj = LiteLLMLogging(
            model="",
            messages=[],
            stream=False,
            call_type="create_container",
            start_time=None,
            litellm_call_id="test_call_id",
            function_id="test_function_id"
        )

    def test_get_supported_openai_params(self):
        """Test that supported OpenAI parameters are returned correctly."""
        supported_params = self.config.get_supported_openai_params()
        
        # Check that essential container parameters are supported
        assert "name" in supported_params
        assert "expires_after" in supported_params
        assert "file_ids" in supported_params

    def test_map_openai_params_basic(self):
        """Test basic parameter mapping for OpenAI."""
        from litellm.types.containers.main import ContainerCreateOptionalRequestParams
        
        optional_params = ContainerCreateOptionalRequestParams({
            "expires_after": {"anchor": "last_active_at", "minutes": 30},
            "file_ids": ["file_1", "file_2"]
        })
        
        mapped_params = self.config.map_openai_params(optional_params, drop_params=False)
        
        assert mapped_params["expires_after"]["minutes"] == 30
        assert mapped_params["file_ids"] == ["file_1", "file_2"]

    def test_validate_environment(self):
        """Test environment validation adds proper headers."""
        headers = {}
        api_key = "sk-test123"
        
        validated_headers = self.config.validate_environment(
            headers=headers,
            api_key=api_key
        )
        
        assert "Authorization" in validated_headers
        assert validated_headers["Authorization"] == f"Bearer {api_key}"
        # Note: Content-Type is not added by validate_environment method

    def test_get_complete_url(self):
        """Test complete URL generation."""
        api_base = "https://api.openai.com/v1"
        litellm_params = {}
        
        url = self.config.get_complete_url(
            api_base=api_base,
            litellm_params=litellm_params
        )
        
        assert url == "https://api.openai.com/v1/containers"

    def test_get_complete_url_with_custom_base(self):
        """Test complete URL generation with custom API base."""
        api_base = "https://custom.openai.com/v1"
        litellm_params = {}
        
        url = self.config.get_complete_url(
            api_base=api_base,
            litellm_params=litellm_params
        )
        
        assert url == "https://custom.openai.com/v1/containers"

    def test_transform_container_create_request(self):
        """Test container create request transformation."""
        from litellm.types.router import GenericLiteLLMParams
        
        litellm_params = GenericLiteLLMParams()
        headers = {"Authorization": "Bearer sk-test123"}
        name = "Test Container"
        container_create_optional_request_params = {
            "expires_after": {"anchor": "last_active_at", "minutes": 20},
            "file_ids": ["file_123"]
        }
        
        data = self.config.transform_container_create_request(
            name=name,
            container_create_optional_request_params=container_create_optional_request_params,
            litellm_params=litellm_params,
            headers=headers
        )
        
        assert data["name"] == name
        assert data["expires_after"] == container_create_optional_request_params["expires_after"]
        assert data["file_ids"] == container_create_optional_request_params["file_ids"]

    def test_transform_container_create_response(self):
        """Test container create response transformation."""
        # Mock HTTP response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "cntr_123456",
            "object": "container",
            "created_at": 1747857508,
            "status": "running",
            "expires_after": {"anchor": "last_active_at", "minutes": 20},
            "last_active_at": 1747857508,
            "name": "Test Container"
        }
        
        container = self.config.transform_container_create_response(
            raw_response=mock_response,
            logging_obj=self.logging_obj
        )
        
        assert isinstance(container, ContainerObject)
        assert container.id == "cntr_123456"
        assert container.name == "Test Container"
        assert container.status == "running"
        assert container.object == "container"

    def test_transform_container_list_request(self):
        """Test container list request transformation."""
        api_base = "https://api.openai.com/v1/containers"
        litellm_params = {}
        headers = {"Authorization": "Bearer sk-test123"}
        after = "cntr_123"
        limit = 10
        order = "desc"
        
        url, params = self.config.transform_container_list_request(
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
            after=after,
            limit=limit,
            order=order
        )
        
        assert url == api_base
        assert params["after"] == after
        assert params["limit"] == str(limit)  # Should be string for query params
        assert params["order"] == order

    def test_transform_container_list_response(self):
        """Test container list response transformation."""
        # Mock HTTP response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "id": "cntr_1",
                    "object": "container",
                    "created_at": 1747857508,
                    "status": "running",
                    "expires_after": {"anchor": "last_active_at", "minutes": 20},
                    "last_active_at": 1747857508,
                    "name": "Container 1"
                },
                {
                    "id": "cntr_2",
                    "object": "container", 
                    "created_at": 1747857600,
                    "status": "running",
                    "expires_after": {"anchor": "last_active_at", "minutes": 15},
                    "last_active_at": 1747857600,
                    "name": "Container 2"
                }
            ],
            "first_id": "cntr_1",
            "last_id": "cntr_2",
            "has_more": False
        }
        
        container_list = self.config.transform_container_list_response(
            raw_response=mock_response,
            logging_obj=self.logging_obj
        )
        
        assert isinstance(container_list, ContainerListResponse)
        assert len(container_list.data) == 2
        assert container_list.first_id == "cntr_1"
        assert container_list.last_id == "cntr_2"
        assert container_list.has_more == False

    def test_transform_container_retrieve_request(self):
        """Test container retrieve request transformation."""
        container_id = "cntr_test123"
        api_base = "https://api.openai.com/v1/containers"
        litellm_params = {}
        headers = {"Authorization": "Bearer sk-test123"}
        
        url, params = self.config.transform_container_retrieve_request(
            container_id=container_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers
        )
        
        assert url == f"{api_base}/{container_id}"
        assert params == {}  # No query params for retrieve

    def test_transform_container_retrieve_response(self):
        """Test container retrieve response transformation."""
        # Mock HTTP response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "cntr_retrieve_123",
            "object": "container",
            "created_at": 1747857508,
            "status": "running",
            "expires_after": {"anchor": "last_active_at", "minutes": 20},
            "last_active_at": 1747857508,
            "name": "Retrieved Container"
        }
        
        container = self.config.transform_container_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.logging_obj
        )
        
        assert isinstance(container, ContainerObject)
        assert container.id == "cntr_retrieve_123"
        assert container.name == "Retrieved Container"

    def test_transform_container_delete_request(self):
        """Test container delete request transformation."""
        container_id = "cntr_delete_123"
        api_base = "https://api.openai.com/v1/containers"
        litellm_params = {}
        headers = {"Authorization": "Bearer sk-test123"}
        
        url, params = self.config.transform_container_delete_request(
            container_id=container_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers
        )
        
        assert url == f"{api_base}/{container_id}"
        assert params == {}  # No query params for delete

    def test_transform_container_delete_response(self):
        """Test container delete response transformation."""
        # Mock HTTP response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "cntr_delete_123",
            "object": "container.deleted",
            "deleted": True
        }
        
        delete_result = self.config.transform_container_delete_response(
            raw_response=mock_response,
            logging_obj=self.logging_obj
        )
        
        assert isinstance(delete_result, DeleteContainerResult)
        assert delete_result.id == "cntr_delete_123"
        assert delete_result.object == "container.deleted"
        assert delete_result.deleted == True

    def test_get_error_class(self):
        """Test error class handling."""
        import httpx
        from litellm.llms.base_llm.chat.transformation import BaseLLMException
        
        with pytest.raises(BaseLLMException) as exc_info:
            self.config.get_error_class(
                error_message="Test error",
                status_code=400,
                headers={}
            )
        
        assert "Test error" in str(exc_info.value)

    def test_transform_with_none_optional_params(self):
        """Test transformation handles None optional parameters correctly."""
        from litellm.types.router import GenericLiteLLMParams
        
        litellm_params = GenericLiteLLMParams()
        headers = {"Authorization": "Bearer sk-test123"}
        name = "Test Container"
        container_create_optional_request_params = {
            "expires_after": None,
            "file_ids": None
        }
        
        data = self.config.transform_container_create_request(
            name=name,
            container_create_optional_request_params=container_create_optional_request_params,
            litellm_params=litellm_params,
            headers=headers
        )
        
        assert data["name"] == name
        # None values should be included as None
        assert data["expires_after"] is None
        assert data["file_ids"] is None

    def test_container_create_response_includes_cost(self):
        """Test that container create response includes code interpreter cost calculation."""
        # Force use of local model cost map for CI/CD consistency
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        
        from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import StandardBuiltInToolCostTracking
        
        # Mock HTTP response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "cntr_cost_test",
            "object": "container",
            "created_at": 1747857508,
            "status": "running",
            "expires_after": {"anchor": "last_active_at", "minutes": 20},
            "last_active_at": 1747857508,
            "name": "Cost Test Container"
        }
        
        # Transform the response
        container = self.config.transform_container_create_response(
            raw_response=mock_response,
            logging_obj=self.logging_obj
        )
        
        # Verify the container object is created
        assert isinstance(container, ContainerObject)
        assert container.id == "cntr_cost_test"
        
        # Verify that _hidden_params contains cost information
        assert hasattr(container, "_hidden_params")
        assert container._hidden_params is not None
        assert "additional_headers" in container._hidden_params
        assert "llm_provider-x-litellm-response-cost" in container._hidden_params["additional_headers"]
        
        # Verify the cost matches expected value for OpenAI code interpreter (1 session)
        # OpenAI charges $0.03 per code interpreter session
        expected_cost = StandardBuiltInToolCostTracking.get_cost_for_code_interpreter(
            sessions=1,
            provider="openai"
        )
        actual_cost = container._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"]
        
        assert actual_cost == expected_cost
        assert actual_cost == 0.03  # OpenAI code interpreter costs $0.03 per session
