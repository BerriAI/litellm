import os
import sys
from unittest.mock import MagicMock

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.azure.containers.transformation import AzureOpenAIContainerConfig
from litellm.llms.openai.containers.transformation import OpenAIContainerConfig
from litellm.types.containers.main import (
    ContainerFileListResponse,
    ContainerListResponse,
    ContainerObject,
    DeleteContainerResult,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager


class TestAzureContainerConfigInheritance:
    """Test that AzureOpenAIContainerConfig properly inherits from OpenAI."""

    def test_inherits_from_openai(self):
        config = AzureOpenAIContainerConfig()
        assert isinstance(config, OpenAIContainerConfig)

    def test_get_supported_openai_params(self):
        config = AzureOpenAIContainerConfig()
        supported_params = config.get_supported_openai_params()
        assert "name" in supported_params
        assert "expires_after" in supported_params
        assert "file_ids" in supported_params


class TestAzureContainerValidateEnvironment:
    """Test Azure-specific authentication headers."""

    def test_validate_environment_with_api_key(self):
        config = AzureOpenAIContainerConfig()
        headers = {}
        result = config.validate_environment(headers=headers, api_key="test-azure-key")
        assert "api-key" in result
        assert result["api-key"] == "test-azure-key"

    def test_validate_environment_does_not_use_bearer_token(self):
        config = AzureOpenAIContainerConfig()
        headers = {}
        result = config.validate_environment(headers=headers, api_key="test-azure-key")
        assert "Authorization" not in result

    def test_validate_environment_preserves_existing_api_key_header(self):
        config = AzureOpenAIContainerConfig()
        headers = {"api-key": "existing-key"}
        result = config.validate_environment(headers=headers, api_key="new-key")
        assert result["api-key"] == "existing-key"


class TestAzureContainerGetCompleteUrl:
    """Test Azure-specific URL construction."""

    def test_get_complete_url_constructs_azure_url(self):
        config = AzureOpenAIContainerConfig()
        api_base = "https://my-resource.openai.azure.com"
        litellm_params = {"api_version": "2024-12-01-preview"}

        url = config.get_complete_url(
            api_base=api_base,
            litellm_params=litellm_params,
        )

        assert "my-resource.openai.azure.com" in url
        assert "/openai/v1/containers" in url or "/openai/containers" in url
        assert "api-version" in url

    def test_get_complete_url_uses_default_api_version(self):
        config = AzureOpenAIContainerConfig()
        api_base = "https://my-resource.openai.azure.com"
        litellm_params = {}

        url = config.get_complete_url(
            api_base=api_base,
            litellm_params=litellm_params,
        )

        assert "api-version" in url

    def test_get_complete_url_differs_from_openai(self):
        azure_config = AzureOpenAIContainerConfig()
        openai_config = OpenAIContainerConfig()

        azure_url = azure_config.get_complete_url(
            api_base="https://my-resource.openai.azure.com",
            litellm_params={},
        )
        openai_url = openai_config.get_complete_url(
            api_base="https://api.openai.com/v1",
            litellm_params={},
        )

        assert azure_url != openai_url
        assert "azure" in azure_url
        assert "api-version" in azure_url


class TestAzureContainerTransformations:
    """Test that request/response transformations work for Azure (inherited from OpenAI)."""

    def setup_method(self):
        self.config = AzureOpenAIContainerConfig()
        self.logging_obj = LiteLLMLogging(
            model="",
            messages=[],
            stream=False,
            call_type="create_container",
            start_time=None,
            litellm_call_id="test_call_id",
            function_id="test_function_id",
        )

    def test_transform_container_create_request(self):
        litellm_params = GenericLiteLLMParams()
        headers = {"api-key": "test-azure-key"}
        data = self.config.transform_container_create_request(
            name="Test Container",
            container_create_optional_request_params={
                "expires_after": {"anchor": "last_active_at", "minutes": 20},
            },
            litellm_params=litellm_params,
            headers=headers,
        )
        assert data["name"] == "Test Container"
        assert data["expires_after"]["minutes"] == 20

    def test_transform_container_create_response(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "cntr_azure_123",
            "object": "container",
            "created_at": 1747857508,
            "status": "running",
            "expires_after": {"anchor": "last_active_at", "minutes": 20},
            "last_active_at": 1747857508,
            "name": "Azure Container",
        }
        container = self.config.transform_container_create_response(
            raw_response=mock_response,
            logging_obj=self.logging_obj,
        )
        assert isinstance(container, ContainerObject)
        assert container.id == "cntr_azure_123"
        assert container.name == "Azure Container"

    def test_transform_container_create_response_uses_azure_provider_for_cost(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "cntr_azure_cost",
            "object": "container",
            "created_at": 1747857508,
            "status": "running",
            "expires_after": {"anchor": "last_active_at", "minutes": 20},
            "last_active_at": 1747857508,
            "name": "Azure Cost Test",
        }
        from unittest.mock import patch

        with patch(
            "litellm.llms.azure.containers.transformation.StandardBuiltInToolCostTracking.get_cost_for_code_interpreter"
        ) as mock_cost:
            mock_cost.return_value = 0.03
            self.config.transform_container_create_response(
                raw_response=mock_response,
                logging_obj=self.logging_obj,
            )
            mock_cost.assert_called_once_with(sessions=1, provider="azure")

    def test_transform_container_list_response(self):
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
                    "name": "Container 1",
                }
            ],
            "first_id": "cntr_1",
            "last_id": "cntr_1",
            "has_more": False,
        }
        result = self.config.transform_container_list_response(
            raw_response=mock_response,
            logging_obj=self.logging_obj,
        )
        assert isinstance(result, ContainerListResponse)
        assert len(result.data) == 1

    def test_transform_container_file_list_request(self):
        api_base = "https://my-resource.openai.azure.com/openai/v1/containers"
        url, params = self.config.transform_container_file_list_request(
            container_id="cntr_123",
            api_base=api_base,
            litellm_params={},
            headers={"api-key": "test"},
            limit=10,
        )
        assert "cntr_123/files" in url
        assert params["limit"] == "10"

    def test_transform_container_file_content_request(self):
        api_base = "https://my-resource.openai.azure.com/openai/v1/containers"
        url, params = self.config.transform_container_file_content_request(
            container_id="cntr_123",
            file_id="file_456",
            api_base=api_base,
            litellm_params={},
            headers={"api-key": "test"},
        )
        assert "cntr_123/files/file_456/content" in url

    def test_transform_container_file_content_response(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.content = b"file content bytes"
        result = self.config.transform_container_file_content_response(
            raw_response=mock_response,
            logging_obj=self.logging_obj,
        )
        assert result == b"file content bytes"

    def test_transform_container_delete_response(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "cntr_delete_123",
            "object": "container.deleted",
            "deleted": True,
        }
        result = self.config.transform_container_delete_response(
            raw_response=mock_response,
            logging_obj=self.logging_obj,
        )
        assert isinstance(result, DeleteContainerResult)
        assert result.deleted is True


class TestAzureContainerProviderRegistration:
    """Test that Azure is properly registered as a container provider."""

    def test_provider_config_manager_returns_azure_config(self):
        config = ProviderConfigManager.get_provider_container_config(
            provider=litellm.LlmProviders.AZURE,
        )
        assert config is not None
        assert isinstance(config, AzureOpenAIContainerConfig)

    def test_provider_config_manager_returns_none_for_unsupported(self):
        config = ProviderConfigManager.get_provider_container_config(
            provider=litellm.LlmProviders.ANTHROPIC,
        )
        assert config is None
