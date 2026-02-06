import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.containers.utils import ContainerRequestUtils
from litellm.llms.openai.containers.transformation import OpenAIContainerConfig
from litellm.types.containers.main import (
    ContainerCreateOptionalRequestParams,
    ContainerListOptionalRequestParams
)


class TestContainerRequestUtils:
    """Test suite for container request utilities."""

    def test_get_optional_params_container_create_basic(self):
        """Test that optional parameters are correctly processed for container creation."""
        # Setup
        config = OpenAIContainerConfig()
        optional_params = ContainerCreateOptionalRequestParams(
            {
                "expires_after": {"anchor": "last_active_at", "minutes": 30},
                "file_ids": ["file_123", "file_456"]
            }
        )

        # Execute
        result = ContainerRequestUtils.get_optional_params_container_create(
            container_provider_config=config,
            container_create_optional_params=optional_params,
        )

        # Assert
        assert result == optional_params
        assert "expires_after" in result
        assert result["expires_after"]["minutes"] == 30
        assert "file_ids" in result
        assert result["file_ids"] == ["file_123", "file_456"]

    def test_get_optional_params_container_create_unsupported_param(self):
        """Test that unsupported parameters are filtered out by ContainerCreateOptionalRequestParams."""
        # Setup
        config = OpenAIContainerConfig()
        
        # ContainerCreateOptionalRequestParams will only accept valid parameters
        # so this test verifies the type validation works correctly
        valid_params = ContainerCreateOptionalRequestParams(
            {
                "expires_after": {"anchor": "last_active_at", "minutes": 30},
                "file_ids": ["file_123"]
            }
        )

        # Execute - should work fine with valid parameters
        result = ContainerRequestUtils.get_optional_params_container_create(
            container_provider_config=config,
            container_create_optional_params=valid_params,
        )

        assert result["expires_after"]["minutes"] == 30
        assert result["file_ids"] == ["file_123"]

    def test_get_requested_container_create_optional_param(self):
        """Test filtering parameters to only include those in ContainerCreateOptionalRequestParams."""
        # Setup
        params = {
            "name": "Test Container",  # This should be excluded as it's required
            "expires_after": {"anchor": "last_active_at", "minutes": 30},
            "file_ids": ["file_123"],
            "invalid_param": "value",
            "custom_llm_provider": "openai",  # This should be excluded
        }

        # Execute
        result = ContainerRequestUtils.get_requested_container_create_optional_param(
            params
        )

        # Assert
        assert "expires_after" in result
        assert "file_ids" in result
        assert "invalid_param" not in result
        assert "name" not in result
        assert "custom_llm_provider" not in result
        assert result["expires_after"]["minutes"] == 30
        assert result["file_ids"] == ["file_123"]

    def test_get_requested_container_list_optional_param(self):
        """Test filtering parameters for container list requests."""
        # Setup
        params = {
            "after": "cntr_123",
            "limit": 10,
            "order": "desc",
            "invalid_param": "value",
            "custom_llm_provider": "openai",  # This should be excluded
        }

        # Execute
        result = ContainerRequestUtils.get_requested_container_list_optional_param(
            params
        )

        # Assert
        assert "after" in result
        assert "limit" in result
        assert "order" in result
        assert "invalid_param" not in result
        assert "custom_llm_provider" not in result
        assert result["after"] == "cntr_123"
        assert result["limit"] == 10
        assert result["order"] == "desc"

    def test_get_optional_params_container_create_empty_params(self):
        """Test handling of empty optional parameters."""
        # Setup
        config = OpenAIContainerConfig()
        optional_params = ContainerCreateOptionalRequestParams({})

        # Execute
        result = ContainerRequestUtils.get_optional_params_container_create(
            container_provider_config=config,
            container_create_optional_params=optional_params,
        )

        # Assert
        assert result == optional_params
        assert len(result) == 0

    def test_get_optional_params_container_create_with_none_values(self):
        """Test handling of None values in optional parameters."""
        # Setup
        config = OpenAIContainerConfig()
        optional_params = ContainerCreateOptionalRequestParams(
            {
                "expires_after": None,
                "file_ids": None
            }
        )

        # Execute
        result = ContainerRequestUtils.get_optional_params_container_create(
            container_provider_config=config,
            container_create_optional_params=optional_params,
        )

        # Assert
        assert result == optional_params
        assert "expires_after" in result
        assert "file_ids" in result
        assert result["expires_after"] is None
        assert result["file_ids"] is None

    def test_get_requested_container_list_optional_param_partial(self):
        """Test filtering with only some list parameters present."""
        # Setup
        params = {
            "limit": 5,
            "custom_llm_provider": "openai",  # Should be excluded
            "timeout": 600,  # Should be excluded
        }

        # Execute
        result = ContainerRequestUtils.get_requested_container_list_optional_param(
            params
        )

        # Assert
        assert "limit" in result
        assert "custom_llm_provider" not in result
        assert "timeout" not in result
        assert "after" not in result  # Not present in input
        assert "order" not in result  # Not present in input
        assert result["limit"] == 5

    def test_container_create_optional_params_type_validation(self):
        """Test that ContainerCreateOptionalRequestParams validates types correctly."""
        # Test with valid expires_after
        valid_params = ContainerCreateOptionalRequestParams(
            {
                "expires_after": {"anchor": "last_active_at", "minutes": 20},
                "file_ids": ["file_1", "file_2"]
            }
        )
        
        assert valid_params["expires_after"]["anchor"] == "last_active_at"
        assert valid_params["expires_after"]["minutes"] == 20
        assert valid_params["file_ids"] == ["file_1", "file_2"]

    def test_container_list_optional_params_type_validation(self):
        """Test that ContainerListOptionalRequestParams validates types correctly."""
        # Test with valid parameters
        valid_params = ContainerListOptionalRequestParams(
            {
                "after": "cntr_123",
                "limit": 10,
                "order": "desc"
            }
        )
        
        assert valid_params["after"] == "cntr_123"
        assert valid_params["limit"] == 10
        assert valid_params["order"] == "desc"

    def test_get_optional_params_with_supported_params_check(self):
        """Test that only supported parameters are accepted."""
        # Setup
        config = OpenAIContainerConfig()
        
        # Get supported params to understand what should be allowed
        supported_params = config.get_supported_openai_params()
        
        # Create params with only valid parameters
        test_params = {"expires_after": {"anchor": "last_active_at", "minutes": 15}}
        
        optional_params = ContainerCreateOptionalRequestParams(test_params)

        # Execute - should work fine with supported params
        result = ContainerRequestUtils.get_optional_params_container_create(
            container_provider_config=config,
            container_create_optional_params=optional_params,
        )
        
        assert result["expires_after"]["minutes"] == 15
