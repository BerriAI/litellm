import sys
import os
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from litellm import Router


class TestRouterIndexManagement:
    """Test cases for router index management functions"""

    @pytest.fixture
    def router(self):
        """Create a router instance for testing"""
        return Router(model_list=[])

    def test_update_deployment_indices_after_removal(self, router):
        """Test _update_deployment_indices_after_removal function"""
        # Setup: Add models to router with proper structure
        router.model_list = [
            {"model": "test1", "model_info": {"id": "model-1"}}, 
            {"model": "test2", "model_info": {"id": "model-2"}}, 
            {"model": "test3", "model_info": {"id": "model-3"}}
        ]
        router.model_id_to_deployment_index_map = {"model-1": 0, "model-2": 1, "model-3": 2}

        # Test: Remove model-2 (index 1)
        router._update_deployment_indices_after_removal(model_id="model-2", removal_idx=1)

        # Verify: model-2 is removed from index
        assert "model-2" not in router.model_id_to_deployment_index_map
        # Verify: model-3 index is updated (2 -> 1)
        assert router.model_id_to_deployment_index_map["model-3"] == 1
        # Verify: model-1 index remains unchanged
        assert router.model_id_to_deployment_index_map["model-1"] == 0

    def test_build_model_id_to_deployment_index_map(self, router):
        """Test _build_model_id_to_deployment_index_map function"""
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
                "model_info": {"id": "model-1"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4"},
                "model_info": {"id": "model-2"},
            },
        ]

        # Test: Build index from model list
        router._build_model_id_to_deployment_index_map(model_list)

        # Verify: model_list is populated
        assert len(router.model_list) == 2
        # Verify: model_id_to_deployment_index_map is correctly built
        assert router.model_id_to_deployment_index_map["model-1"] == 0
        assert router.model_id_to_deployment_index_map["model-2"] == 1
