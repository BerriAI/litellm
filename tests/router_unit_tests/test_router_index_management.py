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

    def test_remove_deployment_from_index(self, router):
        """Test _remove_deployment_from_index function"""
        # Setup: Add models to router
        router.model_list = [{"model": "test1"}, {"model": "test2"}, {"model": "test3"}]
        router.model_index = {"model-1": 0, "model-2": 1, "model-3": 2}

        # Test: Remove model-2 (index 1)
        router._remove_deployment_from_index("model-2", 1)

        # Verify: model-2 is removed from index
        assert "model-2" not in router.model_index
        # Verify: model-3 index is updated (2 -> 1)
        assert router.model_index["model-3"] == 1
        # Verify: model-1 index remains unchanged
        assert router.model_index["model-1"] == 0

    def test_build_model_index_from_list(self, router):
        """Test _build_model_index_from_list function"""
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
        router._build_model_index_from_list(model_list)

        # Verify: model_list is populated
        assert len(router.model_list) == 2
        # Verify: model_index is correctly built
        assert router.model_index["model-1"] == 0
        assert router.model_index["model-2"] == 1