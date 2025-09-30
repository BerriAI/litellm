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

    def test_add_model_to_list_and_index_map_from_model_info(self, router):
        """Test _add_model_to_list_and_index_map extracting model_id from model_info"""
        # Setup: Empty router
        router.model_list = []
        router.model_id_to_deployment_index_map = {}
        
        # Test: Add model without explicit model_id
        model = {"model": "test-model", "model_info": {"id": "model-info-id"}}
        router._add_model_to_list_and_index_map(model=model)
        
        # Verify: Model added to list
        assert len(router.model_list) == 1
        assert router.model_list[0] == model
        
        # Verify: Index map uses model_info.id
        assert router.model_id_to_deployment_index_map["model-info-id"] == 0


    def test_add_model_to_list_and_index_map_multiple_models(self, router):
        """Test _add_model_to_list_and_index_map with multiple models to verify indexing"""
        # Setup: Empty router
        router.model_list = []
        router.model_id_to_deployment_index_map = {}
        
        # Test: Add multiple models
        model1 = {"model": "model1", "model_info": {"id": "id-1"}}
        model2 = {"model": "model2", "model_info": {"id": "id-2"}}
        model3 = {"model": "model3", "model_info": {"id": "id-3"}}
        
        router._add_model_to_list_and_index_map(model=model1, model_id="id-1")
        router._add_model_to_list_and_index_map(model=model2, model_id="id-2")
        router._add_model_to_list_and_index_map(model=model3, model_id="id-3")
        
        # Verify: All models added to list
        assert len(router.model_list) == 3
        assert router.model_list[0] == model1
        assert router.model_list[1] == model2
        assert router.model_list[2] == model3
        
        # Verify: Correct indices in map
        assert router.model_id_to_deployment_index_map["id-1"] == 0
        assert router.model_id_to_deployment_index_map["id-2"] == 1
        assert router.model_id_to_deployment_index_map["id-3"] == 2

    def test_has_model_id(self, router):
        """Test has_model_id function for O(1) membership check"""
        # Setup: Add models to router
        router.model_list = [
            {"model": "test1", "model_info": {"id": "model-1"}}, 
            {"model": "test2", "model_info": {"id": "model-2"}}, 
            {"model": "test3", "model_info": {"id": "model-3"}}
        ]
        router.model_id_to_deployment_index_map = {"model-1": 0, "model-2": 1, "model-3": 2}

        # Test: Check existing model IDs
        assert router.has_model_id("model-1") == True
        assert router.has_model_id("model-2") == True
        assert router.has_model_id("model-3") == True

        # Test: Check non-existing model IDs
        assert router.has_model_id("non-existent") == False
        assert router.has_model_id("") == False
        assert router.has_model_id("model-4") == False

        # Test: Empty router
        empty_router = Router(model_list=[])
        assert empty_router.has_model_id("any-id") == False
