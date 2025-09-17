"""
Test router filtering of inactive models
"""
import os
import sys
import uuid

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm import Router
from litellm.types.router import DeploymentTypedDict


class TestRouterInactiveFiltering:
    def test_get_all_deployments_filters_inactive_models(self):
        """Test that _get_all_deployments filters out inactive models"""
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-3.5-turbo",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "api_key": "test-key",
                    },
                    "is_active": True,
                },
                {
                    "model_name": "gpt-4",
                    "litellm_params": {
                        "model": "gpt-4",
                        "api_key": "test-key",
                    },
                    "is_active": False,  # This model is inactive
                },
                {
                    "model_name": "claude-instant",
                    "litellm_params": {
                        "model": "claude-instant-1.2",
                        "api_key": "test-key",
                    },
                    # is_active not specified, should default to True
                },
            ]
        )
        
        # Test getting deployments for gpt-3.5-turbo (active)
        deployments = router._get_all_deployments("gpt-3.5-turbo")
        assert len(deployments) == 1
        assert deployments[0]["model_name"] == "gpt-3.5-turbo"
        
        # Test getting deployments for gpt-4 (inactive)
        deployments = router._get_all_deployments("gpt-4")
        assert len(deployments) == 0  # Should be filtered out
        
        # Test getting deployments for claude-instant (default active)
        deployments = router._get_all_deployments("claude-instant")
        assert len(deployments) == 1
        assert deployments[0]["model_name"] == "claude-instant"

    def test_get_deployment_by_litellm_model_filters_inactive(self):
        """Test that _get_deployment_by_litellm_model filters out inactive models"""
        router = Router(
            model_list=[
                {
                    "model_name": "active-model",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "api_key": "test-key",
                    },
                    "is_active": True,
                },
                {
                    "model_name": "inactive-model",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",  # Same underlying model
                        "api_key": "test-key",
                    },
                    "is_active": False,
                },
            ]
        )
        
        deployments = router._get_deployment_by_litellm_model("gpt-3.5-turbo")
        
        # Should only return the active deployment
        assert len(deployments) == 1
        assert deployments[0]["model_name"] == "active-model"
        assert deployments[0].get("is_active", True) == True

    def test_get_available_deployment_skips_inactive(self):
        """Test that get_available_deployment doesn't select inactive models"""
        router = Router(
            model_list=[
                {
                    "model_name": "model-group",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "api_key": "test-key",
                    },
                    "is_active": False,  # Only model is inactive
                },
            ],
            routing_strategy="simple-shuffle",
        )
        
        # Should raise error because no active deployments
        with pytest.raises(litellm.BadRequestError) as exc_info:
            router.get_available_deployment(
                model="model-group",
                messages=[{"role": "user", "content": "test"}]
            )
        
        assert "no healthy deployments" in str(exc_info.value.message).lower()

    def test_mixed_active_inactive_models(self):
        """Test router handles mix of active and inactive models correctly"""
        router = Router(
            model_list=[
                {
                    "model_name": "model-group",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "api_key": "test-key-1",
                    },
                    "model_info": {"id": "model-1"},
                    "is_active": True,
                },
                {
                    "model_name": "model-group",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "api_key": "test-key-2",
                    },
                    "model_info": {"id": "model-2"},
                    "is_active": False,
                },
                {
                    "model_name": "model-group",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "api_key": "test-key-3",
                    },
                    "model_info": {"id": "model-3"},
                    "is_active": True,
                },
            ],
            routing_strategy="simple-shuffle",
        )
        
        # Get available deployments - should only get active ones
        deployments = router._get_all_deployments("model-group")
        
        assert len(deployments) == 2  # Only 2 active models
        model_ids = [d["model_info"]["id"] for d in deployments]
        assert "model-1" in model_ids
        assert "model-2" not in model_ids  # Inactive, should be filtered
        assert "model-3" in model_ids

    def test_backward_compatibility_is_active_defaults_true(self):
        """Test that models without is_active field default to active"""
        router = Router(
            model_list=[
                {
                    "model_name": "legacy-model",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "api_key": "test-key",
                    },
                    # No is_active field - should default to True
                }
            ]
        )
        
        deployments = router._get_all_deployments("legacy-model")
        assert len(deployments) == 1  # Should not be filtered out
        
        # Verify it's treated as active
        deployment = router.get_available_deployment(
            model="legacy-model",
            messages=[{"role": "user", "content": "test"}]
        )
        assert deployment is not None

    def test_explicitly_active_true_models_work(self):
        """Test that models with is_active=True work correctly"""
        router = Router(
            model_list=[
                {
                    "model_name": "explicit-active",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "api_key": "test-key",
                    },
                    "is_active": True,  # Explicitly set to True
                }
            ]
        )
        
        deployments = router._get_all_deployments("explicit-active")
        assert len(deployments) == 1
        assert deployments[0]["model_name"] == "explicit-active"

    def test_all_models_inactive_error(self):
        """Test error when all models in a group are inactive"""
        router = Router(
            model_list=[
                {
                    "model_name": "all-inactive",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "api_key": "test-key-1",
                    },
                    "is_active": False,
                },
                {
                    "model_name": "all-inactive",
                    "litellm_params": {
                        "model": "gpt-4",
                        "api_key": "test-key-2",
                    },
                    "is_active": False,
                },
            ]
        )
        
        deployments = router._get_all_deployments("all-inactive")
        assert len(deployments) == 0  # All filtered out
        
        # Should raise error when trying to get available deployment
        with pytest.raises(litellm.BadRequestError) as exc_info:
            router.get_available_deployment(
                model="all-inactive",
                messages=[{"role": "user", "content": "test"}]
            )
        
        assert "no healthy deployments" in str(exc_info.value.message).lower()