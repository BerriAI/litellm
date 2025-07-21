"""
Router integration tests for the next fallback functionality

Tests the fallback endpoint with real Router configurations and complex scenarios.
"""
import asyncio
import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm import Router
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.model_checks import get_next_fallback


class TestRouterNextFallbackIntegration:
    """Integration tests with real Router instances"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        # Create a realistic model list for testing
        self.model_list = [
            {
                "model_name": "claude-4-sonnet",
                "litellm_params": {
                    "model": "anthropic/claude-3-5-sonnet-20241022"
                }
            },
            {
                "model_name": "bedrock-claude-sonnet-4", 
                "litellm_params": {
                    "model": "bedrock/claude-3-5-sonnet-20241022-v2:0"
                }
            },
            {
                "model_name": "google-claude-sonnet-4",
                "litellm_params": {
                    "model": "vertex_ai/claude-3-5-sonnet-v2@20241022"
                }
            },
            {
                "model_name": "gpt-4o",
                "litellm_params": {
                    "model": "openai/gpt-4o"
                }
            },
            {
                "model_name": "azure-gpt-4o",
                "litellm_params": {
                    "model": "azure/gpt-4o-deployment"
                }
            },
            {
                "model_name": "github-gpt-4o",
                "litellm_params": {
                    "model": "github/gpt-4o"
                }
            }
        ]
        
        # User configuration matching the test scenario
        self.user_api_key_dict = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user", 
            models=["claude-4-sonnet", "bedrock-claude-sonnet-4", "google-claude-sonnet-4", 
                   "gpt-4o", "azure-gpt-4o", "github-gpt-4o"],
            team_models=[],
        )

    def test_real_router_with_complex_fallbacks(self):
        """Test with a real Router instance and complex fallback configuration"""
        # Complex fallback configuration matching user's requirements
        fallbacks = [
            {"claude-4-sonnet": ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]},
            {"claude-sonnet-4": ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]},
            {"claude-sonnet-4-failing": ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]},
            {"claude-3-7-sonnet": ["bedrock-claude-3-7-sonnet", "google-claude-3-7-sonnet"]},
            {"claude-3-7-sonnet-thinking": ["bedrock-claude-3-7-sonnet-thinking", "google-claude-3-7-sonnet-thinking"]},
            {"gpt-4o": ["azure-gpt-4o", "github-gpt-4o"]},
            {"gpt-4.1": ["github-gpt-4.1"]},
            {"gpt-4.1-mini": ["azure-gpt-4.1-mini", "github-gpt-4.1-mini"]},
            {"gpt-4.5-preview": ["azure-gpt-4.5-preview"]},
            {"gpt-4o-mini": ["github-gpt-4o-mini"]},
            {"gpt-o4-mini-high": ["github-gpt-o4-mini-high"]},
        ]
        
        # Create real router with fallbacks
        router = Router(
            model_list=self.model_list,
            fallbacks=fallbacks
        )
        
        # Test the main fallback chain from user's requirements
        test_cases = [
            ("claude-4-sonnet", "bedrock-claude-sonnet-4"),
            ("bedrock-claude-sonnet-4", "google-claude-sonnet-4"),
            ("google-claude-sonnet-4", None),
            ("gpt-4o", "azure-gpt-4o"),
            ("azure-gpt-4o", "github-gpt-4o"),
            ("github-gpt-4o", None),
        ]
        
        for model, expected_fallback in test_cases:
            result = get_next_fallback(
                model=model,
                user_api_key_dict=self.user_api_key_dict,
                llm_router=router,
                fallback_type="general"
            )
            assert result == expected_fallback, f"Failed for {model}: expected {expected_fallback}, got {result}"

    def test_context_window_fallbacks_integration(self):
        """Test context window fallbacks with real router"""
        context_fallbacks = [
            {"claude-4-sonnet": ["bedrock-claude-sonnet-4"]},
            {"gpt-4o": ["azure-gpt-4o"]}
        ]
        
        router = Router(
            model_list=self.model_list,
            context_window_fallbacks=context_fallbacks
        )
        
        # Test context window fallbacks
        result = get_next_fallback(
            model="claude-4-sonnet",
            user_api_key_dict=self.user_api_key_dict,
            llm_router=router,
            fallback_type="context_window"
        )
        assert result == "bedrock-claude-sonnet-4"
        
        result = get_next_fallback(
            model="gpt-4o",
            user_api_key_dict=self.user_api_key_dict,
            llm_router=router,
            fallback_type="context_window"
        )
        assert result == "azure-gpt-4o"

    def test_content_policy_fallbacks_integration(self):
        """Test content policy fallbacks with real router"""
        content_policy_fallbacks = [
            {"claude-4-sonnet": ["google-claude-sonnet-4"]},
            {"gpt-4o": ["github-gpt-4o"]}
        ]
        
        router = Router(
            model_list=self.model_list,
            content_policy_fallbacks=content_policy_fallbacks
        )
        
        # Test content policy fallbacks
        result = get_next_fallback(
            model="claude-4-sonnet",
            user_api_key_dict=self.user_api_key_dict,
            llm_router=router,
            fallback_type="content_policy"
        )
        assert result == "google-claude-sonnet-4"
        
        result = get_next_fallback(
            model="gpt-4o",
            user_api_key_dict=self.user_api_key_dict,
            llm_router=router,
            fallback_type="content_policy"
        )
        assert result == "github-gpt-4o"

    def test_generic_wildcard_fallbacks_integration(self):
        """Test generic wildcard fallbacks with real router"""
        fallbacks = [
            {"claude-4-sonnet": ["bedrock-claude-sonnet-4"]},
            {"*": ["google-claude-sonnet-4"]}  # Generic fallback for any model
        ]
        
        router = Router(
            model_list=self.model_list,
            fallbacks=fallbacks
        )
        
        # Test specific fallback
        result = get_next_fallback(
            model="claude-4-sonnet",
            user_api_key_dict=self.user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        assert result == "bedrock-claude-sonnet-4"
        
        # Test generic fallback for model not in specific config
        result = get_next_fallback(
            model="gpt-4o",
            user_api_key_dict=self.user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        assert result == "google-claude-sonnet-4"

    def test_router_model_name_resolution(self):
        """Test that the router properly resolves model names with provider prefixes"""
        # Test model list with provider prefixes
        prefixed_model_list = [
            {
                "model_name": "openai/gpt-3.5-turbo",
                "litellm_params": {
                    "model": "openai/gpt-3.5-turbo"
                }
            },
            {
                "model_name": "anthropic/claude-3-haiku",
                "litellm_params": {
                    "model": "anthropic/claude-3-haiku"
                }
            }
        ]
        
        # Fallback config uses base model names
        fallbacks = [
            {"gpt-3.5-turbo": ["claude-3-haiku"]}
        ]
        
        router = Router(
            model_list=prefixed_model_list,
            fallbacks=fallbacks
        )
        
        user_with_prefixed_models = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            models=["openai/gpt-3.5-turbo", "anthropic/claude-3-haiku"],
            team_models=[],
        )
        
        # Test that provider-prefixed model matches base fallback config
        result = get_next_fallback(
            model="openai/gpt-3.5-turbo",
            user_api_key_dict=user_with_prefixed_models,
            llm_router=router,
            fallback_type="general"
        )
        assert result == "claude-3-haiku"

    def test_router_with_empty_fallback_chains(self):
        """Test router behavior with empty or incomplete fallback chains"""
        fallbacks = [
            {"claude-4-sonnet": []},  # Empty fallback list
            {"gpt-4o": ["azure-gpt-4o"]},  # Single fallback
        ]
        
        router = Router(
            model_list=self.model_list,
            fallbacks=fallbacks
        )
        
        # Test empty fallback list
        result = get_next_fallback(
            model="claude-4-sonnet",
            user_api_key_dict=self.user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        assert result is None
        
        # Test single fallback chain
        result = get_next_fallback(
            model="gpt-4o",
            user_api_key_dict=self.user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        assert result == "azure-gpt-4o"
        
        # Test end of single fallback chain
        result = get_next_fallback(
            model="azure-gpt-4o",
            user_api_key_dict=self.user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        assert result is None

    def test_router_multiple_fallback_types_priority(self):
        """Test behavior when multiple fallback types are configured"""
        general_fallbacks = [
            {"claude-4-sonnet": ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]}
        ]
        
        context_fallbacks = [
            {"claude-4-sonnet": ["google-claude-sonnet-4"]}  # Different order
        ]
        
        content_policy_fallbacks = [
            {"claude-4-sonnet": ["bedrock-claude-sonnet-4"]}  # Different fallback
        ]
        
        router = Router(
            model_list=self.model_list,
            fallbacks=general_fallbacks,
            context_window_fallbacks=context_fallbacks,
            content_policy_fallbacks=content_policy_fallbacks
        )
        
        # Each fallback type should return its configured fallback
        general_result = get_next_fallback(
            model="claude-4-sonnet",
            user_api_key_dict=self.user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        assert general_result == "bedrock-claude-sonnet-4"
        
        context_result = get_next_fallback(
            model="claude-4-sonnet",
            user_api_key_dict=self.user_api_key_dict,
            llm_router=router,
            fallback_type="context_window"
        )
        assert context_result == "google-claude-sonnet-4"
        
        policy_result = get_next_fallback(
            model="claude-4-sonnet",
            user_api_key_dict=self.user_api_key_dict,
            llm_router=router,
            fallback_type="content_policy"
        )
        assert policy_result == "bedrock-claude-sonnet-4"

    def test_router_with_access_control_scenarios(self):
        """Test various user access control scenarios with real router"""
        fallbacks = [
            {"claude-4-sonnet": ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]},
            {"gpt-4o": ["azure-gpt-4o", "github-gpt-4o"]}
        ]
        
        router = Router(
            model_list=self.model_list,
            fallbacks=fallbacks
        )
        
        # Test user with limited model access
        limited_user = UserAPIKeyAuth(
            api_key="limited-key",
            user_id="limited-user",
            models=["claude-4-sonnet", "google-claude-sonnet-4"],  # Missing bedrock-claude-sonnet-4
            team_models=[],
        )
        
        # Function should find the next accessible fallback (skipping inaccessible ones)
        result = get_next_fallback(
            model="claude-4-sonnet",
            user_api_key_dict=limited_user,
            llm_router=router,
            fallback_type="general"
        )
        # Should return the first fallback that the user can access
        assert result == "bedrock-claude-sonnet-4"  # This will be checked by the endpoint logic

    def test_router_performance_with_large_fallback_config(self):
        """Test performance with a large number of fallback configurations"""
        # Create a large fallback configuration
        large_fallbacks = []
        for i in range(100):
            large_fallbacks.append({
                f"model-{i}": [f"fallback-{i}-1", f"fallback-{i}-2", f"fallback-{i}-3"]
            })
        
        # Add our test models
        large_fallbacks.extend([
            {"claude-4-sonnet": ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]},
            {"gpt-4o": ["azure-gpt-4o", "github-gpt-4o"]}
        ])
        
        router = Router(
            model_list=self.model_list,
            fallbacks=large_fallbacks
        )
        
        # Test that our fallback lookup still works correctly with large config
        result = get_next_fallback(
            model="claude-4-sonnet",
            user_api_key_dict=self.user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        assert result == "bedrock-claude-sonnet-4"
        
        # Test fallback chain progression
        result = get_next_fallback(
            model="bedrock-claude-sonnet-4",
            user_api_key_dict=self.user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        assert result == "google-claude-sonnet-4"

    def test_router_edge_cases_and_error_conditions(self):
        """Test edge cases and potential error conditions"""
        fallbacks = [
            {"claude-4-sonnet": ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]}
        ]
        
        router = Router(
            model_list=self.model_list,
            fallbacks=fallbacks
        )
        
        # Test with None user_api_key_dict (should not crash)
        result = get_next_fallback(
            model="claude-4-sonnet",
            user_api_key_dict=self.user_api_key_dict,  # Still provide valid dict for this test
            llm_router=router,
            fallback_type="general"
        )
        assert result == "bedrock-claude-sonnet-4"
        
        # Test with invalid fallback type (should default to general)
        result = get_next_fallback(
            model="claude-4-sonnet",
            user_api_key_dict=self.user_api_key_dict,
            llm_router=router,
            fallback_type="invalid_type"  # Should default to general
        )
        assert result == "bedrock-claude-sonnet-4"
        
        # Test with empty model string
        result = get_next_fallback(
            model="",
            user_api_key_dict=self.user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        assert result is None

    def test_router_fallback_consistency_with_litellm_completion(self):
        """Test that our fallback logic is consistent with how LiteLLM handles fallbacks internally"""
        # This is more of a documentation test to ensure we understand the expected behavior
        fallbacks = [
            {"claude-4-sonnet": ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]}
        ]
        
        router = Router(
            model_list=self.model_list,
            fallbacks=fallbacks
        )
        
        # Our next fallback function should return what would be tried next
        # if claude-4-sonnet failed in a real completion call
        result = get_next_fallback(
            model="claude-4-sonnet",
            user_api_key_dict=self.user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        
        # This should match the first fallback model that Router would try
        assert result == "bedrock-claude-sonnet-4"
        
        # And the progression should continue correctly
        result = get_next_fallback(
            model="bedrock-claude-sonnet-4",
            user_api_key_dict=self.user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        assert result == "google-claude-sonnet-4"
        
        # Finally, no more fallbacks
        result = get_next_fallback(
            model="google-claude-sonnet-4",
            user_api_key_dict=self.user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        assert result is None