"""
Unit tests for the get_next_fallback() function in model_checks.py

Tests the core logic for finding the next fallback model in a fallback chain.
"""
from unittest.mock import MagicMock, Mock
from typing import List, Dict, Any, Optional

import pytest

from litellm.proxy._types import UserAPIKeyAuth  
from litellm.proxy.auth.model_checks import get_next_fallback
from litellm.router import Router


class TestGetNextFallback:
    """Test cases for get_next_fallback function"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        # Create a mock user API key dict
        self.mock_user_api_key_dict = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            models=["claude-4-sonnet", "bedrock-claude-sonnet-4", "google-claude-sonnet-4"],
            team_models=[],
        )
        
        # Sample fallback configuration matching the user's requirements
        self.sample_fallbacks = [
            {"claude-4-sonnet": ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]},
            {"claude-sonnet-4": ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]},
            {"claude-sonnet-4-failing": ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]},
            {"claude-3-7-sonnet": ["bedrock-claude-3-7-sonnet", "google-claude-3-7-sonnet"]},
            {"gpt-4o": ["azure-gpt-4o", "github-gpt-4o"]},
            {"gpt-4.1": ["github-gpt-4.1"]},
        ]

    def create_mock_router(self, fallbacks: List[Dict[str, List[str]]] = None) -> Mock:
        """Create a mock router with fallback configuration"""
        mock_router = Mock(spec=Router)
        mock_router.fallbacks = fallbacks if fallbacks is not None else self.sample_fallbacks
        mock_router.context_window_fallbacks = []
        mock_router.content_policy_fallbacks = []
        return mock_router

    def test_primary_model_returns_first_fallback(self):
        """Test that a primary model returns its first fallback"""
        router = self.create_mock_router()
        
        result = get_next_fallback(
            model="claude-4-sonnet",
            user_api_key_dict=self.mock_user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        
        assert result == "bedrock-claude-sonnet-4"

    def test_fallback_model_returns_next_in_chain(self):
        """Test that a fallback model returns the next model in the chain"""
        router = self.create_mock_router()
        
        result = get_next_fallback(
            model="bedrock-claude-sonnet-4",
            user_api_key_dict=self.mock_user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        
        assert result == "google-claude-sonnet-4"

    def test_last_fallback_returns_none(self):
        """Test that the last model in a fallback chain returns None"""
        router = self.create_mock_router()
        
        result = get_next_fallback(
            model="google-claude-sonnet-4",
            user_api_key_dict=self.mock_user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        
        assert result is None

    def test_nonexistent_model_returns_none(self):
        """Test that a model not in any fallback configuration returns None"""
        router = self.create_mock_router()
        
        result = get_next_fallback(
            model="non-existent-model",
            user_api_key_dict=self.mock_user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        
        assert result is None

    def test_single_fallback_chain(self):
        """Test a model with only one fallback"""
        router = self.create_mock_router()
        
        # Test primary model with single fallback
        result = get_next_fallback(
            model="gpt-4.1",
            user_api_key_dict=self.mock_user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        assert result == "github-gpt-4.1"
        
        # Test the single fallback returns None
        result = get_next_fallback(
            model="github-gpt-4.1",
            user_api_key_dict=self.mock_user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        assert result is None

    def test_context_window_fallbacks(self):
        """Test context window fallback type"""
        router = self.create_mock_router()
        router.context_window_fallbacks = [
            {"claude-4-sonnet": ["bedrock-claude-sonnet-4"]}
        ]
        
        result = get_next_fallback(
            model="claude-4-sonnet",
            user_api_key_dict=self.mock_user_api_key_dict,
            llm_router=router,
            fallback_type="context_window"
        )
        
        assert result == "bedrock-claude-sonnet-4"

    def test_content_policy_fallbacks(self):
        """Test content policy fallback type"""
        router = self.create_mock_router()
        router.content_policy_fallbacks = [
            {"claude-4-sonnet": ["google-claude-sonnet-4"]}
        ]
        
        result = get_next_fallback(
            model="claude-4-sonnet",
            user_api_key_dict=self.mock_user_api_key_dict,
            llm_router=router,
            fallback_type="content_policy"
        )
        
        assert result == "google-claude-sonnet-4"

    def test_empty_fallbacks(self):
        """Test behavior with empty fallback configurations"""
        router = self.create_mock_router(fallbacks=[])
        
        result = get_next_fallback(
            model="claude-4-sonnet",
            user_api_key_dict=self.mock_user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        
        assert result is None

    def test_no_router(self):
        """Test behavior when no router is provided"""
        result = get_next_fallback(
            model="claude-4-sonnet",
            user_api_key_dict=self.mock_user_api_key_dict,
            llm_router=None,
            fallback_type="general"
        )
        
        assert result is None

    def test_stripped_model_names(self):
        """Test fallback resolution with provider-prefixed model names"""
        # Test with fallback config that uses base model names
        fallbacks = [
            {"gpt-3.5-turbo": ["claude-3-haiku"]}
        ]
        router = self.create_mock_router(fallbacks=fallbacks)
        
        # Test that provider-prefixed model matches base name in config
        result = get_next_fallback(
            model="openai/gpt-3.5-turbo",
            user_api_key_dict=self.mock_user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        
        assert result == "claude-3-haiku"

    def test_multiple_fallback_chains(self):
        """Test with multiple independent fallback chains"""
        router = self.create_mock_router()
        
        # Test gpt-4o chain
        result = get_next_fallback(
            model="gpt-4o",
            user_api_key_dict=self.mock_user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        assert result == "azure-gpt-4o"
        
        result = get_next_fallback(
            model="azure-gpt-4o",
            user_api_key_dict=self.mock_user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        assert result == "github-gpt-4o"
        
        result = get_next_fallback(
            model="github-gpt-4o",
            user_api_key_dict=self.mock_user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        assert result is None

    def test_generic_wildcard_fallbacks(self):
        """Test generic fallbacks using wildcard (*) key"""
        fallbacks = [
            {"claude-4-sonnet": ["bedrock-claude-sonnet-4"]},
            {"*": ["generic-fallback-model"]}  # Generic fallback
        ]
        router = self.create_mock_router(fallbacks=fallbacks)
        
        # Test that model not in specific fallbacks uses generic fallback
        result = get_next_fallback(
            model="unknown-model",
            user_api_key_dict=self.mock_user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        
        assert result == "generic-fallback-model"

    @pytest.mark.parametrize("fallback_type,expected_attr", [
        ("general", "fallbacks"),
        ("context_window", "context_window_fallbacks"),
        ("content_policy", "content_policy_fallbacks"),
    ])
    def test_fallback_type_attribute_mapping(self, fallback_type, expected_attr):
        """Test that different fallback types access the correct router attribute"""
        router = self.create_mock_router()
        
        # Clear all fallback types
        router.fallbacks = []
        router.context_window_fallbacks = []
        router.content_policy_fallbacks = []
        
        # Set only the specific fallback type being tested
        test_fallbacks = [{"test-model": ["test-fallback"]}]
        setattr(router, expected_attr, test_fallbacks)
        
        result = get_next_fallback(
            model="test-model",
            user_api_key_dict=self.mock_user_api_key_dict,
            llm_router=router,
            fallback_type=fallback_type
        )
        
        assert result == "test-fallback"

    def test_edge_case_empty_fallback_list(self):
        """Test edge case where fallback list is empty"""
        fallbacks = [
            {"claude-4-sonnet": []},  # Empty fallback list
        ]
        router = self.create_mock_router(fallbacks=fallbacks)
        
        result = get_next_fallback(
            model="claude-4-sonnet",
            user_api_key_dict=self.mock_user_api_key_dict,
            llm_router=router,
            fallback_type="general"
        )
        
        assert result is None

    def test_complex_fallback_scenario(self):
        """Test complex scenario with user's exact configuration"""
        complex_fallbacks = [
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
        
        router = self.create_mock_router(fallbacks=complex_fallbacks)
        
        # Test multiple chains from user's config
        test_cases = [
            ("claude-4-sonnet", "bedrock-claude-sonnet-4"),
            ("bedrock-claude-sonnet-4", "google-claude-sonnet-4"),
            ("google-claude-sonnet-4", None),
            ("gpt-4.5-preview", "azure-gpt-4.5-preview"),
            ("azure-gpt-4.5-preview", None),
            ("non-existent-model", None),
        ]
        
        for model, expected in test_cases:
            result = get_next_fallback(
                model=model,
                user_api_key_dict=self.mock_user_api_key_dict,
                llm_router=router,
                fallback_type="general"
            )
            assert result == expected, f"Failed for model {model}: expected {expected}, got {result}"