"""
Test file for Perplexity chat transformation functionality.

Tests the response transformation to extract citation tokens and search queries
from Perplexity API responses.
"""

import os
import sys
from unittest.mock import Mock

import pytest

# Add the project root to Python path
sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm import ModelResponse
from litellm.llms.perplexity.chat.transformation import PerplexityChatConfig
from litellm.types.utils import Usage


class TestPerplexityChatTransformation:
    """Test suite for Perplexity chat transformation functionality."""

    def test_enhance_usage_with_citation_tokens(self):
        """Test extraction of citation tokens from API response."""
        config = PerplexityChatConfig()
        
        # Create a ModelResponse with basic usage
        model_response = ModelResponse()
        model_response.usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        # Mock raw response with citations
        raw_response_dict = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            },
            "citations": [
                "This is a citation with some text content",
                "Another citation with more text here",
                "Third citation with additional information"
            ]
        }
        
        # Enhance the usage with Perplexity fields
        config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)
        
        # Check that citation tokens were added
        assert hasattr(model_response.usage, "citation_tokens")
        citation_tokens = getattr(model_response.usage, "citation_tokens")
        
        # Should have extracted citation tokens (estimated based on character count)
        assert citation_tokens > 0
        assert isinstance(citation_tokens, int)

    def test_enhance_usage_with_search_queries_from_usage(self):
        """Test extraction of search queries from usage field in API response."""
        config = PerplexityChatConfig()
        
        # Create a ModelResponse with basic usage
        model_response = ModelResponse()
        model_response.usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        # Mock raw response with search queries in usage
        raw_response_dict = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "num_search_queries": 3
            }
        }
        
        # Enhance the usage with Perplexity fields
        config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)
        
        # Check that search queries were added to prompt_tokens_details
        assert hasattr(model_response.usage, "prompt_tokens_details")
        assert model_response.usage.prompt_tokens_details is not None
        assert hasattr(model_response.usage.prompt_tokens_details, "web_search_requests")
        
        web_search_requests = model_response.usage.prompt_tokens_details.web_search_requests
        assert web_search_requests == 3

    def test_enhance_usage_with_search_queries_from_root(self):
        """Test extraction of search queries from root level in API response."""
        config = PerplexityChatConfig()
        
        # Create a ModelResponse with basic usage
        model_response = ModelResponse()
        model_response.usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        # Mock raw response with search queries at root level
        raw_response_dict = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            },
            "num_search_queries": 2
        }
        
        # Enhance the usage with Perplexity fields
        config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)
        
        # Check that search queries were added to prompt_tokens_details
        assert hasattr(model_response.usage, "prompt_tokens_details")
        assert model_response.usage.prompt_tokens_details is not None
        assert hasattr(model_response.usage.prompt_tokens_details, "web_search_requests")
        
        web_search_requests = model_response.usage.prompt_tokens_details.web_search_requests
        assert web_search_requests == 2

    def test_enhance_usage_with_both_citations_and_search_queries(self):
        """Test extraction of both citation tokens and search queries."""
        config = PerplexityChatConfig()
        
        # Create a ModelResponse with basic usage
        model_response = ModelResponse()
        model_response.usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        # Mock raw response with both citations and search queries
        raw_response_dict = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "num_search_queries": 2
            },
            "citations": [
                "Citation one with some content",
                "Citation two with more information"
            ]
        }
        
        # Enhance the usage with Perplexity fields
        config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)
        
        # Check that both fields were added
        assert hasattr(model_response.usage, "citation_tokens")
        assert hasattr(model_response.usage, "prompt_tokens_details")
        assert model_response.usage.prompt_tokens_details is not None
        assert hasattr(model_response.usage.prompt_tokens_details, "web_search_requests")
        
        citation_tokens = getattr(model_response.usage, "citation_tokens")
        web_search_requests = model_response.usage.prompt_tokens_details.web_search_requests
        
        assert citation_tokens > 0
        assert web_search_requests == 2

    def test_enhance_usage_with_empty_citations(self):
        """Test handling of empty citations array."""
        config = PerplexityChatConfig()
        
        # Create a ModelResponse with basic usage
        model_response = ModelResponse()
        model_response.usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        # Mock raw response with empty citations
        raw_response_dict = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            },
            "citations": []
        }
        
        # Enhance the usage with Perplexity fields
        config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)
        
        # Should not set citation_tokens for empty citations
        citation_tokens = getattr(model_response.usage, "citation_tokens", 0)
        assert citation_tokens == 0

    def test_enhance_usage_with_missing_fields(self):
        """Test handling when both citations and search queries are missing."""
        config = PerplexityChatConfig()
        
        # Create a ModelResponse with basic usage
        model_response = ModelResponse()
        model_response.usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        # Mock raw response without citations or search queries
        raw_response_dict = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        
        # Should not raise an error
        config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)
        
        # Should not have added custom fields
        citation_tokens = getattr(model_response.usage, "citation_tokens", 0)
        assert citation_tokens == 0
        
        # prompt_tokens_details might be None or have web_search_requests as 0
        if hasattr(model_response.usage, "prompt_tokens_details") and model_response.usage.prompt_tokens_details:
            web_search_requests = getattr(model_response.usage.prompt_tokens_details, "web_search_requests", 0)
            assert web_search_requests == 0

    def test_citation_token_estimation(self):
        """Test that citation token estimation is reasonable."""
        config = PerplexityChatConfig()
        
        # Test cases with known character counts
        test_cases = [
            # (citation_text, expected_min_tokens, expected_max_tokens)
            ("Short", 1, 2),
            ("This is a longer citation with multiple words", 10, 15),
            ("A very long citation with many words and characters that should result in more tokens", 18, 25),
        ]
        
        for citation_text, min_tokens, max_tokens in test_cases:
            model_response = ModelResponse()
            model_response.usage = Usage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150
            )
            
            raw_response_dict = {
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                "citations": [citation_text]
            }
            
            config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)
            
            citation_tokens = getattr(model_response.usage, "citation_tokens")
            
            # Should be within reasonable range
            assert min_tokens <= citation_tokens <= max_tokens, f"Citation '{citation_text}' resulted in {citation_tokens} tokens, expected {min_tokens}-{max_tokens}"

    def test_multiple_citations_aggregation(self):
        """Test that multiple citations are aggregated correctly."""
        config = PerplexityChatConfig()
        
        model_response = ModelResponse()
        model_response.usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        raw_response_dict = {
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            "citations": [
                "First citation with some text",
                "Second citation with different content",
                "Third citation with more information"
            ]
        }
        
        config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)
        
        citation_tokens = getattr(model_response.usage, "citation_tokens")
        
        # Should have aggregated all citations
        total_chars = sum(len(citation) for citation in raw_response_dict["citations"])
        expected_tokens = total_chars // 4  # Our estimation logic
        
        assert citation_tokens == expected_tokens

    def test_search_queries_priority_usage_over_root(self):
        """Test that search queries from usage field take priority over root level."""
        config = PerplexityChatConfig()
        
        # Create a ModelResponse with basic usage
        model_response = ModelResponse()
        model_response.usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        # Mock raw response with search queries in both locations
        raw_response_dict = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "num_search_queries": 5  # This should take priority
            },
            "num_search_queries": 3  # This should be ignored
        }
        
        # Enhance the usage with Perplexity fields
        config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)
        
        # Check that usage field took priority
        assert hasattr(model_response.usage, "prompt_tokens_details")
        assert model_response.usage.prompt_tokens_details is not None
        web_search_requests = model_response.usage.prompt_tokens_details.web_search_requests
        
        assert web_search_requests == 5  # Should use the usage field value, not root

    def test_no_usage_object_handling(self):
        """Test handling when model_response has no usage object."""
        config = PerplexityChatConfig()
        
        # Create a ModelResponse without usage
        model_response = ModelResponse()
        
        # Mock raw response with Perplexity-specific fields
        raw_response_dict = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "num_search_queries": 2
            },
            "citations": ["Some citation"]
        }
        
        # Should not raise an error when usage is None
        config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)
        
        # Usage should be created with the Perplexity fields
        assert model_response.usage is not None
        assert hasattr(model_response.usage, "citation_tokens")
        assert hasattr(model_response.usage, "prompt_tokens_details")
        assert model_response.usage.prompt_tokens_details is not None
        assert hasattr(model_response.usage.prompt_tokens_details, "web_search_requests")
        
        citation_tokens = getattr(model_response.usage, "citation_tokens")
        web_search_requests = model_response.usage.prompt_tokens_details.web_search_requests
        
        assert citation_tokens > 0
        assert web_search_requests == 2

    @pytest.mark.parametrize("search_query_location", ["usage", "root"])
    def test_search_queries_extraction_locations(self, search_query_location):
        """Test search queries extraction from different response locations."""
        config = PerplexityChatConfig()
        
        # Create a ModelResponse with basic usage
        model_response = ModelResponse()
        model_response.usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        # Create response dict based on parameter
        if search_query_location == "usage":
            raw_response_dict = {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "num_search_queries": 4
                }
            }
        else:  # root
            raw_response_dict = {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150
                },
                "num_search_queries": 4
            }
        
        # Enhance the usage with Perplexity fields
        config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)
        
        # Should extract search queries from either location
        assert hasattr(model_response.usage, "prompt_tokens_details")
        assert model_response.usage.prompt_tokens_details is not None
        web_search_requests = model_response.usage.prompt_tokens_details.web_search_requests
        
        assert web_search_requests == 4 