"""
Test case for the Perplexity response_format fix.
This test ensures that response_format with type "text" is filtered out for Perplexity models.
"""
import pytest

from litellm.llms.perplexity.chat.transformation import PerplexityChatConfig


class TestPerplexityResponseFormatFiltering:
    """Test suite for Perplexity response_format filtering functionality."""

    def test_response_format_text_filtering(self):
        """
        Test that response_format with type "text" is filtered out for Perplexity models.
        
        This addresses GitHub issue #18694 where Perplexity API returns a 
        "Structured output schema error" when response_format: {"type": "text"} is sent.
        """
        config = PerplexityChatConfig()
        
        # Test with response_format type "text" - should be filtered out
        optional_params = {
            "response_format": {"type": "text"},
            "stream": True,
            "temperature": 0.7,
        }
        
        result = config.transform_request(
            model="perplexity/sonar-deep-research",
            messages=[{"role": "user", "content": "Test message"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        
        # Verify that response_format is removed
        assert "response_format" not in result, \
            "response_format with type 'text' should be filtered out for Perplexity"
        
        # Verify other parameters remain intact
        assert result["stream"] is True, "Other parameters should remain unchanged"
        assert result["temperature"] == 0.7, "Other parameters should remain unchanged"

    def test_valid_response_format_preservation(self):
        """Test that valid response_format parameters are preserved."""
        config = PerplexityChatConfig()
        
        # Test with valid json_schema response_format - should be preserved
        optional_params = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": {"type": "object", "properties": {"name": {"type": "string"}}}
                }
            },
            "stream": True,
        }
        
        result = config.transform_request(
            model="perplexity/sonar-deep-research",
            messages=[{"role": "user", "content": "Test message"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        
        # Verify that valid response_format is preserved
        assert "response_format" in result, "Valid response_format should be preserved"
        assert result["response_format"]["type"] == "json_schema", \
            "Valid json_schema response_format should remain unchanged"

    def test_regex_response_format_preservation(self):
        """Test that regex response_format is preserved for sonar models."""
        config = PerplexityChatConfig()
        
        # Test with valid regex response_format - should be preserved
        optional_params = {
            "response_format": {
                "type": "regex",
                "regex": {"regex": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"}
            },
            "stream": True,
        }
        
        result = config.transform_request(
            model="perplexity/sonar",  # regex is supported on sonar model
            messages=[{"role": "user", "content": "Find email address"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        
        # Verify that valid regex response_format is preserved
        assert "response_format" in result, "Valid regex response_format should be preserved"
        assert result["response_format"]["type"] == "regex", \
            "Valid regex response_format should remain unchanged"

    def test_no_response_format_parameter(self):
        """Test that requests without response_format work correctly."""
        config = PerplexityChatConfig()
        
        optional_params = {
            "stream": True,
            "temperature": 0.5,
        }
        
        result = config.transform_request(
            model="perplexity/sonar-deep-research",
            messages=[{"role": "user", "content": "Test message"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        
        # Verify no response_format is added when not provided
        assert "response_format" not in result, \
            "No response_format should be added when not provided"
        assert result["stream"] is True, "Other parameters should remain"

    def test_github_issue_18694_scenario(self):
        """
        Test the exact scenario reported in GitHub issue #18694.
        
        This reproduces the original problem where sonar-deep-research with
        response_format: {"type": "text"} caused an API error.
        """
        config = PerplexityChatConfig()
        
        # Reproduce the exact parameters from the GitHub issue
        optional_params = {
            "cache": False,
            "timeout": None,
            "response_format": {"type": "text"},  # This was causing the error
            "stream": True,
        }
        
        result = config.transform_request(
            model="perplexity/sonar-deep-research",
            messages=[
                {"role": "system", "content": "YOUR PERSONAL GOAL: Do research"},
                {"role": "user", "content": "Ruben Amorin"},
            ],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        
        # Verify the problematic parameter is filtered out
        assert "response_format" not in result, \
            "response_format should be filtered out to prevent API error"
        
        # Verify other parameters are preserved
        assert result["stream"] is True, "stream parameter should be preserved"
        assert result["cache"] is False, "cache parameter should be preserved"
        assert result["timeout"] is None, "timeout parameter should be preserved"
