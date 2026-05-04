"""Tests for base_llm/fetch/transformation.py.

Covers BaseFetchConfig validation and WebFetchResponse model.
"""

import pytest
from pydantic import ValidationError

from litellm.llms.base_llm.fetch.transformation import (
    BaseFetchConfig,
    WebFetchResponse,
)


class TestWebFetchResponse:
    """Test WebFetchResponse Pydantic model."""

    def test_basic_creation(self):
        """Create with required fields."""
        response = WebFetchResponse(
            url="https://example.com",
            content="Hello World",
        )
        assert response.url == "https://example.com"
        assert response.content == "Hello World"
        assert response.title is None
        assert response.metadata is None

    def test_with_optional_fields(self):
        """Create with all fields."""
        response = WebFetchResponse(
            url="https://example.com",
            title="Test Page",
            content="Hello",
            metadata={"author": "test"},
        )
        assert response.title == "Test Page"
        assert response.metadata["author"] == "test"

    def test_extra_fields_allowed(self):
        """Extra fields are allowed via ConfigDict."""
        response = WebFetchResponse(
            url="https://example.com",
            content="Hello",
            custom_field="value",
        )
        assert response.custom_field == "value"

    def test_missing_url_raises(self):
        """Missing required URL raises ValidationError."""
        with pytest.raises(ValidationError):
            WebFetchResponse(content="Hello")

    def test_missing_content_raises(self):
        """Missing required content raises ValidationError."""
        with pytest.raises(ValidationError):
            WebFetchResponse(url="https://example.com")


class TestBaseFetchConfig:
    """Test BaseFetchConfig ABC."""

    def test_cannot_instantiate(self):
        """BaseFetchConfig is abstract."""
        with pytest.raises(TypeError):
            BaseFetchConfig()

    def test_required_methods(self):
        """Subclasses must implement abstract methods."""
        # Verify the abstract methods exist
        assert hasattr(BaseFetchConfig, "validate_environment")
        assert hasattr(BaseFetchConfig, "ui_friendly_name")
        assert hasattr(BaseFetchConfig, "scrape_url")
        assert hasattr(BaseFetchConfig, "afetch_url")
