"""
Unit tests for Apple Foundation Models common utilities (v0.2.0+ SDK).
"""

import sys
from unittest.mock import Mock, patch

import pytest

from litellm.llms.apple_foundation_models.common_utils import (
    get_apple_async_session_class,
    get_apple_session_class,
)


@pytest.fixture(autouse=True)
def cleanup_apple_module():
    """Clean up applefoundationmodels from sys.modules after each test."""
    yield
    sys.modules.pop("applefoundationmodels", None)


class TestAppleFoundationModelsCommonUtils:
    """Test suite for Apple Foundation Models common utilities (v0.2.0+ SDK)."""

    def test_get_apple_session_class_import_error(self):
        """Test that ImportError is raised when package not installed."""
        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'applefoundationmodels'"),
        ):
            with pytest.raises(ImportError) as exc_info:
                get_apple_session_class()

            assert "apple-foundation-models" in str(exc_info.value)
            assert "pip install" in str(exc_info.value)

    def test_get_apple_session_class_availability_failure(self):
        """Test that RuntimeError is raised when Apple Intelligence not available."""
        mock_session_class = Mock()
        mock_module = Mock(
            Session=mock_session_class,
            AsyncSession=Mock(),
            apple_intelligence_available=Mock(return_value=False),
        )

        with patch.dict("sys.modules", {"applefoundationmodels": mock_module}):
            with pytest.raises(RuntimeError) as exc_info:
                get_apple_session_class()

            assert "Apple Intelligence" in str(exc_info.value)
            mock_module.apple_intelligence_available.assert_called_once()

    def test_get_apple_session_class_success(self):
        """Test successful Session class retrieval."""
        mock_session_class = Mock()
        mock_module = Mock(
            Session=mock_session_class,
            AsyncSession=Mock(),
            apple_intelligence_available=Mock(return_value=True),
        )

        with patch.dict("sys.modules", {"applefoundationmodels": mock_module}):
            result = get_apple_session_class()

            assert result == mock_session_class
            mock_module.apple_intelligence_available.assert_called_once()

    def test_get_apple_async_session_class_import_error(self):
        """Test that ImportError is raised when package not installed (async)."""
        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'applefoundationmodels'"),
        ):
            with pytest.raises(ImportError) as exc_info:
                get_apple_async_session_class()

            assert "apple-foundation-models" in str(exc_info.value)
            assert "pip install" in str(exc_info.value)

    def test_get_apple_async_session_class_availability_failure(self):
        """Test that RuntimeError is raised when Apple Intelligence not available (async)."""
        mock_async_session_class = Mock()
        mock_module = Mock(
            Session=Mock(),
            AsyncSession=mock_async_session_class,
            apple_intelligence_available=Mock(return_value=False),
        )

        with patch.dict("sys.modules", {"applefoundationmodels": mock_module}):
            with pytest.raises(RuntimeError) as exc_info:
                get_apple_async_session_class()

            assert "Apple Intelligence" in str(exc_info.value)
            mock_module.apple_intelligence_available.assert_called_once()

    def test_get_apple_async_session_class_success(self):
        """Test successful AsyncSession class retrieval."""
        mock_async_session_class = Mock()
        mock_module = Mock(
            Session=Mock(),
            AsyncSession=mock_async_session_class,
            apple_intelligence_available=Mock(return_value=True),
        )

        with patch.dict("sys.modules", {"applefoundationmodels": mock_module}):
            result = get_apple_async_session_class()

            assert result == mock_async_session_class
            mock_module.apple_intelligence_available.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
