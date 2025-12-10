"""
Tests for Langfuse masking function support.
See: https://langfuse.com/docs/observability/features/masking
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.langfuse.langfuse_handler import LangFuseHandler
from litellm.integrations.langfuse.langfuse import LangFuseLogger
from litellm.litellm_core_utils.litellm_logging import DynamicLoggingCache
from litellm.types.utils import StandardCallbackDynamicParams


def sample_masking_function(data):
    """Sample masking function for testing"""
    if "input" in data:
        data["input"] = "[REDACTED]"
    if "output" in data:
        data["output"] = "[REDACTED]"
    return data


def another_masking_function(data):
    """Another masking function for testing caching behavior"""
    return data


@pytest.fixture
def dynamic_logging_cache():
    return DynamicLoggingCache()


class TestLangfuseMaskingFunctionDetection:
    """Test that masking function is properly detected as a dynamic parameter"""

    def test_masking_function_only_is_detected(self):
        """
        Test that _dynamic_langfuse_credentials_are_passed returns True when only masking function is passed
        """
        params_with_masking = StandardCallbackDynamicParams(
            langfuse_masking_function=sample_masking_function
        )
        assert (
            LangFuseHandler._dynamic_langfuse_credentials_are_passed(params_with_masking)
            is True
        )

    def test_masking_function_with_credentials_is_detected(self):
        """
        Test that masking function with other credentials is detected
        """
        params_full = StandardCallbackDynamicParams(
            langfuse_public_key="test_key",
            langfuse_secret="test_secret",
            langfuse_host="https://test.langfuse.com",
            langfuse_masking_function=sample_masking_function,
        )
        assert (
            LangFuseHandler._dynamic_langfuse_credentials_are_passed(params_full) is True
        )


class TestLangfuseLoggingConfig:
    """Test that logging config properly includes masking function"""

    def test_config_includes_masking_function(self):
        """
        Test that get_dynamic_langfuse_logging_config includes the masking function
        """
        dynamic_params = StandardCallbackDynamicParams(
            langfuse_public_key="dynamic_key",
            langfuse_secret="dynamic_secret",
            langfuse_host="https://dynamic.langfuse.com",
            langfuse_masking_function=sample_masking_function,
        )
        config = LangFuseHandler.get_dynamic_langfuse_logging_config(dynamic_params)
        assert config["langfuse_public_key"] == "dynamic_key"
        assert config["langfuse_secret"] == "dynamic_secret"
        assert config["langfuse_host"] == "https://dynamic.langfuse.com"
        assert config["langfuse_masking_function"] == sample_masking_function

    def test_config_without_masking_function(self):
        """
        Test config without masking function has None for the field
        """
        dynamic_params = StandardCallbackDynamicParams(
            langfuse_public_key="dynamic_key",
            langfuse_secret="dynamic_secret",
        )
        config = LangFuseHandler.get_dynamic_langfuse_logging_config(dynamic_params)
        assert config["langfuse_masking_function"] is None


class TestCacheFriendlyCredentials:
    """Test that cache-friendly credentials properly handle masking function"""

    def test_masking_function_converted_to_id(self):
        """
        Test that _get_cache_friendly_credentials converts masking function to id
        """
        credentials = {
            "langfuse_public_key": "test_key",
            "langfuse_secret": "test_secret",
            "langfuse_host": "https://test.langfuse.com",
            "langfuse_masking_function": sample_masking_function,
        }
        cache_credentials = LangFuseHandler._get_cache_friendly_credentials(credentials)

        # Masking function should be replaced with its id
        assert "langfuse_masking_function" not in cache_credentials
        assert "langfuse_masking_function_id" in cache_credentials
        assert cache_credentials["langfuse_masking_function_id"] == id(
            sample_masking_function
        )

        # Other credentials should remain unchanged
        assert cache_credentials["langfuse_public_key"] == "test_key"
        assert cache_credentials["langfuse_secret"] == "test_secret"
        assert cache_credentials["langfuse_host"] == "https://test.langfuse.com"

    def test_without_masking_function(self):
        """
        Test cache credentials without masking function
        """
        credentials_no_mask = {
            "langfuse_public_key": "test_key",
            "langfuse_secret": "test_secret",
        }
        cache_credentials_no_mask = LangFuseHandler._get_cache_friendly_credentials(
            credentials_no_mask
        )

        assert "langfuse_masking_function" not in cache_credentials_no_mask
        assert "langfuse_masking_function_id" not in cache_credentials_no_mask


class TestLangfuseLoggerInitialization:
    """Test that LangFuseLogger properly passes masking function to Langfuse client"""

    def test_init_with_masking_function(self):
        """
        Test that LangFuseLogger passes the masking function to the Langfuse client
        """
        with patch("langfuse.Langfuse") as mock_langfuse_class:
            mock_client = Mock()
            mock_client.client = Mock()
            mock_langfuse_class.return_value = mock_client

            # Create logger with masking function
            LangFuseLogger(
                langfuse_public_key="test_key",
                langfuse_secret="test_secret",
                langfuse_host="https://test.langfuse.com",
                langfuse_masking_function=sample_masking_function,
            )

            # Verify Langfuse was called with the mask parameter
            call_kwargs = mock_langfuse_class.call_args[1]
            assert "mask" in call_kwargs
            assert call_kwargs["mask"] == sample_masking_function

    def test_init_without_masking_function(self):
        """
        Test that LangFuseLogger does not pass mask parameter when no masking function is provided
        """
        with patch("langfuse.Langfuse") as mock_langfuse_class:
            mock_client = Mock()
            mock_client.client = Mock()
            mock_langfuse_class.return_value = mock_client

            # Create logger without masking function
            LangFuseLogger(
                langfuse_public_key="test_key",
                langfuse_secret="test_secret",
                langfuse_host="https://test.langfuse.com",
            )

            # Verify Langfuse was called without the mask parameter
            call_kwargs = mock_langfuse_class.call_args[1]
            assert "mask" not in call_kwargs or call_kwargs.get("mask") is None


class TestLangfuseLoggerCaching:
    """Test that loggers are properly cached based on masking function"""

    def test_logger_created_with_masking_function(self, dynamic_logging_cache):
        """
        Test that get_langfuse_logger_for_request correctly creates a logger with masking function
        """
        with patch("langfuse.Langfuse") as mock_langfuse_class:
            mock_client = Mock()
            mock_client.client = Mock()
            mock_langfuse_class.return_value = mock_client

            dynamic_params = StandardCallbackDynamicParams(
                langfuse_public_key="test_key",
                langfuse_secret="test_secret",
                langfuse_host="https://test.langfuse.com",
                langfuse_masking_function=sample_masking_function,
            )

            result = LangFuseHandler.get_langfuse_logger_for_request(
                standard_callback_dynamic_params=dynamic_params,
                in_memory_dynamic_logger_cache=dynamic_logging_cache,
                globalLangfuseLogger=None,
            )

            assert isinstance(result, LangFuseLogger)
            assert result.public_key == "test_key"
            assert result.secret_key == "test_secret"

    def test_different_masking_functions_create_different_loggers(
        self, dynamic_logging_cache
    ):
        """
        Test that different masking functions result in different cached loggers
        """
        with patch("langfuse.Langfuse") as mock_langfuse_class:
            mock_client = Mock()
            mock_client.client = Mock()
            mock_langfuse_class.return_value = mock_client

            # First logger with sample_masking_function
            params1 = StandardCallbackDynamicParams(
                langfuse_public_key="test_key",
                langfuse_secret="test_secret",
                langfuse_host="https://test.langfuse.com",
                langfuse_masking_function=sample_masking_function,
            )

            logger1 = LangFuseHandler.get_langfuse_logger_for_request(
                standard_callback_dynamic_params=params1,
                in_memory_dynamic_logger_cache=dynamic_logging_cache,
                globalLangfuseLogger=None,
            )

            # Second logger with another_masking_function
            params2 = StandardCallbackDynamicParams(
                langfuse_public_key="test_key",
                langfuse_secret="test_secret",
                langfuse_host="https://test.langfuse.com",
                langfuse_masking_function=another_masking_function,
            )

            logger2 = LangFuseHandler.get_langfuse_logger_for_request(
                standard_callback_dynamic_params=params2,
                in_memory_dynamic_logger_cache=dynamic_logging_cache,
                globalLangfuseLogger=None,
            )

            # The loggers should be different instances because they have different masking functions
            assert logger1 is not logger2

    def test_same_masking_function_returns_cached_logger(self, dynamic_logging_cache):
        """
        Test that the same masking function returns the same cached logger
        """
        with patch("langfuse.Langfuse") as mock_langfuse_class:
            mock_client = Mock()
            mock_client.client = Mock()
            mock_langfuse_class.return_value = mock_client

            params = StandardCallbackDynamicParams(
                langfuse_public_key="test_key",
                langfuse_secret="test_secret",
                langfuse_host="https://test.langfuse.com",
                langfuse_masking_function=sample_masking_function,
            )

            logger1 = LangFuseHandler.get_langfuse_logger_for_request(
                standard_callback_dynamic_params=params,
                in_memory_dynamic_logger_cache=dynamic_logging_cache,
                globalLangfuseLogger=None,
            )

            logger2 = LangFuseHandler.get_langfuse_logger_for_request(
                standard_callback_dynamic_params=params,
                in_memory_dynamic_logger_cache=dynamic_logging_cache,
                globalLangfuseLogger=None,
            )

            # The loggers should be the same instance because they have the same masking function
            assert logger1 is logger2
