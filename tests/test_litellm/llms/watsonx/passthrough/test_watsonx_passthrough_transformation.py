"""
Unit tests for WatsonxPassthroughConfig transformation.

Tests the Watsonx-specific passthrough configuration including URL construction,
streaming detection, and authentication handling.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.watsonx.passthrough.transformation import WatsonxPassthroughConfig


class TestWatsonxPassthroughConfig:
    """Tests for WatsonxPassthroughConfig class."""

    def test_is_streaming_request_true(self):
        """Test that streaming is detected when stream=True in request data."""
        config = WatsonxPassthroughConfig()
        request_data = {"stream": True, "input": "test"}

        result = config.is_streaming_request(
            endpoint="ml/v1/text/generation", request_data=request_data
        )

        assert result is True

    def test_is_streaming_request_false(self):
        """Test that streaming is not detected when stream=False in request data."""
        config = WatsonxPassthroughConfig()
        request_data = {"stream": False, "input": "test"}

        result = config.is_streaming_request(
            endpoint="ml/v1/text/generation", request_data=request_data
        )

        assert result is False

    def test_is_streaming_request_missing_stream_key(self):
        """Test that streaming defaults to False when stream key is missing."""
        config = WatsonxPassthroughConfig()
        request_data = {"input": "test"}

        result = config.is_streaming_request(
            endpoint="ml/v1/text/generation", request_data=request_data
        )

        assert result is False

    def test_get_complete_url_with_api_base(self):
        """Test URL construction with explicit api_base."""
        config = WatsonxPassthroughConfig()
        api_base = "https://us-south.ml.cloud.ibm.com"
        endpoint = "ml/v1/text/generation"
        request_query_params = {"version": "2024-03-19"}

        complete_url, base_target_url = config.get_complete_url(
            api_base=api_base,
            api_key=None,
            model="ibm/granite-13b-chat-v2",
            endpoint=endpoint,
            request_query_params=request_query_params,
            litellm_params={},
        )

        assert isinstance(complete_url, httpx.URL)
        assert str(complete_url).startswith(api_base)
        assert endpoint in str(complete_url)
        assert "version=2024-03-19" in str(complete_url)
        assert base_target_url == api_base

    @patch("litellm.llms.watsonx.common_utils.get_secret_str")
    def test_get_complete_url_with_env_api_base(self, mock_get_secret):
        """Test URL construction with api_base from environment."""
        config = WatsonxPassthroughConfig()
        env_api_base = "https://eu-de.ml.cloud.ibm.com"
        mock_get_secret.return_value = env_api_base

        endpoint = "ml/v1/text/tokenization"
        request_query_params = {"version": "2024-03-19"}

        complete_url, base_target_url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="ibm/granite-13b-chat-v2",
            endpoint=endpoint,
            request_query_params=request_query_params,
            litellm_params={},
        )

        assert isinstance(complete_url, httpx.URL)
        assert str(complete_url).startswith(env_api_base)
        assert endpoint in str(complete_url)
        assert base_target_url == env_api_base

    def test_get_complete_url_with_query_params(self):
        """Test that query parameters are correctly added to URL."""
        config = WatsonxPassthroughConfig()
        api_base = "https://us-south.ml.cloud.ibm.com"
        endpoint = "ml/v1/text/generation"
        request_query_params = {
            "version": "2024-03-19",
        }

        complete_url, _ = config.get_complete_url(
            api_base=api_base,
            api_key=None,
            model="ibm/granite-13b-chat-v2",
            endpoint=endpoint,
            request_query_params=request_query_params,
            litellm_params={},
        )

        url_str = str(complete_url)
        assert "version=2024-03-19" in url_str

    def test_get_complete_url_without_query_params(self):
        """Test URL construction without query parameters."""
        config = WatsonxPassthroughConfig()
        api_base = "https://us-south.ml.cloud.ibm.com"
        endpoint = "ml/v1/models"

        complete_url, base_target_url = config.get_complete_url(
            api_base=api_base,
            api_key=None,
            model="",
            endpoint=endpoint,
            request_query_params=None,
            litellm_params={},
        )

        assert isinstance(complete_url, httpx.URL)
        assert str(complete_url) == f"{api_base}/{endpoint}"
        assert base_target_url == api_base
        assert "version=2024-03-19" not in str(complete_url)

    @patch("litellm.llms.watsonx.common_utils.get_secret_str")
    def test_get_api_base_with_explicit_value(self, mock_get_secret):
        """Test get_api_base returns explicit value when provided."""
        explicit_base = "https://custom.watsonx.com"

        result = WatsonxPassthroughConfig.get_api_base(api_base=explicit_base)

        assert result == explicit_base
        mock_get_secret.assert_not_called()

    @patch("litellm.llms.watsonx.common_utils.get_secret_str")
    def test_get_api_base_from_environment(self, mock_get_secret):
        """Test get_api_base retrieves from environment when not provided."""
        env_base = "https://env.watsonx.com"
        mock_get_secret.return_value = env_base

        result = WatsonxPassthroughConfig.get_api_base(api_base=None)

        assert result == env_base
        mock_get_secret.assert_called_once_with("WATSONX_API_BASE")

    @patch("litellm.llms.watsonx.common_utils.get_secret_str")
    def test_get_api_key_with_explicit_value(self, mock_get_secret):
        """Test get_api_key returns explicit value when provided."""
        explicit_key = "test-api-key-123"

        result = WatsonxPassthroughConfig.get_api_key(api_key=explicit_key)

        assert result == explicit_key
        mock_get_secret.assert_not_called()

    @patch("litellm.llms.watsonx.common_utils.get_secret_str")
    def test_get_api_key_from_environment(self, mock_get_secret):
        """Test get_api_key retrieves from environment when not provided."""
        env_key = "env-api-key-456"
        mock_get_secret.return_value = env_key

        result = WatsonxPassthroughConfig.get_api_key(api_key=None)

        assert result == env_key
        mock_get_secret.assert_any_call("WATSONX_APIKEY")

    def test_get_base_model_returns_model(self):
        """Test get_base_model returns the model as-is."""
        model = "ibm/granite-13b-chat-v2"

        result = WatsonxPassthroughConfig.get_base_model(model)

        assert result == model

    def test_get_base_model_with_deployment(self):
        """Test get_base_model with deployment model."""
        model = "deployment/test-deployment-id"

        result = WatsonxPassthroughConfig.get_base_model(model)

        assert result == model

    def test_get_complete_url_with_different_endpoints(self):
        """Test URL construction with various endpoint paths."""
        config = WatsonxPassthroughConfig()
        api_base = "https://us-south.ml.cloud.ibm.com"

        endpoints = [
            "ml/v1/text/generation",
            "ml/v1/text/tokenization",
            "ml/v1/deployments/test-id/text/generation",
            "ml/v1/models",
            "ml/v1/foundation_model_specs",
        ]

        for endpoint in endpoints:
            complete_url, base_target_url = config.get_complete_url(
                api_base=api_base,
                api_key=None,
                model="",
                endpoint=endpoint,
                request_query_params={"version": "2024-03-19"},
                litellm_params={},
            )

            assert isinstance(complete_url, httpx.URL)
            assert endpoint in str(complete_url)
            assert base_target_url == api_base

    def test_get_complete_url_preserves_query_param_order(self):
        """Test that query parameters maintain their values correctly."""
        config = WatsonxPassthroughConfig()
        api_base = "https://us-south.ml.cloud.ibm.com"
        endpoint = "ml/v1/text/generation"
        request_query_params = {
            "version": "2024-03-19",
            "project_id": "abc-123",
            "space_id": "xyz-789",
        }

        complete_url, _ = config.get_complete_url(
            api_base=api_base,
            api_key=None,
            model="",
            endpoint=endpoint,
            request_query_params=request_query_params,
            litellm_params={},
        )

        url_str = str(complete_url)
        # Verify all params are present
        assert "version=2024-03-19" in url_str
        assert "project_id=abc-123" in url_str
        assert "space_id=xyz-789" in url_str

    def test_is_streaming_request_with_various_stream_values(self):
        """Test streaming detection with different stream value types."""
        config = WatsonxPassthroughConfig()

        # Test with boolean True
        assert config.is_streaming_request("endpoint", {"stream": True}) is True

        # Test with boolean False
        assert config.is_streaming_request("endpoint", {"stream": False}) is False

        # Test with string "true" (truthy string)
        result = config.is_streaming_request("endpoint", {"stream": "true"})
        assert result == "true"  # Returns the value as-is from .get()

        # Test with integer 1 (truthy)
        result = config.is_streaming_request("endpoint", {"stream": 1})
        assert result == 1

        # Test with integer 0 (falsy)
        result = config.is_streaming_request("endpoint", {"stream": 0})
        assert result == 0

        # Test with None
        result = config.is_streaming_request("endpoint", {"stream": None})
        assert result is None

        # Test with empty dict (defaults to False)
        assert config.is_streaming_request("endpoint", {}) is False
