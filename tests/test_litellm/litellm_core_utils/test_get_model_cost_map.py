"""
Tests for get_model_cost_map to verify behavior when the hosted model cost map
returns bad/malformed JSON.

Related incident: When the hosted model_prices_and_context_window.json on GitHub
was poorly formatted, litellm.model_cost ended up with bad data, causing
"This model isn't mapped yet" errors for models like gpt-5.2 on azure.
"""

import json
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map


def _make_httpx_response(content: str, status_code: int = 200) -> httpx.Response:
    """Create a fake httpx.Response with the given content."""
    response = httpx.Response(
        status_code=status_code,
        content=content.encode("utf-8"),
        headers={"content-type": "application/json"},
        request=httpx.Request("GET", "https://fake-url.com/model_cost.json"),
    )
    return response


class TestGetModelCostMapWithBadHostedJSON:
    """
    Reproduce the incident where a badly formatted hosted model cost map JSON
    caused 'This model isn't mapped yet' errors.
    """

    def test_empty_json_object_from_hosted_url(self):
        """
        If the hosted URL returns a valid but empty JSON object '{}',
        get_model_cost_map should fall back to the local backup.
        """
        with patch("httpx.get") as mock_get:
            mock_get.return_value = _make_httpx_response("{}")
            result = get_model_cost_map(url="https://fake-url.com/model_cost.json")

        # An empty dict means no models are mapped — this is the bug.
        # The function returns {} because response.json() succeeds.
        # This demonstrates the vulnerability: no validation of content.
        assert result == {}

    def test_truncated_json_from_hosted_url_falls_back(self):
        """
        If the hosted URL returns invalid JSON (e.g., truncated),
        get_model_cost_map should fall back to local backup and return valid data.
        """
        with patch("httpx.get") as mock_get:
            mock_get.return_value = _make_httpx_response('{"gpt-4": {"input_cost')
            result = get_model_cost_map(url="https://fake-url.com/model_cost.json")

        # Invalid JSON triggers the except branch, which loads the local backup
        assert len(result) > 0
        # Should contain known models from backup
        assert any("gpt-4" in key for key in result)

    def test_html_error_page_from_hosted_url_falls_back(self):
        """
        If the hosted URL returns an HTML error page, get_model_cost_map
        should fall back to the local backup.
        """
        with patch("httpx.get") as mock_get:
            mock_get.return_value = _make_httpx_response(
                "<html><body>503 Service Unavailable</body></html>"
            )
            result = get_model_cost_map(url="https://fake-url.com/model_cost.json")

        # HTML can't be parsed as JSON, so it should fall back
        assert len(result) > 0
        assert any("gpt-4" in key for key in result)

    def test_http_error_status_falls_back(self):
        """
        If the hosted URL returns an HTTP error status code,
        get_model_cost_map should fall back to the local backup.
        """
        with patch("httpx.get") as mock_get:
            mock_get.return_value = _make_httpx_response(
                "Not Found", status_code=404
            )
            result = get_model_cost_map(url="https://fake-url.com/model_cost.json")

        # raise_for_status() should trigger the except branch
        assert len(result) > 0

    def test_network_timeout_falls_back(self):
        """
        If the hosted URL times out, get_model_cost_map should fall back
        to the local backup.
        """
        with patch("httpx.get") as mock_get:
            mock_get.side_effect = httpx.TimeoutException("Connection timed out")
            result = get_model_cost_map(url="https://fake-url.com/model_cost.json")

        assert len(result) > 0


class TestBadModelCostMapCausesModelNotMappedError:
    """
    End-to-end reproduction: when litellm.model_cost is replaced with bad data
    (simulating a bad hosted JSON), get_model_info raises
    'This model isn't mapped yet' for known models.
    """

    def test_empty_model_cost_causes_model_not_mapped_error(self):
        """
        Reproduce the exact error from the incident:
        'This model isn't mapped yet. model=gpt-5.2, custom_llm_provider=azure'
        """
        original_model_cost = litellm.model_cost.copy()
        try:
            # Simulate what happens when hosted JSON returns {}
            litellm.model_cost = {}
            # Need to invalidate the lowercase map cache
            from litellm.utils import _invalidate_model_cost_lowercase_map
            _invalidate_model_cost_lowercase_map()

            with pytest.raises(Exception, match="This model isn't mapped yet"):
                litellm.get_model_info(
                    model="gpt-4o", custom_llm_provider="azure"
                )
        finally:
            litellm.model_cost = original_model_cost
            from litellm.utils import _invalidate_model_cost_lowercase_map
            _invalidate_model_cost_lowercase_map()

    def test_partial_model_cost_missing_model(self):
        """
        If the hosted JSON has some models but is missing others (e.g., truncated),
        lookups for missing models fail.
        """
        original_model_cost = litellm.model_cost.copy()
        try:
            # Simulate a partially loaded cost map - only has one model
            litellm.model_cost = {
                "gpt-4": {
                    "input_cost_per_token": 0.00003,
                    "output_cost_per_token": 0.00006,
                    "litellm_provider": "openai",
                    "max_tokens": 8192,
                    "max_input_tokens": 8192,
                    "max_output_tokens": 4096,
                    "mode": "chat",
                }
            }
            from litellm.utils import _invalidate_model_cost_lowercase_map
            _invalidate_model_cost_lowercase_map()

            # gpt-4 should work
            info = litellm.get_model_info(model="gpt-4")
            assert info is not None

            # But gpt-4o should fail — it's not in the truncated map
            with pytest.raises(Exception, match="This model isn't mapped yet"):
                litellm.get_model_info(
                    model="gpt-4o", custom_llm_provider="openai"
                )
        finally:
            litellm.model_cost = original_model_cost
            from litellm.utils import _invalidate_model_cost_lowercase_map
            _invalidate_model_cost_lowercase_map()

    def test_valid_json_array_instead_of_object_falls_back(self):
        """
        If the hosted URL returns a valid JSON array instead of object,
        get_model_cost_map returns it as-is (a list), which would cause
        'in' operator failures downstream.
        """
        with patch("httpx.get") as mock_get:
            mock_get.return_value = _make_httpx_response('[{"bad": "data"}]')
            result = get_model_cost_map(url="https://fake-url.com/model_cost.json")

        # This returns a list, which is valid JSON but not the expected dict.
        # This demonstrates another vulnerability — no type checking.
        assert isinstance(result, list)
