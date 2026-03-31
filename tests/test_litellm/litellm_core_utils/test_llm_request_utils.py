"""
Tests for llm_request_utils helper functions.
"""

import pytest

from litellm.litellm_core_utils.llm_request_utils import (
    get_proxy_server_request_headers,
)


class TestGetProxyServerRequestHeaders:
    """Tests for get_proxy_server_request_headers."""

    def test_should_return_empty_dict_when_litellm_params_is_none(self):
        assert get_proxy_server_request_headers(None) == {}

    def test_should_return_empty_dict_when_proxy_server_request_is_none(self):
        """When proxy_server_request is explicitly None (e.g. MCP tool calls),
        should return {} instead of raising 'NoneType has no attribute get'."""
        litellm_params = {"proxy_server_request": None}
        assert get_proxy_server_request_headers(litellm_params) == {}

    def test_should_return_empty_dict_when_proxy_server_request_missing(self):
        litellm_params = {}
        assert get_proxy_server_request_headers(litellm_params) == {}

    def test_should_return_empty_dict_when_headers_is_none(self):
        litellm_params = {"proxy_server_request": {"headers": None}}
        assert get_proxy_server_request_headers(litellm_params) == {}

    def test_should_return_headers_when_present(self):
        expected = {"Authorization": "Bearer sk-123", "Content-Type": "application/json"}
        litellm_params = {"proxy_server_request": {"headers": expected}}
        assert get_proxy_server_request_headers(litellm_params) == expected

    def test_should_return_empty_dict_when_proxy_server_request_has_no_headers_key(self):
        litellm_params = {"proxy_server_request": {"body": "{}"}}
        assert get_proxy_server_request_headers(litellm_params) == {}
