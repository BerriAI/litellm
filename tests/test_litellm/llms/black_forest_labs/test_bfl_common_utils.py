"""
Tests for Black Forest Labs common_utils — specifically assert_bfl_polling_url.

BFL uses regional subdomains (e.g. gateway.bfl.ai) for polling URLs that
differ from the submission host (api.bfl.ai). These tests verify that the
domain-aware check accepts legitimate BFL subdomains while still rejecting
off-domain URLs.
"""

import pytest

from litellm.llms.black_forest_labs.common_utils import (
    BlackForestLabsError,
    assert_bfl_polling_url,
)


class TestAssertBflPollingUrl:
    # --- should pass ---

    def test_exact_registered_domain(self):
        assert_bfl_polling_url("https://bfl.ai/v1/get_result?id=abc")

    def test_api_subdomain(self):
        assert_bfl_polling_url("https://api.bfl.ai/v1/get_result?id=abc")

    def test_gateway_subdomain(self):
        # BFL uses gateway.bfl.ai for polling — this was the original bug trigger
        assert_bfl_polling_url("https://gateway.bfl.ai/v1/get_result?id=abc")

    def test_regional_subdomain(self):
        assert_bfl_polling_url("https://eu.api.bfl.ai/v1/get_result?id=abc")

    def test_deep_subdomain(self):
        assert_bfl_polling_url("https://region.gateway.bfl.ai/poll?id=xyz")

    def test_http_scheme_allowed(self):
        # http is permitted (not ideal, but matches allowed_schemes in url_utils)
        assert_bfl_polling_url("http://api.bfl.ai/v1/get_result?id=abc")

    # --- should raise BlackForestLabsError ---

    def test_rejects_off_domain(self):
        with pytest.raises(BlackForestLabsError, match="host is not within"):
            assert_bfl_polling_url("https://evil.com/steal-key")

    def test_rejects_lookalike_domain(self):
        with pytest.raises(BlackForestLabsError, match="host is not within"):
            assert_bfl_polling_url("https://notbfl.ai/v1/get_result?id=abc")

    def test_rejects_bfl_ai_as_suffix_only(self):
        # "fakebfl.ai" must not match — the check is on registered domain boundary
        with pytest.raises(BlackForestLabsError, match="host is not within"):
            assert_bfl_polling_url("https://fakebfl.ai/v1/get_result?id=abc")

    def test_rejects_bfl_in_path(self):
        with pytest.raises(BlackForestLabsError, match="host is not within"):
            assert_bfl_polling_url("https://evil.com/bfl.ai/steal")

    def test_rejects_ftp_scheme(self):
        with pytest.raises(BlackForestLabsError, match="scheme is not allowed"):
            assert_bfl_polling_url("ftp://api.bfl.ai/v1/get_result?id=abc")

    def test_rejects_javascript_scheme(self):
        with pytest.raises(BlackForestLabsError, match="scheme is not allowed"):
            assert_bfl_polling_url("javascript://api.bfl.ai/alert(1)")
