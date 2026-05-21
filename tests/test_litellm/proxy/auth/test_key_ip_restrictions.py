"""
Unit tests for per-key IP address restrictions.

Tests that virtual keys with allowed_ips enforce IP-based access control,
supporting both exact IP addresses and CIDR ranges.
"""

from unittest.mock import MagicMock

import pytest

from litellm.proxy.auth.auth_utils import _check_key_ip_allowed


def _make_request(client_ip="192.168.1.100", xff_header=None):
    request = MagicMock()
    request.client = MagicMock()
    request.client.host = client_ip
    headers = {}
    if xff_header is not None:
        headers["x-forwarded-for"] = xff_header
    request.headers = headers
    return request


class TestCheckKeyIpAllowed:
    """Tests for _check_key_ip_allowed function."""

    def test_should_allow_when_allowed_ips_is_none(self):
        request = _make_request()
        is_allowed, client_ip = _check_key_ip_allowed(allowed_ips=None, request=request)
        assert is_allowed is True
        assert client_ip is None

    def test_should_allow_when_allowed_ips_is_empty(self):
        request = _make_request()
        is_allowed, client_ip = _check_key_ip_allowed(allowed_ips=[], request=request)
        assert is_allowed is True
        assert client_ip is None

    def test_should_allow_exact_ip_match(self):
        request = _make_request(client_ip="10.0.0.5")
        is_allowed, client_ip = _check_key_ip_allowed(
            allowed_ips=["10.0.0.5"], request=request
        )
        assert is_allowed is True
        assert client_ip == "10.0.0.5"

    def test_should_reject_ip_not_in_list(self):
        request = _make_request(client_ip="10.0.0.99")
        is_allowed, client_ip = _check_key_ip_allowed(
            allowed_ips=["10.0.0.5", "10.0.0.10"], request=request
        )
        assert is_allowed is False
        assert client_ip == "10.0.0.99"

    def test_should_allow_ip_in_cidr_range(self):
        request = _make_request(client_ip="10.0.0.42")
        is_allowed, client_ip = _check_key_ip_allowed(
            allowed_ips=["10.0.0.0/24"], request=request
        )
        assert is_allowed is True

    def test_should_reject_ip_outside_cidr_range(self):
        request = _make_request(client_ip="10.0.1.42")
        is_allowed, client_ip = _check_key_ip_allowed(
            allowed_ips=["10.0.0.0/24"], request=request
        )
        assert is_allowed is False

    def test_should_allow_with_mixed_exact_and_cidr(self):
        allowed = ["192.168.1.1", "10.0.0.0/8"]
        request = _make_request(client_ip="10.5.3.2")
        is_allowed, _ = _check_key_ip_allowed(allowed_ips=allowed, request=request)
        assert is_allowed is True

        request2 = _make_request(client_ip="192.168.1.1")
        is_allowed2, _ = _check_key_ip_allowed(allowed_ips=allowed, request=request2)
        assert is_allowed2 is True

        request3 = _make_request(client_ip="172.16.0.1")
        is_allowed3, _ = _check_key_ip_allowed(allowed_ips=allowed, request=request3)
        assert is_allowed3 is False

    def test_should_support_wide_cidr_range(self):
        request = _make_request(client_ip="172.20.5.10")
        is_allowed, _ = _check_key_ip_allowed(
            allowed_ips=["172.16.0.0/12"], request=request
        )
        assert is_allowed is True

    def test_should_reject_ip_outside_wide_cidr(self):
        request = _make_request(client_ip="172.32.0.1")
        is_allowed, _ = _check_key_ip_allowed(
            allowed_ips=["172.16.0.0/12"], request=request
        )
        assert is_allowed is False

    def test_should_use_xff_header_when_enabled(self):
        request = _make_request(client_ip="127.0.0.1", xff_header="203.0.113.50")
        is_allowed, client_ip = _check_key_ip_allowed(
            allowed_ips=["203.0.113.50"],
            request=request,
            use_x_forwarded_for=True,
        )
        assert is_allowed is True
        assert client_ip == "203.0.113.50"

    def test_should_ignore_xff_when_disabled(self):
        request = _make_request(client_ip="10.0.0.1", xff_header="203.0.113.50")
        is_allowed, client_ip = _check_key_ip_allowed(
            allowed_ips=["10.0.0.1"],
            request=request,
            use_x_forwarded_for=False,
        )
        assert is_allowed is True
        assert client_ip == "10.0.0.1"

    def test_should_use_leftmost_xff_ip(self):
        request = _make_request(
            client_ip="127.0.0.1",
            xff_header="203.0.113.50, 10.0.0.1, 192.168.1.1",
        )
        is_allowed, _ = _check_key_ip_allowed(
            allowed_ips=["203.0.113.0/24"],
            request=request,
            use_x_forwarded_for=True,
        )
        assert is_allowed is True

    def test_should_reject_xff_ip_not_in_allowed(self):
        request = _make_request(
            client_ip="127.0.0.1",
            xff_header="8.8.8.8, 10.0.0.1",
        )
        is_allowed, _ = _check_key_ip_allowed(
            allowed_ips=["203.0.113.0/24"],
            request=request,
            use_x_forwarded_for=True,
        )
        assert is_allowed is False

    def test_should_reject_empty_client_ip(self):
        request = MagicMock()
        request.client = None
        request.headers = {}
        is_allowed, client_ip = _check_key_ip_allowed(
            allowed_ips=["10.0.0.0/8"], request=request
        )
        assert is_allowed is False

    def test_should_reject_invalid_client_ip(self):
        request = _make_request(client_ip="not-an-ip")
        is_allowed, _ = _check_key_ip_allowed(
            allowed_ips=["10.0.0.0/8"], request=request
        )
        assert is_allowed is False

    def test_should_skip_invalid_allowed_ip_entries(self):
        request = _make_request(client_ip="10.0.0.1")
        is_allowed, _ = _check_key_ip_allowed(
            allowed_ips=["bad-cidr", "10.0.0.0/24"], request=request
        )
        assert is_allowed is True

    def test_should_support_ipv6_exact(self):
        request = _make_request(client_ip="::1")
        is_allowed, _ = _check_key_ip_allowed(allowed_ips=["::1"], request=request)
        assert is_allowed is True

    def test_should_support_ipv6_cidr(self):
        request = _make_request(client_ip="fd12:3456:789a::1")
        is_allowed, _ = _check_key_ip_allowed(
            allowed_ips=["fd12:3456:789a::/48"], request=request
        )
        assert is_allowed is True

    def test_should_reject_ipv6_outside_cidr(self):
        request = _make_request(client_ip="fd12:3456:789b::1")
        is_allowed, _ = _check_key_ip_allowed(
            allowed_ips=["fd12:3456:789a::/48"], request=request
        )
        assert is_allowed is False

    def test_should_support_single_host_cidr(self):
        request = _make_request(client_ip="10.0.0.5")
        is_allowed, _ = _check_key_ip_allowed(
            allowed_ips=["10.0.0.5/32"], request=request
        )
        assert is_allowed is True

        request2 = _make_request(client_ip="10.0.0.6")
        is_allowed2, _ = _check_key_ip_allowed(
            allowed_ips=["10.0.0.5/32"], request=request2
        )
        assert is_allowed2 is False


class TestKeyIpRestrictionIntegration:
    """Tests that the IP restriction integrates properly with UserAPIKeyAuth."""

    def test_should_populate_allowed_ips_on_token(self):
        from litellm.proxy._types import UserAPIKeyAuth

        token = UserAPIKeyAuth(
            api_key="sk-test",
            allowed_ips=["10.0.0.0/8", "192.168.1.1"],
        )
        assert token.allowed_ips == ["10.0.0.0/8", "192.168.1.1"]

    def test_should_default_allowed_ips_to_none(self):
        from litellm.proxy._types import UserAPIKeyAuth

        token = UserAPIKeyAuth(api_key="sk-test")
        assert token.allowed_ips is None

    def test_should_populate_allowed_ips_on_verification_token(self):
        from litellm.proxy._types import LiteLLM_VerificationToken

        token = LiteLLM_VerificationToken(
            token="hashed-token",
            allowed_ips=["10.0.0.0/8"],
        )
        assert token.allowed_ips == ["10.0.0.0/8"]

    def test_should_include_allowed_ips_in_key_request(self):
        from litellm.proxy._types import GenerateKeyRequest

        req = GenerateKeyRequest(
            allowed_ips=["192.168.0.0/16", "10.0.0.1"],
        )
        assert req.allowed_ips == ["192.168.0.0/16", "10.0.0.1"]

    def test_should_include_allowed_ips_in_update_key_request(self):
        from litellm.proxy._types import UpdateKeyRequest

        req = UpdateKeyRequest(
            key="sk-test",
            allowed_ips=["10.0.0.0/8"],
        )
        assert req.allowed_ips == ["10.0.0.0/8"]

    def test_should_exclude_unset_allowed_ips_from_model_dump(self):
        from litellm.proxy._types import UpdateKeyRequest

        req = UpdateKeyRequest(key="sk-test")
        dumped = req.model_dump(exclude_unset=True)
        assert "allowed_ips" not in dumped

    def test_should_include_set_allowed_ips_in_model_dump(self):
        from litellm.proxy._types import UpdateKeyRequest

        req = UpdateKeyRequest(key="sk-test", allowed_ips=["10.0.0.0/8"])
        dumped = req.model_dump(exclude_unset=True)
        assert dumped["allowed_ips"] == ["10.0.0.0/8"]
