"""
Unit tests for MCP IP-based access control.

Tests that internal callers see all MCP servers while
external callers only see servers with available_on_public_internet=True.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Request
from pydantic import ValidationError

import litellm.proxy.auth.ip_address_utils as ip_mod
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ConfigGeneralSettings
from litellm.proxy.auth.ip_address_utils import (
    IPAddressUtils,
    _HopCount,
    _HopCountInvalid,
    _HopCountUnset,
)
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def _make_server(server_id, available_on_public_internet=False):
    return MCPServer(
        server_id=server_id,
        name=server_id,
        server_name=server_id,
        transport="http",
        available_on_public_internet=available_on_public_internet,
    )


def _make_manager(servers):
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )

    manager = MCPServerManager()
    for s in servers:
        manager.config_mcp_servers[s.server_id] = s
    return manager


class TestIsInternalIp:
    """Tests that IP classification works for private, public, and edge cases."""

    def test_private_ranges_are_internal(self):
        assert IPAddressUtils.is_internal_ip("127.0.0.1") is True
        assert IPAddressUtils.is_internal_ip("10.0.0.1") is True
        assert IPAddressUtils.is_internal_ip("172.16.0.1") is True
        assert IPAddressUtils.is_internal_ip("192.168.1.1") is True
        assert IPAddressUtils.is_internal_ip("::1") is True

    def test_public_ips_are_external(self):
        assert IPAddressUtils.is_internal_ip("8.8.8.8") is False
        assert IPAddressUtils.is_internal_ip("1.1.1.1") is False
        assert IPAddressUtils.is_internal_ip("172.32.0.1") is False

    def test_xff_chain_uses_leftmost_ip(self):
        assert IPAddressUtils.is_internal_ip("8.8.8.8, 10.0.0.1") is False
        assert IPAddressUtils.is_internal_ip("10.0.0.1, 8.8.8.8") is True

    def test_fails_closed_on_bad_input(self):
        assert IPAddressUtils.is_internal_ip("") is False
        assert IPAddressUtils.is_internal_ip(None) is False
        assert IPAddressUtils.is_internal_ip("not-an-ip") is False


class TestMCPClientIPExtraction:
    def test_fails_closed_when_xff_enabled_without_trusted_proxy_ranges(self):
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "203.0.113.5"
        request.headers = {"x-forwarded-for": "10.0.0.1"}

        result = IPAddressUtils.get_mcp_client_ip(
            request,
            general_settings={"use_x_forwarded_for": True},
        )

        # XFF is untrusted (no mcp_trusted_proxy_ranges) so it must be ignored,
        # and we must not trust the direct peer either: fail closed so the caller
        # is classified as external and is_internal_ip("") is False.
        assert result == ""
        assert IPAddressUtils.is_internal_ip(result) is False

    def test_no_trusted_ranges_warning_matches_fail_closed_behavior(self, caplog):
        ip_mod._warned_xff_without_trusted_ranges = False

        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "203.0.113.5"
        request.headers = {"x-forwarded-for": "1.2.3.4, 10.0.0.1"}
        general_settings = {"use_x_forwarded_for": True}

        with caplog.at_level(logging.WARNING, logger=verbose_proxy_logger.name):
            assert (
                IPAddressUtils.is_request_from_trusted_proxy(
                    request, general_settings=general_settings
                )
                is False
            )

        warning = next(
            (
                record.getMessage()
                for record in caplog.records
                if "mcp_trusted_proxy_ranges" in record.getMessage()
            ),
            None,
        )
        assert (
            warning is not None
        ), "Expected a warning containing 'mcp_trusted_proxy_ranges' but none was logged"

        assert "fails closed" in warning
        assert "treated as external" in warning
        assert "client IPs will use the proxy's literal request values" not in warning

        assert (
            IPAddressUtils.get_mcp_client_ip(request, general_settings=general_settings)
            == ""
        )

    def test_private_proxy_peer_does_not_grant_internal_access(self):
        # Regression: behind an internal reverse proxy with use_x_forwarded_for
        # enabled but mcp_trusted_proxy_ranges unset, the direct peer is the
        # proxy's private IP. Returning it would mis-classify an external caller
        # as internal and expose available_on_public_internet=false servers.
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "10.0.0.7"
        request.headers = {"x-forwarded-for": "8.8.8.8"}

        result = IPAddressUtils.get_mcp_client_ip(
            request,
            general_settings={"use_x_forwarded_for": True},
        )

        assert result == ""
        assert IPAddressUtils.is_internal_ip(result) is False

    def test_honours_xff_from_trusted_proxy(self):
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "10.0.0.5"
        request.headers = {"x-forwarded-for": "192.168.1.10"}

        result = IPAddressUtils.get_mcp_client_ip(
            request,
            general_settings={
                "use_x_forwarded_for": True,
                "mcp_trusted_proxy_ranges": ["10.0.0.0/8"],
            },
        )

        assert result == "192.168.1.10"

    def test_ignores_xff_from_untrusted_direct_caller(self):
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "203.0.113.5"
        request.headers = {"x-forwarded-for": "10.0.0.1"}

        result = IPAddressUtils.get_mcp_client_ip(
            request,
            general_settings={
                "use_x_forwarded_for": True,
                "mcp_trusted_proxy_ranges": ["10.0.0.0/8"],
            },
        )

        assert result == "203.0.113.5"


def _make_request(client_host, xff):
    request = MagicMock(spec=Request)
    request.client = MagicMock()
    request.client.host = client_host
    request.headers = {"x-forwarded-for": xff}
    return request


class TestExtractClientIpFromXffHops:
    def test_single_hop_picks_rightmost(self):
        assert (
            IPAddressUtils.extract_client_ip_from_xff_hops(
                "10.0.0.99, 203.0.113.9", num_trusted_hops=1
            )
            == "203.0.113.9"
        )

    def test_two_hops_picks_second_from_right(self):
        assert (
            IPAddressUtils.extract_client_ip_from_xff_hops(
                "10.0.0.99, 203.0.113.9, 172.16.0.1", num_trusted_hops=2
            )
            == "203.0.113.9"
        )

    def test_whitespace_and_empty_entries_are_ignored(self):
        assert (
            IPAddressUtils.extract_client_ip_from_xff_hops(
                " 1.1.1.1 ,  , 203.0.113.9 ", num_trusted_hops=1
            )
            == "203.0.113.9"
        )

    def test_chain_shorter_than_hops_returns_none(self):
        assert (
            IPAddressUtils.extract_client_ip_from_xff_hops(
                "203.0.113.9", num_trusted_hops=2
            )
            is None
        )

    def test_zero_or_negative_hops_returns_none(self):
        assert (
            IPAddressUtils.extract_client_ip_from_xff_hops(
                "203.0.113.9", num_trusted_hops=0
            )
            is None
        )

    def test_invalid_selected_entry_returns_none(self):
        assert (
            IPAddressUtils.extract_client_ip_from_xff_hops(
                "not-an-ip, 10.0.0.1", num_trusted_hops=2
            )
            is None
        )


class TestResolveNumTrustedHops:
    def test_unset_is_unset(self):
        assert IPAddressUtils._resolve_num_trusted_hops(None) == _HopCountUnset()

    def test_int_value(self):
        assert IPAddressUtils._resolve_num_trusted_hops(2) == _HopCount(2)

    def test_numeric_string_is_coerced(self):
        assert IPAddressUtils._resolve_num_trusted_hops("3") == _HopCount(3)

    def test_zero_is_invalid_not_disabled(self):
        assert IPAddressUtils._resolve_num_trusted_hops(0) == _HopCountInvalid()

    def test_non_numeric_value_is_invalid(self):
        assert IPAddressUtils._resolve_num_trusted_hops("abc") == _HopCountInvalid()

    def test_below_minimum_warns_so_misconfig_is_visible(self):
        with patch(
            "litellm.proxy.auth.ip_address_utils.verbose_proxy_logger"
        ) as mock_logger:
            assert IPAddressUtils._resolve_num_trusted_hops(0) == _HopCountInvalid()
            assert IPAddressUtils._resolve_num_trusted_hops(-3) == _HopCountInvalid()
        assert mock_logger.warning.call_count == 2

    def test_unset_does_not_warn(self):
        with patch(
            "litellm.proxy.auth.ip_address_utils.verbose_proxy_logger"
        ) as mock_logger:
            assert IPAddressUtils._resolve_num_trusted_hops(None) == _HopCountUnset()
        mock_logger.warning.assert_not_called()

    def test_valid_value_does_not_warn(self):
        with patch(
            "litellm.proxy.auth.ip_address_utils.verbose_proxy_logger"
        ) as mock_logger:
            assert IPAddressUtils._resolve_num_trusted_hops(2) == _HopCount(2)
        mock_logger.warning.assert_not_called()


class TestConfigGeneralSettingsHopsValidation:
    """mcp_xff_num_trusted_hops must reject sub-minimum values at config-parse time."""

    @pytest.mark.parametrize("bad_value", [0, -1])
    def test_below_minimum_is_rejected(self, bad_value):
        with pytest.raises(ValidationError):
            ConfigGeneralSettings(mcp_xff_num_trusted_hops=bad_value)

    def test_valid_value_and_unset_are_accepted(self):
        assert (
            ConfigGeneralSettings(mcp_xff_num_trusted_hops=1).mcp_xff_num_trusted_hops
            == 1
        )
        assert ConfigGeneralSettings().mcp_xff_num_trusted_hops is None


class TestXffTrustedHopsAccessControl:
    def test_spoofed_internal_leftmost_is_defeated(self):
        request = _make_request("10.0.0.5", "10.0.0.99, 203.0.113.9")

        result = IPAddressUtils.get_mcp_client_ip(
            request,
            general_settings={
                "use_x_forwarded_for": True,
                "mcp_trusted_proxy_ranges": ["10.0.0.0/8"],
                "mcp_xff_num_trusted_hops": 1,
            },
        )

        assert result == "203.0.113.9"
        assert IPAddressUtils.is_internal_ip(result) is False

    def test_genuine_internal_client_is_preserved(self):
        request = _make_request("10.0.0.5", "10.0.0.50")

        result = IPAddressUtils.get_mcp_client_ip(
            request,
            general_settings={
                "use_x_forwarded_for": True,
                "mcp_trusted_proxy_ranges": ["10.0.0.0/8"],
                "mcp_xff_num_trusted_hops": 1,
            },
        )

        assert result == "10.0.0.50"
        assert IPAddressUtils.is_internal_ip(result) is True

    def test_two_hops_skips_proxy_appended_addresses(self):
        request = _make_request("10.0.0.5", "10.0.0.99, 203.0.113.9, 172.16.0.1")

        result = IPAddressUtils.get_mcp_client_ip(
            request,
            general_settings={
                "use_x_forwarded_for": True,
                "mcp_trusted_proxy_ranges": ["10.0.0.0/8"],
                "mcp_xff_num_trusted_hops": 2,
            },
        )

        assert result == "203.0.113.9"

    def test_short_chain_fails_closed(self):
        request = _make_request("10.0.0.5", "203.0.113.9")

        result = IPAddressUtils.get_mcp_client_ip(
            request,
            general_settings={
                "use_x_forwarded_for": True,
                "mcp_trusted_proxy_ranges": ["10.0.0.0/8"],
                "mcp_xff_num_trusted_hops": 2,
            },
        )

        assert result == ""
        assert IPAddressUtils.is_internal_ip(result) is False

    def test_hops_without_trusted_ranges_still_fails_closed(self):
        request = _make_request("203.0.113.5", "10.0.0.99, 8.8.8.8")

        result = IPAddressUtils.get_mcp_client_ip(
            request,
            general_settings={
                "use_x_forwarded_for": True,
                "mcp_xff_num_trusted_hops": 1,
            },
        )

        assert result == ""

    def test_unset_hops_keeps_legacy_leftmost_behavior(self):
        request = _make_request("10.0.0.5", "10.0.0.99, 203.0.113.9")

        result = IPAddressUtils.get_mcp_client_ip(
            request,
            general_settings={
                "use_x_forwarded_for": True,
                "mcp_trusted_proxy_ranges": ["10.0.0.0/8"],
            },
        )

        assert result == "10.0.0.99, 203.0.113.9"

    @pytest.mark.parametrize("bad_value", [0, -1, "abc", 1.5])
    def test_invalid_hops_config_fails_closed_not_legacy(self, bad_value):
        request = _make_request("10.0.0.5", "10.0.0.99, 203.0.113.9")

        result = IPAddressUtils.get_mcp_client_ip(
            request,
            general_settings={
                "use_x_forwarded_for": True,
                "mcp_trusted_proxy_ranges": ["10.0.0.0/8"],
                "mcp_xff_num_trusted_hops": bad_value,
            },
        )

        assert result == ""
        assert IPAddressUtils.is_internal_ip(result) is False


class TestXffPresentButDisabledWarning:
    """When an XFF header arrives but use_x_forwarded_for is off, the proxy must
    loudly warn (the internal-only check is silently trusting the load balancer's
    IP) yet still serve the request, so a crafted header can't DoS a no-LB deploy."""

    def _reset_warning_flag(self):
        from litellm.proxy.auth import ip_address_utils

        ip_address_utils._warned_xff_present_but_disabled = False

    def _request_with_xff(self):
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "10.0.0.7"
        request.headers = {"x-forwarded-for": "8.8.8.8"}
        return request

    def test_warns_and_does_not_fail_when_xff_present_but_disabled(self):
        self._reset_warning_flag()
        request = self._request_with_xff()

        with patch(
            "litellm.proxy.auth.ip_address_utils.verbose_proxy_logger.error"
        ) as mock_error:
            result = IPAddressUtils.get_mcp_client_ip(
                request, general_settings={"use_x_forwarded_for": False}
            )

        # Does not hard-fail: falls back to the direct peer (the load balancer).
        assert result == "10.0.0.7"
        mock_error.assert_called_once()
        assert "use_x_forwarded_for" in str(mock_error.call_args)

    def test_warning_is_one_shot(self):
        self._reset_warning_flag()

        with patch(
            "litellm.proxy.auth.ip_address_utils.verbose_proxy_logger.error"
        ) as mock_error:
            IPAddressUtils.get_mcp_client_ip(
                self._request_with_xff(),
                general_settings={"use_x_forwarded_for": False},
            )
            IPAddressUtils.get_mcp_client_ip(
                self._request_with_xff(),
                general_settings={"use_x_forwarded_for": False},
            )

        # One-shot so a flood of crafted XFF headers cannot spam the logs.
        mock_error.assert_called_once()

    def test_re_arms_after_xff_is_enabled_then_disabled_again(self):
        self._reset_warning_flag()

        with patch(
            "litellm.proxy.auth.ip_address_utils.verbose_proxy_logger.error"
        ) as mock_error:
            IPAddressUtils.get_mcp_client_ip(
                self._request_with_xff(),
                general_settings={"use_x_forwarded_for": False},
            )
            # Operator fixes the config; observing it enabled re-arms the warning.
            IPAddressUtils.get_mcp_client_ip(
                self._request_with_xff(),
                general_settings={
                    "use_x_forwarded_for": True,
                    "mcp_trusted_proxy_ranges": ["10.0.0.0/8"],
                },
            )
            # Config rolls back to disabled: the misconfiguration must warn again.
            IPAddressUtils.get_mcp_client_ip(
                self._request_with_xff(),
                general_settings={"use_x_forwarded_for": False},
            )

        assert mock_error.call_count == 2

    def test_no_warning_without_xff_header(self):
        self._reset_warning_flag()
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "10.0.0.7"
        request.headers = {}

        with patch(
            "litellm.proxy.auth.ip_address_utils.verbose_proxy_logger.error"
        ) as mock_error:
            IPAddressUtils.get_mcp_client_ip(
                request, general_settings={"use_x_forwarded_for": False}
            )

        mock_error.assert_not_called()

    def test_no_warning_when_xff_enabled(self):
        self._reset_warning_flag()

        with patch(
            "litellm.proxy.auth.ip_address_utils.verbose_proxy_logger.error"
        ) as mock_error:
            IPAddressUtils.get_mcp_client_ip(
                self._request_with_xff(),
                general_settings={
                    "use_x_forwarded_for": True,
                    "mcp_trusted_proxy_ranges": ["10.0.0.0/8"],
                },
            )

        mock_error.assert_not_called()


class TestMCPServerIPFiltering:
    """Tests that external callers only see public MCP servers."""

    @patch("litellm.public_mcp_servers", [])
    @patch("litellm.proxy.proxy_server.general_settings", {})
    def test_external_ip_only_sees_public_servers(self):
        pub = _make_server("pub", available_on_public_internet=True)
        priv = _make_server("priv", available_on_public_internet=False)
        manager = _make_manager([pub, priv])

        result = manager.filter_server_ids_by_ip(["pub", "priv"], client_ip="8.8.8.8")
        assert result == ["pub"]

    @patch("litellm.public_mcp_servers", [])
    @patch("litellm.proxy.proxy_server.general_settings", {})
    def test_internal_ip_sees_all_servers(self):
        pub = _make_server("pub", available_on_public_internet=True)
        priv = _make_server("priv", available_on_public_internet=False)
        manager = _make_manager([pub, priv])

        result = manager.filter_server_ids_by_ip(
            ["pub", "priv"], client_ip="192.168.1.1"
        )
        assert result == ["pub", "priv"]

    @patch("litellm.public_mcp_servers", [])
    @patch("litellm.proxy.proxy_server.general_settings", {})
    def test_no_ip_means_no_filtering(self):
        priv = _make_server("priv", available_on_public_internet=False)
        manager = _make_manager([priv])

        result = manager.filter_server_ids_by_ip(["priv"], client_ip=None)
        assert result == ["priv"]


class TestFilterServerIdsByIpWithInfo:
    """Tests that filter_server_ids_by_ip_with_info returns accurate block counts."""

    @patch("litellm.public_mcp_servers", [])
    @patch("litellm.proxy.proxy_server.general_settings", {})
    def test_external_ip_reports_blocked_count(self):
        pub = _make_server("pub", available_on_public_internet=True)
        priv = _make_server("priv", available_on_public_internet=False)
        manager = _make_manager([pub, priv])

        allowed, blocked = manager.filter_server_ids_by_ip_with_info(
            ["pub", "priv"], client_ip="8.8.8.8"
        )
        assert allowed == ["pub"]
        assert blocked == 1

    @patch("litellm.public_mcp_servers", [])
    @patch("litellm.proxy.proxy_server.general_settings", {})
    def test_internal_ip_reports_zero_blocked(self):
        pub = _make_server("pub", available_on_public_internet=True)
        priv = _make_server("priv", available_on_public_internet=False)
        manager = _make_manager([pub, priv])

        allowed, blocked = manager.filter_server_ids_by_ip_with_info(
            ["pub", "priv"], client_ip="192.168.1.1"
        )
        assert allowed == ["pub", "priv"]
        assert blocked == 0

    @patch("litellm.public_mcp_servers", [])
    @patch("litellm.proxy.proxy_server.general_settings", {})
    def test_no_ip_returns_all_with_zero_blocked(self):
        priv = _make_server("priv", available_on_public_internet=False)
        manager = _make_manager([priv])

        allowed, blocked = manager.filter_server_ids_by_ip_with_info(
            ["priv"], client_ip=None
        )
        assert allowed == ["priv"]
        assert blocked == 0

    @patch("litellm.public_mcp_servers", [])
    @patch("litellm.proxy.proxy_server.general_settings", {})
    def test_all_private_external_ip_reports_all_blocked(self):
        priv1 = _make_server("priv1", available_on_public_internet=False)
        priv2 = _make_server("priv2", available_on_public_internet=False)
        manager = _make_manager([priv1, priv2])

        allowed, blocked = manager.filter_server_ids_by_ip_with_info(
            ["priv1", "priv2"], client_ip="1.2.3.4"
        )
        assert allowed == []
        assert blocked == 2
