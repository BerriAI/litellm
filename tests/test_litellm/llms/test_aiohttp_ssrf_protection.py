import pytest
from unittest.mock import patch

from litellm.llms.custom_httpx.aiohttp_handler import _assert_not_private_url


class TestAiohttpSSRFProtection:
    def test_aws_metadata_endpoint_blocked(self):
        with pytest.raises(ValueError, match="private/reserved"):
            _assert_not_private_url("http://169.254.169.254/latest/meta-data/")

    def test_localhost_blocked(self):
        with pytest.raises(ValueError, match="private/reserved"):
            _assert_not_private_url("http://127.0.0.1/admin")

    def test_private_10_network_blocked(self):
        with pytest.raises(ValueError, match="private/reserved"):
            _assert_not_private_url("http://10.0.0.1/internal")

    def test_private_172_16_network_blocked(self):
        with pytest.raises(ValueError, match="private/reserved"):
            _assert_not_private_url("http://172.16.0.1/internal")

    def test_private_192_168_network_blocked(self):
        with pytest.raises(ValueError, match="private/reserved"):
            _assert_not_private_url("http://192.168.1.1/internal")

    def test_cgnat_blocked(self):
        with pytest.raises(ValueError, match="private/reserved"):
            _assert_not_private_url("http://100.64.0.1/internal")

    def test_public_ip_allowed(self):
        with patch(
            "socket.getaddrinfo",
            return_value=[(None, None, None, None, ("104.18.7.8", None))],
        ):
            # Should not raise for a public IP
            _assert_not_private_url("https://api.openai.com/v1/chat/completions")

    def test_empty_hostname_allowed(self):
        # No hostname → skip check, no raise
        _assert_not_private_url("not-a-url")

    def test_dns_failure_does_not_block(self):
        import socket as _socket

        with patch("socket.getaddrinfo", side_effect=_socket.gaierror("DNS fail")):
            # DNS failures should not block — let the request fail naturally
            _assert_not_private_url("https://nonexistent.invalid/path")
