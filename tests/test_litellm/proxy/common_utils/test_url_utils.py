import pytest

from litellm.proxy.common_utils.url_utils import SSRFError, validate_url


class TestValidateUrl:
    def test_blocks_loopback(self):
        with pytest.raises(SSRFError):
            validate_url("http://127.0.0.1/test")

    def test_blocks_imds(self):
        with pytest.raises(SSRFError):
            validate_url("http://169.254.169.254/latest/meta-data/")

    def test_blocks_rfc1918_class_a(self):
        with pytest.raises(SSRFError):
            validate_url("http://10.0.1.5:8080/v1/completions")

    def test_blocks_rfc1918_class_b(self):
        with pytest.raises(SSRFError):
            validate_url("http://172.16.0.1/")

    def test_blocks_rfc1918_class_c(self):
        with pytest.raises(SSRFError):
            validate_url("http://192.168.1.1/")

    def test_blocks_file_scheme(self):
        with pytest.raises(SSRFError):
            validate_url("file:///etc/passwd")

    def test_blocks_ftp_scheme(self):
        with pytest.raises(SSRFError):
            validate_url("ftp://internal.host/data")

    def test_blocks_no_hostname(self):
        with pytest.raises(SSRFError):
            validate_url("http:///path")

    def test_allows_public_https(self):
        rewritten, host = validate_url("https://example.com/image.png")
        assert host == "example.com"
        assert rewritten == "https://example.com/image.png"

    def test_rewrites_public_http_to_ip(self):
        rewritten, host = validate_url("http://example.com/image.png")
        assert host == "example.com"
        assert "example.com" not in rewritten

    def test_preserves_path_and_query(self):
        rewritten, host = validate_url("http://example.com/path?key=value")
        assert "/path" in rewritten
        assert "key=value" in rewritten

    def test_dns_failure_raises(self):
        with pytest.raises(SSRFError, match="DNS resolution failed"):
            validate_url("http://this-domain-does-not-exist-xyz123.invalid/test")

    def test_blocks_localhost_hostname(self):
        with pytest.raises(SSRFError):
            validate_url("http://localhost/")

    def test_blocks_ipv6_loopback(self):
        with pytest.raises(SSRFError):
            validate_url("http://[::1]/")
