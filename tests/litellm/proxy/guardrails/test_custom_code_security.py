import pytest
from unittest.mock import patch
from litellm.proxy.guardrails.guardrail_hooks.custom_code.code_validator import (
    validate_custom_code,
    CustomCodeValidationError,
)
from litellm.proxy.guardrails.guardrail_hooks.custom_code.custom_code_guardrail import (
    CustomCodeGuardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.custom_code.primitives import (
    _validate_url_for_ssrf,
)

# Phase 4.1: Test forbidden pattern validation


def test_validate_custom_code_import_os():
    code = "import os\ndef apply_guardrail(inputs, req, ty):\n    return allow()"
    with pytest.raises(CustomCodeValidationError, match="import statements are not"):
        validate_custom_code(code)


def test_validate_custom_code_from_subprocess():
    code = (
        "from subprocess import call\ndef apply_guardrail(i, r, t):\n    return allow()"
    )
    with pytest.raises(
        CustomCodeValidationError, match="import statements are not allowed"
    ):
        validate_custom_code(code)


def test_validate_custom_code_exec():
    code = "def apply_guardrail(i, r, t):\n    exec('print(1)')\n    return allow()"
    with pytest.raises(CustomCodeValidationError, match=r"exec\(\) is not allowed"):
        validate_custom_code(code)


def test_validate_custom_code_builtins():
    code = "def apply_guardrail(i, r, t):\n    print(__builtins__)\n    return allow()"
    with pytest.raises(
        CustomCodeValidationError, match="__builtins__ access is not allowed"
    ):
        validate_custom_code(code)


def test_validate_custom_code_subclasses():
    code = "def apply_guardrail(i, r, t):\n    print(''.__class__.__mro__[1].__subclasses__())\n    return allow()"
    with pytest.raises(
        CustomCodeValidationError, match="__subclasses__ access is not allowed"
    ):
        validate_custom_code(code)


def test_validate_custom_code_clean():
    code = (
        "def apply_guardrail(inputs, request_data, input_type):\n    return allow()\n"
    )
    # Should not raise any exception
    validate_custom_code(code)


# Phase 4.2: Test __builtins__ restriction in execution


def test_custom_code_compile_valid():
    code = "def apply_guardrail(inputs, request_data, input_type):\n    return allow()"
    guardrail = CustomCodeGuardrail(custom_code=code, guardrail_name="test")
    # if it doesn't fail, we successfully compiled
    assert guardrail._compiled_function is not None


def test_custom_code_override_builtins():
    # Verify that even if pattern validation is bypassed, __builtins__ = {} blocks dangerous builtins.
    # We test this by compiling safe code and verifying builtins are not accessible in the sandbox.
    code = "def apply_guardrail(inputs, request_data, input_type):\n    return allow()"
    guardrail = CustomCodeGuardrail(custom_code=code, guardrail_name="test")
    # The compiled function's globals should have empty __builtins__
    fn_globals = guardrail._compiled_function.__globals__
    assert fn_globals.get("__builtins__") == {}


@pytest.mark.asyncio
async def test_custom_code_guardrail_apply():
    code = "def apply_guardrail(inputs, request_data, input_type):\n    return allow()"
    guardrail = CustomCodeGuardrail(custom_code=code, guardrail_name="test")
    from litellm.types.utils import GenericGuardrailAPIInputs

    result = await guardrail.apply_guardrail(
        inputs=GenericGuardrailAPIInputs(texts=["test"]),
        request_data={},
        input_type="request",
    )
    assert result["texts"][0] == "test"


# =============================================================================
# Phase 4.3: SSRF Protection Tests
# =============================================================================


def _mock_resolve(hostname, port=None, proto=None):
    """Helper: simulate DNS resolution returning specific IPs."""

    def _resolver(host, p, **kwargs):
        if host != hostname:
            raise OSError(f"unexpected host: {host}")
        import socket

        return [
            (socket.AF_INET, socket.SOCK_STREAM, proto or 6, "", (ip, p or 443))
            for ip in (port if isinstance(port, list) else [port])
        ]

    return _resolver


class TestSSRFValidation:
    """Tests for _validate_url_for_ssrf SSRF protection."""

    def test_should_block_localhost(self):
        """Loopback addresses must be blocked."""
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.custom_code.primitives.socket.getaddrinfo"
        ) as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("127.0.0.1", 443)),
            ]
            result = _validate_url_for_ssrf("http://localhost/secret")
            assert result is not None
            assert "loopback" in result.lower()

    def test_should_block_127_0_0_1(self):
        """Direct 127.0.0.1 must be blocked."""
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.custom_code.primitives.socket.getaddrinfo"
        ) as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("127.0.0.1", 80)),
            ]
            result = _validate_url_for_ssrf("http://127.0.0.1/")
            assert result is not None
            assert "loopback" in result.lower()

    def test_should_block_ipv6_loopback(self):
        """IPv6 loopback ::1 must be blocked."""
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.custom_code.primitives.socket.getaddrinfo"
        ) as mock_dns:
            mock_dns.return_value = [
                (10, 1, 6, "", ("::1", 443, 0, 0)),
            ]
            result = _validate_url_for_ssrf("http://[::1]/")
            assert result is not None
            assert "loopback" in result.lower()

    def test_should_block_private_10_network(self):
        """RFC1918 10.0.0.0/8 must be blocked."""
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.custom_code.primitives.socket.getaddrinfo"
        ) as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("10.0.0.1", 8080)),
            ]
            result = _validate_url_for_ssrf("http://10.0.0.1:8080/admin")
            assert result is not None
            assert "private" in result.lower()

    def test_should_block_private_172_network(self):
        """RFC1918 172.16.0.0/12 must be blocked."""
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.custom_code.primitives.socket.getaddrinfo"
        ) as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("172.16.0.1", 443)),
            ]
            result = _validate_url_for_ssrf("http://172.16.0.1/")
            assert result is not None
            assert "private" in result.lower()

    def test_should_block_private_192_168_network(self):
        """RFC1918 192.168.0.0/16 must be blocked."""
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.custom_code.primitives.socket.getaddrinfo"
        ) as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("192.168.1.1", 443)),
            ]
            result = _validate_url_for_ssrf("http://192.168.1.1/")
            assert result is not None
            assert "private" in result.lower()

    def test_should_block_aws_metadata_ip(self):
        """AWS metadata IP 169.254.169.254 must be blocked by hostname blocklist."""
        result = _validate_url_for_ssrf(
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/"
        )
        assert result is not None
        assert "metadata" in result.lower()

    def test_should_block_gcp_metadata_hostname(self):
        """GCP metadata hostname must be blocked by hostname blocklist."""
        result = _validate_url_for_ssrf(
            "http://metadata.google.internal/computeMetadata/v1/"
        )
        assert result is not None
        assert "metadata" in result.lower()

    def test_should_block_link_local(self):
        """Link-local 169.254.x.x (non-metadata) must be blocked."""
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.custom_code.primitives.socket.getaddrinfo"
        ) as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("169.254.1.1", 443)),
            ]
            result = _validate_url_for_ssrf("http://169.254.1.1/")
            assert result is not None
            assert "not allowed" in result.lower()

    def test_should_block_dns_rebinding_to_private(self):
        """A public hostname that resolves to a private IP must be blocked."""
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.custom_code.primitives.socket.getaddrinfo"
        ) as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("10.0.0.5", 443)),
            ]
            result = _validate_url_for_ssrf("http://evil-rebind.example.com/")
            assert result is not None
            assert "private" in result.lower()

    def test_should_allow_public_url(self):
        """Legitimate public URLs must be allowed."""
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.custom_code.primitives.socket.getaddrinfo"
        ) as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("93.184.216.34", 443)),
            ]
            result = _validate_url_for_ssrf("https://example.com/api/check")
            assert result is None

    def test_should_allow_public_ip(self):
        """Direct public IPs must be allowed."""
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.custom_code.primitives.socket.getaddrinfo"
        ) as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("8.8.8.8", 443)),
            ]
            result = _validate_url_for_ssrf("https://8.8.8.8/")
            assert result is None

    def test_should_block_unresolvable_hostname(self):
        """Unresolvable hostnames should be blocked."""
        import socket

        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.custom_code.primitives.socket.getaddrinfo"
        ) as mock_dns:
            mock_dns.side_effect = socket.gaierror("Name resolution failed")
            result = _validate_url_for_ssrf("http://nonexistent.invalid/")
            assert result is not None
            assert "resolve" in result.lower()

    def test_should_block_metadata_hostname_with_trailing_dot(self):
        """Trailing dot normalization must not bypass hostname blocklist."""
        result = _validate_url_for_ssrf(
            "http://metadata.google.internal./computeMetadata/v1/"
        )
        assert result is not None
        assert "metadata" in result.lower()

    def test_should_reject_url_with_no_hostname(self):
        """URLs without a hostname must be rejected."""
        result = _validate_url_for_ssrf("file:///etc/passwd")
        assert result is not None

    def test_should_block_unspecified_address(self):
        """0.0.0.0 must be blocked."""
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.custom_code.primitives.socket.getaddrinfo"
        ) as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("0.0.0.0", 80)),
            ]
            result = _validate_url_for_ssrf("http://0.0.0.0/")
            assert result is not None


class TestSSRFPocBlocked:
    """Verify that the specific SSRF PoC payloads are now blocked."""

    def test_should_block_aws_metadata_poc(self):
        """The AWS metadata SSRF PoC must be blocked."""
        result = _validate_url_for_ssrf(
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/"
        )
        assert result is not None

    def test_should_block_gcp_metadata_poc(self):
        """The GCP metadata SSRF PoC must be blocked."""
        result = _validate_url_for_ssrf(
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
        )
        assert result is not None

    def test_should_block_internal_scan_poc(self):
        """Internal network scanning targets must be blocked."""
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.custom_code.primitives.socket.getaddrinfo"
        ) as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("10.0.0.1", 8080)),
            ]
            result = _validate_url_for_ssrf("http://10.0.0.1:8080/")
            assert result is not None

        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.custom_code.primitives.socket.getaddrinfo"
        ) as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("192.168.1.1", 443)),
            ]
            result = _validate_url_for_ssrf("http://192.168.1.1/admin")
            assert result is not None

        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.custom_code.primitives.socket.getaddrinfo"
        ) as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("127.0.0.1", 6379)),
            ]
            result = _validate_url_for_ssrf("http://localhost:6379/")
            assert result is not None
