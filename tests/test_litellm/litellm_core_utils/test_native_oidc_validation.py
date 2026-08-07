"""Tests for the shared native OIDC validation primitives.

These primitives are the trust anchor for both the proxy discovery endpoint and
the `lite` CLI, so the security-relevant behaviours (byte-for-byte issuer
preservation, numeric-loopback-only plaintext HTTP, RFC 6749 scope-token ABNF)
are pinned explicitly.
"""

import pytest

from litellm.litellm_core_utils.native_oidc_validation import (
    derive_provider_configuration_url,
    format_scopes,
    has_control_characters,
    is_numeric_loopback_host,
    is_trusted_metadata_origin,
    is_printable_ascii,
    is_valid_nqchar_string,
    is_valid_scope_token,
    validate_endpoint_url,
    validate_issuer,
    validate_scope_tokens,
)


class TestControlCharacters:
    @pytest.mark.parametrize("value", ["\x00", "a\tb", "a\nb", "\x1f", "\x7f", "\x9f"])
    def test_detects_control_characters(self, value):
        assert has_control_characters(value) is True

    @pytest.mark.parametrize("value", ["", "abc", "a b", "https://x/y?z=1", "é"])
    def test_allows_printable_text(self, value):
        assert has_control_characters(value) is False


class TestNumericLoopbackHost:
    @pytest.mark.parametrize("host", ["127.0.0.1", "127.1.2.3", "::1"])
    def test_numeric_loopback_accepted(self, host):
        assert is_numeric_loopback_host(host) is True

    @pytest.mark.parametrize(
        "host",
        [
            "localhost",  # a name is attacker-resolvable; never widens the exception
            "127.0.0.1.evil.com",
            "10.0.0.1",
            "0.0.0.0",
            "::",
            "",
            "not-an-ip",
        ],
    )
    def test_names_and_non_loopback_rejected(self, host):
        assert is_numeric_loopback_host(host) is False


class TestTerminalSafeText:
    @pytest.mark.parametrize(
        "value",
        [
            "\x1b[31mred",
            "\x1b]0;retitled\x07",
            "carriage\rreturn",
            "new\nline",
            "bell\x07",
            "nul\x00",
            "c1\x9b[31m",
            "",
        ],
    )
    def test_escape_sequences_are_not_nqchar(self, value):
        assert is_valid_nqchar_string(value) is False

    @pytest.mark.parametrize("value", ["invalid_grant", "access_denied", "a!b#c[d]e~"])
    def test_oauth_error_codes_are_nqchar(self, value):
        assert is_valid_nqchar_string(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "\x1b[31mred",
            "\x1b]0;retitled\x07",
            "carriage\rreturn",
            "bell\x07",
            "c1\x9b[31m",
            "unicodé",
            "",
        ],
    )
    def test_escape_sequences_are_not_printable_ascii(self, value):
        assert is_printable_ascii(value) is False

    @pytest.mark.parametrize("value", ["WDJB-MJHT", "1234 5678", 'quote"and\\slash'])
    def test_plain_user_codes_are_printable_ascii(self, value):
        assert is_printable_ascii(value) is True


class TestScopeTokens:
    @pytest.mark.parametrize("scope", ["openid", "profile", "api://resource/.default", "a!b#c[d]e~"])
    def test_valid_scope_tokens(self, scope):
        assert is_valid_scope_token(scope) is True

    @pytest.mark.parametrize(
        "scope",
        [
            "",
            "has space",
            'has"quote',
            "has\\backslash",
            "tab\there",
            "new\nline",
            "del\x7f",
            "unicodé",
        ],
    )
    def test_invalid_scope_tokens(self, scope):
        assert is_valid_scope_token(scope) is False

    def test_validate_scope_tokens_preserves_order(self):
        assert validate_scope_tokens(["openid", "email", "profile"]) == (
            "openid",
            "email",
            "profile",
        )

    def test_validate_scope_tokens_rejects_duplicates(self):
        with pytest.raises(ValueError, match="duplicate"):
            validate_scope_tokens(["openid", "openid"])

    def test_validate_scope_tokens_rejects_empty_list(self):
        with pytest.raises(ValueError, match="at least one scope"):
            validate_scope_tokens([])

    def test_validate_scope_tokens_rejects_non_string(self):
        with pytest.raises(ValueError, match="scope-tokens"):
            validate_scope_tokens(["openid", 1])

    def test_error_message_never_echoes_the_offending_value(self):
        with pytest.raises(ValueError) as excinfo:
            validate_scope_tokens(["secret-looking value"])
        assert "secret-looking" not in str(excinfo.value)


class TestValidateIssuer:
    def test_returns_value_byte_for_byte(self):
        issuer = "https://IdP.Example.com:8443/tenant/"
        assert validate_issuer(issuer) == issuer

    @pytest.mark.parametrize(
        "issuer",
        [
            "https://idp.example.com",
            "https://idp.example.com/tenant",
            "https://idp.example.com:8443",
            "http://127.0.0.1:8080",
            "http://[::1]:8080",
        ],
    )
    def test_accepts_valid_issuers(self, issuer):
        assert validate_issuer(issuer) == issuer

    @pytest.mark.parametrize(
        "issuer,message",
        [
            ("", "non-empty"),
            (" https://idp.example.com", "whitespace"),
            ("https://idp.example.com ", "whitespace"),
            ("https://idp.example\n.com", "whitespace or control"),
            ("https://idp.example.com\x00", "whitespace or control"),
            ("ftp://idp.example.com", "absolute http"),
            ("//idp.example.com", "absolute http"),
            ("https://", "host"),
            ("https://user:pass@idp.example.com", "credentials"),
            ("https://idp.example.com#frag", "fragment"),
            ("https://idp.example.com?a=b", "query"),
            ("https://idp.example.com:notaport", "valid port"),
            ("http://idp.example.com", "HTTPS"),
            ("http://localhost:8080", "HTTPS"),
        ],
    )
    def test_rejects_unsafe_issuers(self, issuer, message):
        with pytest.raises(ValueError, match=message):
            validate_issuer(issuer)

    def test_non_string_rejected(self):
        with pytest.raises(ValueError, match="non-empty string"):
            validate_issuer(None)  # type: ignore[arg-type]


class TestValidateEndpointUrl:
    def test_query_component_allowed(self):
        url = "https://idp.example.com/authorize?tenant=a"
        assert validate_endpoint_url(url) == url

    def test_may_live_on_a_different_host_than_the_issuer(self):
        url = "https://login.example.net/token"
        assert validate_endpoint_url(url) == url

    @pytest.mark.parametrize(
        "url,message",
        [
            ("http://idp.example.com/token", "HTTPS"),
            ("https://idp.example.com/token#frag", "fragment"),
            ("https://a:b@idp.example.com/token", "credentials"),
        ],
    )
    def test_rejects_unsafe_endpoints(self, url, message):
        with pytest.raises(ValueError, match=message):
            validate_endpoint_url(url)

    def test_loopback_plaintext_allowed(self):
        url = "http://127.0.0.1:9000/token"
        assert validate_endpoint_url(url) == url


class TestDeriveProviderConfigurationUrl:
    @pytest.mark.parametrize(
        "issuer,expected",
        [
            (
                "https://idp.example.com",
                "https://idp.example.com/.well-known/openid-configuration",
            ),
            (
                "https://idp.example.com/",
                "https://idp.example.com/.well-known/openid-configuration",
            ),
            (
                "https://idp.example.com/tenant",
                "https://idp.example.com/tenant/.well-known/openid-configuration",
            ),
            (
                "https://idp.example.com/tenant/",
                "https://idp.example.com/tenant/.well-known/openid-configuration",
            ),
        ],
    )
    def test_single_trailing_slash_removed(self, issuer, expected):
        assert derive_provider_configuration_url(issuer) == expected

    def test_case_and_port_are_not_normalized(self):
        assert derive_provider_configuration_url("https://IdP.Example.com:8443").startswith(
            "https://IdP.Example.com:8443/"
        )


class TestIsTrustedMetadataOrigin:
    @pytest.mark.parametrize(
        "base_url",
        [
            "https://proxy.example.com",
            "https://proxy.example.com:4000/litellm",
            "http://127.0.0.1:4000",
            "http://[::1]:4000",
        ],
    )
    def test_trusted(self, base_url):
        assert is_trusted_metadata_origin(base_url) is True

    @pytest.mark.parametrize(
        "base_url",
        [
            "http://proxy.example.com",
            "http://localhost:4000",
            "ftp://proxy.example.com",
            "proxy.example.com",
            "",
            "https://proxy.example.com:notaport",
        ],
    )
    def test_untrusted(self, base_url):
        assert is_trusted_metadata_origin(base_url) is False


def test_format_scopes_joins_with_single_space():
    assert format_scopes(("openid", "profile", "email")) == "openid profile email"
