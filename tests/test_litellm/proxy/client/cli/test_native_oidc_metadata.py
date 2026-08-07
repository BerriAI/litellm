"""Tests for the bounded HTTP helpers and the two metadata trust levels.

`http_client` is the only place the CLI talks to the network during login, so
its safety properties (no redirects, byte ceiling, JSON-only decoding, no raw
bodies in errors) are pinned here alongside the LiteLLM-owned `native_oidc`
contract and the third-party provider configuration document.
"""

import json

import pytest
import requests

from litellm.proxy.client.cli.native_oidc import (
    http_client,
)
from litellm.proxy.client.cli.native_oidc import (
    metadata as metadata_module,
)
from litellm.proxy.client.cli.native_oidc.errors import (
    NativeOIDCError,
    NativeOIDCUnavailable,
)
from litellm.proxy.client.cli.native_oidc.metadata import (
    NativeOIDCMetadata,
    ProviderMetadata,
    build_litellm_discovery_url,
    fetch_native_oidc_metadata,
    fetch_provider_metadata,
    parse_native_oidc_metadata,
    parse_provider_metadata,
)

ISSUER = "https://idp.example.com"
CLIENT_ID = "litellm-cli"
BASE_URL = "https://proxy.example.com"
DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"


class FakeResponse:
    """Minimal stand-in for a streamed `requests.Response`."""

    def __init__(self, status_code=200, body=b"", content_type="application/json", headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = {"content-type": content_type, **(headers or {})}
        self.closed = False

    def iter_content(self, chunk_size=8192):
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.closed = True


def json_response(payload, **kwargs):
    return FakeResponse(body=json.dumps(payload).encode("utf-8"), **kwargs)


@pytest.fixture
def capture_request(monkeypatch):
    """Patch `requests.request` and record the call arguments."""
    calls = []

    def install(response_or_error):
        def fake_request(method, url, **kwargs):
            calls.append({"method": method, "url": url, **kwargs})
            if isinstance(response_or_error, Exception):
                raise response_or_error
            return response_or_error

        monkeypatch.setattr(http_client.requests, "request", fake_request)
        return calls

    return install


class TestRequestSafety:
    def test_redirects_are_never_followed(self, capture_request):
        calls = capture_request(json_response({}))
        http_client.get_json_response(f"{BASE_URL}/x")
        assert calls[0]["allow_redirects"] is False

    def test_timeouts_and_streaming_are_always_set(self, capture_request):
        calls = capture_request(json_response({}))
        http_client.get_json_response(f"{BASE_URL}/x")
        assert calls[0]["timeout"] == http_client.DEFAULT_TIMEOUT
        assert calls[0]["stream"] is True

    @pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
    def test_3xx_is_an_error(self, capture_request, status):
        capture_request(FakeResponse(status_code=status))
        with pytest.raises(NativeOIDCError, match="redirects are not followed"):
            http_client.get_json_response(f"{BASE_URL}/x")

    def test_oversized_body_is_refused(self, capture_request):
        oversized = b"x" * (http_client.MAX_RESPONSE_BYTES + 1)
        capture_request(FakeResponse(body=oversized))
        with pytest.raises(NativeOIDCError, match="byte limit"):
            http_client.get_json_response(f"{BASE_URL}/x")

    def test_empty_chunks_do_not_corrupt_the_body(self, capture_request):
        class ChunkedResponse(FakeResponse):
            # A chunked/keep-alive response can yield empty chunks.
            def iter_content(self, chunk_size=8192):
                yield b'{"a":'
                yield b""
                yield b" 1}"

        capture_request(ChunkedResponse())
        assert http_client.get_json_response(f"{BASE_URL}/x").payload == {"a": 1}

    def test_body_at_the_limit_is_accepted(self, capture_request):
        padding = "y" * (http_client.MAX_RESPONSE_BYTES - 20)
        capture_request(json_response({"a": padding}))
        assert http_client.get_json_response(f"{BASE_URL}/x").payload == {"a": padding}

    def test_transport_error_does_not_leak_details(self, capture_request):
        capture_request(requests.ConnectionError("dial tcp 10.0.0.1:443 refused"))
        with pytest.raises(NativeOIDCError) as excinfo:
            http_client.get_json_response(f"{BASE_URL}/x")
        assert "10.0.0.1" not in str(excinfo.value)
        assert "ConnectionError" in str(excinfo.value)


class TestJsonDecoding:
    @pytest.mark.parametrize(
        "content_type",
        ["application/json", "application/json; charset=utf-8", "application/jwk+json"],
    )
    def test_json_content_types_decoded(self, capture_request, content_type):
        capture_request(json_response({"a": 1}, content_type=content_type))
        assert http_client.get_json_response(f"{BASE_URL}/x").payload == {"a": 1}

    @pytest.mark.parametrize("content_type", ["text/html", "application/xml", ""])
    def test_non_json_content_type_yields_no_payload(self, capture_request, content_type):
        capture_request(json_response({"a": 1}, content_type=content_type))
        assert http_client.get_json_response(f"{BASE_URL}/x").payload is None

    @pytest.mark.parametrize(
        "body",
        [b'{"a":1} {"b":2}', b"[1,2,3]", b'"a string"', b"not json", b"\xff\xfe"],
    )
    def test_non_object_or_malformed_bodies_yield_no_payload(self, capture_request, body):
        capture_request(FakeResponse(body=body))
        assert http_client.get_json_response(f"{BASE_URL}/x").payload is None


class TestRetryAfter:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("7", 7),
            ("  7 ", 7),
            ("0", 0),
            ("100000", http_client.MAX_RETRY_AFTER_SECONDS),
            ("-1", None),
            ("Wed, 21 Oct 2015 07:28:00 GMT", None),
            ("", None),
            (None, None),
        ],
    )
    def test_parsing(self, capture_request, raw, expected):
        headers = {} if raw is None else {"retry-after": raw}
        capture_request(json_response({}, headers=headers))
        assert http_client.get_json_response(f"{BASE_URL}/x").retry_after == expected


class TestGetJsonAndPostForm:
    def test_get_json_requires_200(self, capture_request):
        capture_request(json_response({"a": 1}, status_code=500))
        with pytest.raises(NativeOIDCError, match="HTTP 500"):
            http_client.get_json(f"{BASE_URL}/x")

    def test_get_json_requires_a_json_object(self, capture_request):
        capture_request(FakeResponse(body=b"<html/>", content_type="text/html"))
        with pytest.raises(NativeOIDCError, match="did not return a JSON object"):
            http_client.get_json(f"{BASE_URL}/x")

    def test_post_form_sends_form_encoding_and_returns_error_bodies(self, capture_request):
        calls = capture_request(json_response({"error": "invalid_grant"}, status_code=400))
        response = http_client.post_form(f"{ISSUER}/token", {"grant_type": "x"})
        assert calls[0]["method"] == "POST"
        assert calls[0]["data"] == {"grant_type": "x"}
        assert calls[0]["headers"]["Content-Type"] == "application/x-www-form-urlencoded"
        # A 4xx OAuth error body is data for the caller, not an exception.
        assert response.status_code == 400
        assert response.payload == {"error": "invalid_grant"}


class TestParseNativeOIDCMetadata:
    def test_valid_document(self):
        parsed = parse_native_oidc_metadata({"issuer": ISSUER, "client_id": CLIENT_ID, "scopes": ["openid", "profile"]})
        assert parsed == NativeOIDCMetadata(issuer=ISSUER, client_id=CLIENT_ID, scopes=("openid", "profile"))

    def test_unknown_fields_are_rejected(self):
        with pytest.raises(NativeOIDCError, match="unsupported field\\(s\\): client_secret"):
            parse_native_oidc_metadata(
                {
                    "issuer": ISSUER,
                    "client_id": CLIENT_ID,
                    "scopes": ["openid"],
                    "client_secret": "nope",
                }
            )

    @pytest.mark.parametrize(
        "document,message",
        [
            ({"client_id": CLIENT_ID, "scopes": ["openid"]}, "issuer is missing"),
            (
                {"issuer": 1, "client_id": CLIENT_ID, "scopes": ["openid"]},
                "issuer is missing",
            ),
            (
                {
                    "issuer": "http://idp.example.com",
                    "client_id": CLIENT_ID,
                    "scopes": ["openid"],
                },
                "issuer must use HTTPS",
            ),
            ({"issuer": ISSUER, "scopes": ["openid"]}, "client_id is missing or blank"),
            (
                {"issuer": ISSUER, "client_id": "  ", "scopes": ["openid"]},
                "client_id is missing or blank",
            ),
            (
                {"issuer": ISSUER, "client_id": CLIENT_ID},
                "scopes is missing or not a list",
            ),
            (
                {"issuer": ISSUER, "client_id": CLIENT_ID, "scopes": "openid"},
                "scopes is missing or not a list",
            ),
            (
                {"issuer": ISSUER, "client_id": CLIENT_ID, "scopes": []},
                "scopes must contain at least one",
            ),
            (
                {"issuer": ISSUER, "client_id": CLIENT_ID, "scopes": ["open id"]},
                "scopes must contain only",
            ),
        ],
    )
    def test_invalid_documents(self, document, message):
        with pytest.raises(NativeOIDCError, match=message):
            parse_native_oidc_metadata(document)

    def test_non_object_rejected(self):
        with pytest.raises(NativeOIDCError, match="not an object"):
            parse_native_oidc_metadata(["issuer"])


class TestBuildDiscoveryUrl:
    @pytest.mark.parametrize(
        "base_url,expected",
        [
            (BASE_URL, f"{BASE_URL}/.well-known/litellm-ui-config"),
            (f"{BASE_URL}/", f"{BASE_URL}/.well-known/litellm-ui-config"),
            # A configured path prefix must survive; urljoin would drop it.
            (
                f"{BASE_URL}/litellm",
                f"{BASE_URL}/litellm/.well-known/litellm-ui-config",
            ),
        ],
    )
    def test_path_prefix_preserved(self, base_url, expected):
        assert build_litellm_discovery_url(base_url) == expected


class TestFetchNativeOIDCMetadata:
    def test_plaintext_http_origin_refused(self):
        with pytest.raises(NativeOIDCError, match="requires an HTTPS proxy URL"):
            fetch_native_oidc_metadata("http://proxy.example.com")

    def test_loopback_http_origin_allowed(self, capture_request):
        capture_request(
            json_response(
                {
                    "native_oidc": {
                        "issuer": ISSUER,
                        "client_id": CLIENT_ID,
                        "scopes": ["openid"],
                    }
                }
            )
        )
        assert fetch_native_oidc_metadata("http://127.0.0.1:4000").issuer == ISSUER

    @pytest.mark.parametrize("status", [404, 405])
    def test_old_proxy_is_unavailable_not_an_error(self, capture_request, status):
        capture_request(FakeResponse(status_code=status))
        with pytest.raises(NativeOIDCUnavailable, match="predates native OIDC"):
            fetch_native_oidc_metadata(BASE_URL)

    def test_absent_native_oidc_object_is_unavailable(self, capture_request):
        capture_request(json_response({"sso_enabled": True}))
        with pytest.raises(NativeOIDCUnavailable, match="does not advertise"):
            fetch_native_oidc_metadata(BASE_URL)

    def test_other_status_codes_are_hard_failures(self, capture_request):
        capture_request(FakeResponse(status_code=500))
        with pytest.raises(NativeOIDCError, match="HTTP 500") as excinfo:
            fetch_native_oidc_metadata(BASE_URL)
        assert not isinstance(excinfo.value, NativeOIDCUnavailable)

    def test_non_json_body_is_a_hard_failure(self, capture_request):
        capture_request(FakeResponse(body=b"<html/>", content_type="text/html"))
        with pytest.raises(NativeOIDCError, match="did not return a JSON object") as excinfo:
            fetch_native_oidc_metadata(BASE_URL)
        assert not isinstance(excinfo.value, NativeOIDCUnavailable)

    def test_malformed_native_oidc_object_is_a_hard_failure(self, capture_request):
        capture_request(json_response({"native_oidc": {"issuer": ISSUER}}))
        with pytest.raises(NativeOIDCError) as excinfo:
            fetch_native_oidc_metadata(BASE_URL)
        assert not isinstance(excinfo.value, NativeOIDCUnavailable)

    def test_unrelated_top_level_fields_are_ignored(self, capture_request):
        capture_request(
            json_response(
                {
                    "sso_enabled": True,
                    "some_future_field": {"a": 1},
                    "native_oidc": {
                        "issuer": ISSUER,
                        "client_id": CLIENT_ID,
                        "scopes": ["openid"],
                    },
                }
            )
        )
        assert fetch_native_oidc_metadata(BASE_URL).client_id == CLIENT_ID


def provider_document(**overrides):
    document = {
        "issuer": ISSUER,
        "authorization_endpoint": f"{ISSUER}/authorize",
        "token_endpoint": f"{ISSUER}/token",
        "device_authorization_endpoint": f"{ISSUER}/device",
    }
    document.update(overrides)
    return document


class TestParseProviderMetadata:
    def test_issuer_must_match_exactly(self):
        with pytest.raises(NativeOIDCError, match="does not exactly match"):
            parse_provider_metadata(provider_document(issuer=f"{ISSUER}/"), expected_issuer=ISSUER)

    def test_case_differences_are_not_normalized_away(self):
        with pytest.raises(NativeOIDCError, match="does not exactly match"):
            parse_provider_metadata(
                provider_document(issuer="https://IDP.example.com"),
                expected_issuer=ISSUER,
            )

    def test_unknown_fields_are_ignored(self):
        parsed = parse_provider_metadata(
            provider_document(jwks_uri=f"{ISSUER}/jwks", future_extension=[1, 2]),
            expected_issuer=ISSUER,
        )
        assert parsed.issuer == ISSUER

    def test_missing_issuer(self):
        with pytest.raises(NativeOIDCError, match="issuer is missing"):
            parse_provider_metadata({"token_endpoint": f"{ISSUER}/token"}, expected_issuer=ISSUER)

    def test_non_object(self):
        with pytest.raises(NativeOIDCError, match="not a JSON object"):
            parse_provider_metadata("nope", expected_issuer=ISSUER)

    def test_malformed_list_fields(self):
        with pytest.raises(NativeOIDCError, match="grant_types_supported is not a list of strings"):
            parse_provider_metadata(
                provider_document(grant_types_supported=["code", 7]),
                expected_issuer=ISSUER,
            )

    def test_malformed_endpoint_type(self):
        with pytest.raises(NativeOIDCError, match="token_endpoint is not a string"):
            parse_provider_metadata(provider_document(token_endpoint=7), expected_issuer=ISSUER)

    def test_endpoints_are_validated_lazily(self):
        # An unusable device endpoint must not break a browser login.
        parsed = parse_provider_metadata(
            provider_document(device_authorization_endpoint="http://evil.example.com/device"),
            expected_issuer=ISSUER,
        )
        assert parsed.require_authorization_endpoint() == f"{ISSUER}/authorize"
        with pytest.raises(NativeOIDCError, match="device_authorization_endpoint must use HTTPS"):
            parsed.require_device_authorization_endpoint()

    def test_require_endpoint_reports_absence(self):
        parsed = parse_provider_metadata(
            {"issuer": ISSUER, "token_endpoint": f"{ISSUER}/token"},
            expected_issuer=ISSUER,
        )
        with pytest.raises(NativeOIDCError, match="does not advertise a authorization_endpoint"):
            parsed.require_authorization_endpoint()


def make_provider(**overrides):
    fields = {
        "issuer": ISSUER,
        "authorization_endpoint": f"{ISSUER}/authorize",
        "token_endpoint": f"{ISSUER}/token",
        "device_authorization_endpoint": f"{ISSUER}/device",
        "response_types_supported": None,
        "grant_types_supported": None,
        "code_challenge_methods_supported": None,
        "token_endpoint_auth_methods_supported": None,
    }
    fields.update(overrides)
    return ProviderMetadata(**fields)


class TestFlowSupport:
    def test_absent_capability_metadata_is_permissive(self):
        provider = make_provider()
        assert provider.supports_browser_flow() is True
        assert provider.supports_device_flow() is True

    @pytest.mark.parametrize(
        "auth_methods",
        [
            ("private_key_jwt", "client_secret_basic", "client_secret_post", "tls_client_auth", "client_secret_jwt"),
            ("client_secret_basic",),
            ("client_secret_post",),
        ],
    )
    def test_token_endpoint_auth_methods_without_none_do_not_block_a_flow(self, auth_methods):
        """Regression: Keycloak omits 'none' from token_endpoint_auth_methods_supported.

        The first parameter set is Keycloak 26's literal advertisement. Keycloak
        nonetheless accepts public clients at its token endpoint, so treating a
        missing 'none' as a hard precondition made `lite login` impossible against
        one of the most widely deployed identity providers. Client authentication
        is decided by the token request itself, which fails loudly with an OAuth
        error if the provider really does require a secret.
        """
        provider = make_provider(
            response_types_supported=("code",),
            grant_types_supported=("authorization_code", DEVICE_GRANT),
            code_challenge_methods_supported=("S256",),
            token_endpoint_auth_methods_supported=auth_methods,
        )
        assert provider.supports_browser_flow() is True
        assert provider.supports_device_flow() is True
        provider.assert_browser_flow_supported()
        provider.assert_device_flow_supported()

    @pytest.mark.parametrize(
        "overrides,message",
        [
            (
                {"authorization_endpoint": None},
                "does not advertise an authorization_endpoint",
            ),
            ({"token_endpoint": None}, "does not advertise a token_endpoint"),
            (
                {"code_challenge_methods_supported": ("plain",)},
                "does not support PKCE S256",
            ),
            ({"response_types_supported": ("token",)}, "'code' response type"),
            (
                {"grant_types_supported": ("client_credentials",)},
                "authorization_code grant",
            ),
        ],
    )
    def test_browser_flow_rejections(self, overrides, message):
        provider = make_provider(**overrides)
        assert provider.supports_browser_flow() is False
        with pytest.raises(NativeOIDCError, match=message):
            provider.assert_browser_flow_supported()

    @pytest.mark.parametrize(
        "overrides,message",
        [
            (
                {"device_authorization_endpoint": None},
                "does not advertise a device_authorization_endpoint",
            ),
            ({"token_endpoint": None}, "does not advertise a token_endpoint"),
            ({"grant_types_supported": ("authorization_code",)}, "device_code grant"),
        ],
    )
    def test_device_flow_rejections(self, overrides, message):
        provider = make_provider(**overrides)
        assert provider.supports_device_flow() is False
        with pytest.raises(NativeOIDCError, match=message):
            provider.assert_device_flow_supported()

    def test_explicit_support_is_accepted(self):
        provider = make_provider(
            response_types_supported=("code",),
            grant_types_supported=("authorization_code", DEVICE_GRANT),
            code_challenge_methods_supported=("S256", "plain"),
            token_endpoint_auth_methods_supported=("none",),
        )
        provider.assert_browser_flow_supported()
        provider.assert_device_flow_supported()
        assert provider.supports_browser_flow() is True
        assert provider.supports_device_flow() is True


class TestFetchProviderMetadata:
    def test_fetches_the_derived_configuration_url(self, capture_request):
        calls = capture_request(json_response(provider_document()))
        parsed = fetch_provider_metadata(ISSUER)
        assert calls[0]["url"] == f"{ISSUER}/.well-known/openid-configuration"
        assert parsed.issuer == ISSUER

    def test_issuer_mismatch_is_rejected(self, capture_request):
        capture_request(json_response(provider_document(issuer="https://evil.example.com")))
        with pytest.raises(NativeOIDCError, match="does not exactly match"):
            fetch_provider_metadata(ISSUER)


def test_format_scope_list_matches_the_shared_helper():
    assert metadata_module.format_scope_list(("openid", "email")) == "openid email"
