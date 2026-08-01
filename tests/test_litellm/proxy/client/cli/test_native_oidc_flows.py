"""Tests for PKCE, the loopback callback listener, token parsing, and the flows.

These cover the parts of the native OIDC login that handle secrets: the PKCE
pair, the authorization code arriving over loopback, and the token response.
"""

import base64
import hashlib
import json
import threading
import time
from http.client import HTTPConnection
from urllib.parse import parse_qs, urlsplit

import pytest

from litellm.proxy.client.cli.native_oidc import browser_flow, callback, device_flow
from litellm.proxy.client.cli.native_oidc.browser_flow import (
    build_authorization_url,
    exchange_code_for_token,
    run_browser_flow,
)
from litellm.proxy.client.cli.native_oidc.callback import (
    DEFAULT_CALLBACK_PATH,
    LoopbackCallbackListener,
    sanitize_provider_error,
)
from litellm.proxy.client.cli.native_oidc.device_flow import (
    DEFAULT_POLL_INTERVAL_SECONDS,
    MAX_POLL_INTERVAL_SECONDS,
    DeviceAuthorization,
    parse_device_authorization,
    poll_for_device_token,
    request_device_authorization,
    run_device_flow,
)
from litellm.proxy.client.cli.native_oidc.errors import NativeOIDCError
from litellm.proxy.client.cli.native_oidc.http_client import JsonResponse
from litellm.proxy.client.cli.native_oidc.metadata import (
    NativeOIDCMetadata,
    ProviderMetadata,
)
from litellm.proxy.client.cli.native_oidc.pkce import (
    CODE_VERIFIER_MAX_LENGTH,
    CODE_VERIFIER_MIN_LENGTH,
    compute_code_challenge,
    generate_code_verifier,
    generate_pkce_challenge,
    generate_state,
    states_match,
)
from litellm.proxy.client.cli.native_oidc.tokens import (
    FALLBACK_LIFETIME_SECONDS,
    MAX_EXPIRES_IN_SECONDS,
    compute_expires_at,
    describe_token_error,
    extract_oauth_error,
    parse_token_response,
)

ISSUER = "https://idp.example.com"
CLIENT_ID = "litellm-cli"
DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"

METADATA = NativeOIDCMetadata(issuer=ISSUER, client_id=CLIENT_ID, scopes=("openid", "profile"))
PROVIDER = ProviderMetadata(
    issuer=ISSUER,
    authorization_endpoint=f"{ISSUER}/authorize",
    token_endpoint=f"{ISSUER}/token",
    device_authorization_endpoint=f"{ISSUER}/device",
    response_types_supported=("code",),
    grant_types_supported=("authorization_code", DEVICE_GRANT),
    code_challenge_methods_supported=("S256",),
    token_endpoint_auth_methods_supported=("none",),
)


def make_jwt(payload):
    def segment(data):
        return base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")

    return f"{segment({'alg': 'none'})}.{segment(payload)}.signature"


# --------------------------------------------------------------------------- #
# PKCE
# --------------------------------------------------------------------------- #


class TestPkce:
    def test_verifier_length_within_rfc_7636_bounds(self):
        verifier = generate_code_verifier()
        assert CODE_VERIFIER_MIN_LENGTH <= len(verifier) <= CODE_VERIFIER_MAX_LENGTH

    def test_verifier_uses_only_unreserved_characters(self):
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
        assert set(generate_code_verifier()) <= allowed

    def test_verifiers_and_states_are_unique(self):
        assert len({generate_code_verifier() for _ in range(50)}) == 50
        assert len({generate_state() for _ in range(50)}) == 50

    def test_challenge_is_unpadded_base64url_sha256(self):
        verifier = "abc123"
        expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        challenge = compute_code_challenge(verifier)
        assert challenge == expected
        assert "=" not in challenge
        assert "+" not in challenge and "/" not in challenge

    def test_generated_pair_is_consistent_and_s256_only(self):
        challenge = generate_pkce_challenge()
        assert challenge.code_challenge_method == "S256"
        assert challenge.code_challenge == compute_code_challenge(challenge.code_verifier)
        assert challenge.state and challenge.state != challenge.code_verifier

    @pytest.mark.parametrize(
        "expected,received,matches",
        [
            ("abc", "abc", True),
            ("abc", "abd", False),
            ("abc", "ab", False),
            ("", "", True),
        ],
    )
    def test_states_match(self, expected, received, matches):
        assert states_match(expected, received) is matches


# --------------------------------------------------------------------------- #
# Loopback callback listener
# --------------------------------------------------------------------------- #


def call_listener(listener, path, headers=None):
    """Issue one request against the listener and serve exactly one request."""
    captured = {}

    def client():
        connection = HTTPConnection(listener.host, listener.port, timeout=5)
        try:
            connection.request("GET", path, headers=headers or {})
            response = connection.getresponse()
            captured["status"] = response.status
            response.read()
        finally:
            connection.close()

    thread = threading.Thread(target=client)
    thread.start()
    listener._server.handle_request()
    thread.join(5)
    return captured.get("status")


class TestCallbackListener:
    def test_binds_loopback_with_an_ephemeral_port(self):
        with LoopbackCallbackListener(expected_state="s") as listener:
            assert listener.host in ("127.0.0.1", "::1")
            assert listener.port > 0
            assert listener.redirect_uri.endswith(DEFAULT_CALLBACK_PATH)
            # Built from the address actually bound, never from a guess.
            assert str(listener.port) in listener.redirect_uri

    def test_successful_callback_returns_the_code(self):
        with LoopbackCallbackListener(expected_state="state-value") as listener:
            result = {}

            def waiter():
                result["code"] = listener.wait_for_code(timeout=5)

            thread = threading.Thread(target=waiter)
            thread.start()
            connection = HTTPConnection(listener.host, listener.port, timeout=5)
            connection.request("GET", f"{DEFAULT_CALLBACK_PATH}?code=abc&state=state-value")
            assert connection.getresponse().status == 200
            connection.close()
            thread.join(5)
            assert result["code"] == "abc"

    def test_state_mismatch_is_rejected_before_the_code_is_used(self):
        with LoopbackCallbackListener(expected_state="expected") as listener:
            assert call_listener(listener, f"{DEFAULT_CALLBACK_PATH}?code=abc&state=wrong") == 400
            # Refused without a terminal result: any local process that guessed the
            # ephemeral port could otherwise abort a login in progress.
            assert listener._server.result is None
            with pytest.raises(NativeOIDCError, match="timed out"):
                listener.wait_for_code(timeout=0.1)

    def test_missing_state_is_rejected(self):
        with LoopbackCallbackListener(expected_state="expected") as listener:
            assert call_listener(listener, f"{DEFAULT_CALLBACK_PATH}?code=abc") == 400
            assert listener._server.result is None
            with pytest.raises(NativeOIDCError, match="timed out"):
                listener.wait_for_code(timeout=0.1)

    def test_duplicate_parameters_are_rejected(self):
        with LoopbackCallbackListener(expected_state="s") as listener:
            assert call_listener(listener, f"{DEFAULT_CALLBACK_PATH}?code=a&code=b&state=s") == 400
            assert listener._server.result is None
            with pytest.raises(NativeOIDCError, match="timed out"):
                listener.wait_for_code(timeout=0.1)

    def test_a_forged_callback_does_not_prevent_the_real_one(self):
        with LoopbackCallbackListener(expected_state="expected") as listener:
            assert call_listener(listener, f"{DEFAULT_CALLBACK_PATH}?code=forged&state=wrong") == 400
            assert call_listener(listener, f"{DEFAULT_CALLBACK_PATH}?code=real&state=expected") == 200
            assert listener.wait_for_code(timeout=0.1) == "real"

    def test_provider_error_is_surfaced_after_state_validation(self):
        with LoopbackCallbackListener(expected_state="s") as listener:
            assert (
                call_listener(
                    listener,
                    f"{DEFAULT_CALLBACK_PATH}?error=access_denied&error_description=user+said+no&state=s",
                )
                == 400
            )
            with pytest.raises(NativeOIDCError, match="access_denied: user said no"):
                listener.wait_for_code(timeout=0.1)

    def test_missing_code_is_rejected(self):
        with LoopbackCallbackListener(expected_state="s") as listener:
            assert call_listener(listener, f"{DEFAULT_CALLBACK_PATH}?code=&state=s") == 400
            with pytest.raises(NativeOIDCError, match="did not contain a code"):
                listener.wait_for_code(timeout=0.1)

    def test_other_paths_are_not_served(self):
        with LoopbackCallbackListener(expected_state="s") as listener:
            assert call_listener(listener, "/?code=abc&state=s") == 404
            assert listener._server.result is None

    def test_mismatched_host_header_is_rejected(self):
        with LoopbackCallbackListener(expected_state="s") as listener:
            status = call_listener(
                listener,
                f"{DEFAULT_CALLBACK_PATH}?code=abc&state=s",
                headers={"Host": "evil.example.com"},
            )
            assert status == 400
            assert listener._server.result is None

    def test_only_one_authorization_response_is_accepted(self):
        with LoopbackCallbackListener(expected_state="s") as listener:
            assert call_listener(listener, f"{DEFAULT_CALLBACK_PATH}?code=first&state=s") == 200
            assert call_listener(listener, f"{DEFAULT_CALLBACK_PATH}?code=second&state=s") == 400
            assert listener.wait_for_code(timeout=0.1) == "first"

    def test_timeout_raises(self):
        with LoopbackCallbackListener(expected_state="s") as listener:
            with pytest.raises(NativeOIDCError, match="timed out"):
                listener.wait_for_code(timeout=0.1)

    def test_response_page_never_contains_the_code_or_state(self):
        with LoopbackCallbackListener(expected_state="state-secret") as listener:
            body = {}

            def client():
                connection = HTTPConnection(listener.host, listener.port, timeout=5)
                connection.request(
                    "GET",
                    f"{DEFAULT_CALLBACK_PATH}?code=code-secret&state=state-secret",
                )
                body["text"] = connection.getresponse().read().decode()
                connection.close()

            thread = threading.Thread(target=client)
            thread.start()
            listener._server.handle_request()
            thread.join(5)
            assert "code-secret" not in body["text"]
            assert "state-secret" not in body["text"]

    def test_non_loopback_peers_are_rejected(self, monkeypatch):
        # The peer check runs before the path, the state or the code is read.
        monkeypatch.setattr(callback, "is_numeric_loopback_host", lambda host: False)
        with LoopbackCallbackListener(expected_state="state-value") as listener:
            status = call_listener(listener, f"{DEFAULT_CALLBACK_PATH}?code=abc&state=state-value")
            assert status == 403
            assert listener._server.result is None

    def test_a_result_without_a_code_is_still_refused(self):
        with LoopbackCallbackListener(expected_state="s") as listener:
            listener._server.result = callback._CallbackResult()
            with pytest.raises(NativeOIDCError, match="did not contain a code"):
                listener.wait_for_code(timeout=5)


class _FakeV6Server:
    """Stands in for a real ::1 bind, which not every CI host provides."""

    def __init__(self, server_address, handler, *, expected_state, callback_path):
        self.server_address = ("::1", 54321)
        self.expected_state = expected_state
        self.callback_path = callback_path
        self.expected_host_header = ""
        self.result = None
        self.closed = False

    def server_close(self):
        self.closed = True


def _refuse_bind(*args, **kwargs):
    raise OSError("cannot bind")


class TestCallbackBinding:
    def test_ipv6_loopback_is_used_when_ipv4_cannot_bind(self, monkeypatch):
        monkeypatch.setattr(callback, "_CallbackServer", _refuse_bind)
        monkeypatch.setattr(callback, "_CallbackServerV6", _FakeV6Server)
        with LoopbackCallbackListener(expected_state="s") as listener:
            # The IPv6 literal must be bracketed in both the URI and the
            # Host header the handler compares against.
            assert listener.host == "::1"
            assert listener.redirect_uri == f"http://[::1]:54321{DEFAULT_CALLBACK_PATH}"
            assert listener._server.expected_host_header == "[::1]:54321"
        assert listener._server.closed

    def test_no_loopback_port_available_is_reported(self, monkeypatch):
        monkeypatch.setattr(callback, "_CallbackServer", _refuse_bind)
        monkeypatch.setattr(callback, "_CallbackServerV6", _refuse_bind)
        with pytest.raises(NativeOIDCError, match="could not bind a loopback port"):
            LoopbackCallbackListener(expected_state="s")


class TestSanitizeProviderError:
    def test_control_characters_stripped(self):
        assert sanitize_provider_error("bad\x00error\n", None) == "baderror"

    def test_description_is_appended_and_bounded(self):
        message = sanitize_provider_error("invalid_request", "x" * 500)
        assert message.startswith("invalid_request: ")
        assert len(message) <= len("invalid_request: ") + 200

    def test_empty_error_falls_back(self):
        assert sanitize_provider_error("\x00", None) == "unknown_error"

    def test_unprintable_description_is_dropped(self):
        assert sanitize_provider_error("invalid_scope", "\x00\x01") == "invalid_scope"


# --------------------------------------------------------------------------- #
# Token responses
# --------------------------------------------------------------------------- #


class TestParseTokenResponse:
    def test_valid_response(self):
        parsed = parse_token_response(
            {
                "access_token": "at",
                "token_type": "Bearer",
                "expires_in": 300,
                "refresh_token": "rt",
                "scope": "openid profile",
            },
            now=1000.0,
        )
        assert parsed.access_token == "at"
        assert parsed.token_type == "Bearer"
        assert parsed.expires_at == 1300.0
        assert parsed.refresh_token == "rt"
        assert parsed.scopes == ("openid", "profile")

    @pytest.mark.parametrize("token_type", ["bearer", "Bearer", "BEARER"])
    def test_token_type_is_case_insensitive_and_normalized(self, token_type):
        parsed = parse_token_response({"access_token": "at", "token_type": token_type})
        assert parsed.token_type == "Bearer"

    @pytest.mark.parametrize(
        "payload,message",
        [
            ({"token_type": "Bearer"}, "did not contain an access_token"),
            (
                {"access_token": "", "token_type": "Bearer"},
                "did not contain an access_token",
            ),
            (
                {"access_token": 1, "token_type": "Bearer"},
                "did not contain an access_token",
            ),
            ({"access_token": "at"}, "did not contain a token_type"),
            ({"access_token": "at", "token_type": "DPoP"}, "unsupported token_type"),
            (
                {"access_token": "at", "token_type": "Bearer", "refresh_token": ""},
                "refresh_token is not a non-empty string",
            ),
            (
                {"access_token": "at", "token_type": "Bearer", "expires_in": 0},
                "expires_in is out of range",
            ),
            (
                {"access_token": "at", "token_type": "Bearer", "expires_in": -5},
                "expires_in is out of range",
            ),
            (
                {
                    "access_token": "at",
                    "token_type": "Bearer",
                    "expires_in": MAX_EXPIRES_IN_SECONDS + 1,
                },
                "expires_in is out of range",
            ),
            (
                {"access_token": "at", "token_type": "Bearer", "expires_in": True},
                "expires_in is not a positive integer",
            ),
            (
                {"access_token": "at", "token_type": "Bearer", "expires_in": "abc"},
                "expires_in is not a positive integer",
            ),
            (
                {"access_token": "at", "token_type": "Bearer", "expires_in": 1.5},
                "expires_in is not a positive integer",
            ),
            (
                {"access_token": "at", "token_type": "Bearer", "scope": 7},
                "scope is not a string",
            ),
            (
                {"access_token": "at", "token_type": "Bearer", "scope": 'bad"scope'},
                "invalid scope-token",
            ),
        ],
    )
    def test_invalid_responses(self, payload, message):
        with pytest.raises(NativeOIDCError, match=message):
            parse_token_response(payload)

    def test_numeric_string_expires_in_accepted(self):
        parsed = parse_token_response(
            {"access_token": "at", "token_type": "Bearer", "expires_in": " 600 "},
            now=0.0,
        )
        assert parsed.expires_at == 600.0

    def test_non_object_payload(self):
        with pytest.raises(NativeOIDCError, match="did not return a JSON object"):
            parse_token_response(["at"])

    def test_id_token_is_never_persisted(self):
        parsed = parse_token_response(
            {
                "access_token": "at",
                "token_type": "Bearer",
                "id_token": "must-not-appear",
            }
        )
        assert "must-not-appear" not in repr(parsed)


class TestComputeExpiresAt:
    def test_prefers_expires_in(self):
        assert compute_expires_at("opaque", 120, now=1000.0) == 1120.0

    def test_falls_back_to_a_short_default(self):
        assert compute_expires_at("opaque", None, now=1000.0) == 1000.0 + FALLBACK_LIFETIME_SECONDS

    def test_uses_the_untrusted_jwt_exp_when_expires_in_is_absent(self):
        token = make_jwt({"exp": 1500})
        assert compute_expires_at(token, None, now=1000.0) == 1500.0

    def test_takes_the_earlier_of_the_two(self):
        token = make_jwt({"exp": 1200})
        assert compute_expires_at(token, 600, now=1000.0) == 1200.0
        token = make_jwt({"exp": 5000})
        assert compute_expires_at(token, 600, now=1000.0) == 1600.0

    def test_already_expired_jwt_exp_is_ignored(self):
        token = make_jwt({"exp": 500})
        assert compute_expires_at(token, 600, now=1000.0) == 1600.0

    @pytest.mark.parametrize(
        "token",
        [
            "opaque",
            "a.b",
            "a.b.c.d",
            "a.!!!.c",
            make_jwt({"exp": "soon"}),
            make_jwt({"exp": True}),
            make_jwt([1, 2]),
        ],
    )
    def test_unparsable_tokens_do_not_raise(self, token):
        assert compute_expires_at(token, None, now=1000.0) == 1000.0 + FALLBACK_LIFETIME_SECONDS


class TestTokenErrors:
    def test_extract_oauth_error(self):
        assert extract_oauth_error({"error": "invalid_grant"}) == "invalid_grant"
        assert extract_oauth_error({"error": ""}) is None
        assert extract_oauth_error({"error": 7}) is None
        assert extract_oauth_error({}) is None
        assert extract_oauth_error(None) is None

    def test_describe_token_error_surfaces_only_the_code(self):
        message = describe_token_error(
            400,
            {"error": "invalid_grant", "error_description": "code SECRET already used"},
        )
        assert "invalid_grant" in message
        assert "SECRET" not in message

    def test_describe_token_error_without_a_body(self):
        assert describe_token_error(503, None) == "token endpoint returned HTTP 503"


# --------------------------------------------------------------------------- #
# Device flow
# --------------------------------------------------------------------------- #


def device_payload(**overrides):
    payload = {
        "device_code": "dc",
        "user_code": "WXYZ-1234",
        "verification_uri": f"{ISSUER}/activate",
        "expires_in": 600,
        "interval": 5,
    }
    payload.update(overrides)
    return payload


class TestParseDeviceAuthorization:
    def test_valid(self):
        parsed = parse_device_authorization(device_payload(verification_uri_complete=f"{ISSUER}/activate?code=WXYZ"))
        assert parsed.device_code == "dc"
        assert parsed.user_code == "WXYZ-1234"
        assert parsed.verification_uri_complete == f"{ISSUER}/activate?code=WXYZ"

    def test_interval_defaults_to_five_seconds(self):
        payload = device_payload()
        del payload["interval"]
        assert parse_device_authorization(payload).interval == DEFAULT_POLL_INTERVAL_SECONDS

    @pytest.mark.parametrize(
        "overrides,message",
        [
            ({"device_code": None}, "missing device_code"),
            ({"user_code": ""}, "missing user_code"),
            ({"verification_uri": None}, "missing verification_uri"),
            ({"verification_uri": "http://idp.example.com/activate"}, "must use HTTPS"),
            ({"verification_uri": 7}, "is not a string"),
            ({"expires_in": None}, "missing expires_in"),
            ({"expires_in": 0}, "expires_in is out of range"),
            ({"expires_in": 10**6}, "expires_in is out of range"),
            ({"expires_in": "600"}, "expires_in is not an integer"),
            ({"expires_in": True}, "expires_in is not an integer"),
            ({"interval": 0}, "interval is out of range"),
            ({"interval": MAX_POLL_INTERVAL_SECONDS + 1}, "interval is out of range"),
        ],
    )
    def test_invalid(self, overrides, message):
        payload = device_payload()
        for key, value in overrides.items():
            if value is None:
                payload.pop(key, None)
            else:
                payload[key] = value
        with pytest.raises(NativeOIDCError, match=message):
            parse_device_authorization(payload)

    def test_non_object(self):
        with pytest.raises(NativeOIDCError, match="did not return a JSON object"):
            parse_device_authorization("nope")


def stub_post_form(monkeypatch, module, responses):
    """Replace `post_form` with a scripted sequence of responses/exceptions."""
    calls = []
    queue = list(responses)

    def fake_post_form(url, data, **kwargs):
        calls.append({"url": url, "data": data})
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(module, "post_form", fake_post_form)
    return calls


def token_success(**overrides):
    payload = {"access_token": "at", "token_type": "Bearer", "expires_in": 300}
    payload.update(overrides)
    return JsonResponse(status_code=200, payload=payload, retry_after=None)


def oauth_error(error, status_code=400, retry_after=None):
    return JsonResponse(status_code=status_code, payload={"error": error}, retry_after=retry_after)


class TestRequestDeviceAuthorization:
    def test_sends_public_client_parameters(self, monkeypatch):
        calls = stub_post_form(
            monkeypatch,
            device_flow,
            [JsonResponse(status_code=200, payload=device_payload(), retry_after=None)],
        )
        request_device_authorization(f"{ISSUER}/device", METADATA)
        assert calls[0]["data"] == {"client_id": CLIENT_ID, "scope": "openid profile"}
        assert "client_secret" not in calls[0]["data"]

    def test_error_response_is_reported_against_the_right_endpoint(self, monkeypatch):
        stub_post_form(monkeypatch, device_flow, [oauth_error("invalid_client")])
        with pytest.raises(NativeOIDCError, match="device authorization endpoint returned OAuth"):
            request_device_authorization(f"{ISSUER}/device", METADATA)


class TestPollForDeviceToken:
    def make_authorization(self, **overrides):
        fields = {
            "device_code": "dc",
            "user_code": "WXYZ",
            "verification_uri": f"{ISSUER}/activate",
            "verification_uri_complete": None,
            "expires_in": 600,
            "interval": 5,
        }
        fields.update(overrides)
        return DeviceAuthorization(**fields)

    def poll(self, monkeypatch, responses, **overrides):
        sleeps = []
        clock = {"now": 0.0}

        def sleep(seconds):
            sleeps.append(seconds)
            clock["now"] += seconds

        calls = stub_post_form(monkeypatch, device_flow, responses)
        token = poll_for_device_token(
            f"{ISSUER}/token",
            self.make_authorization(**overrides),
            METADATA,
            sleep=sleep,
            monotonic=lambda: clock["now"],
        )
        return token, sleeps, calls

    def test_success_after_pending(self, monkeypatch):
        token, sleeps, calls = self.poll(monkeypatch, [oauth_error("authorization_pending"), token_success()])
        assert token.access_token == "at"
        assert sleeps == [5, 5]
        assert calls[0]["data"] == {
            "grant_type": DEVICE_GRANT,
            "device_code": "dc",
            "client_id": CLIENT_ID,
        }

    def test_slow_down_raises_the_interval_for_every_later_poll(self, monkeypatch):
        _, sleeps, _ = self.poll(
            monkeypatch,
            [
                oauth_error("slow_down"),
                oauth_error("authorization_pending"),
                token_success(),
            ],
        )
        assert sleeps == [5, 10, 10]

    def test_interval_is_capped(self, monkeypatch):
        _, sleeps, _ = self.poll(
            monkeypatch,
            [oauth_error("slow_down"), oauth_error("slow_down"), token_success()],
            interval=MAX_POLL_INTERVAL_SECONDS,
        )
        assert sleeps == [MAX_POLL_INTERVAL_SECONDS] * 3

    def test_retry_after_raises_the_interval(self, monkeypatch):
        _, sleeps, _ = self.poll(
            monkeypatch,
            [oauth_error("authorization_pending", retry_after=20), token_success()],
        )
        assert sleeps == [5, 20]

    def test_retry_after_never_lowers_the_interval(self, monkeypatch):
        _, sleeps, _ = self.poll(
            monkeypatch,
            [oauth_error("authorization_pending", retry_after=1), token_success()],
            interval=10,
        )
        assert sleeps == [10, 10]

    def test_connection_errors_back_off_instead_of_busy_looping(self, monkeypatch):
        _, sleeps, _ = self.poll(monkeypatch, [NativeOIDCError("could not reach"), token_success()])
        assert sleeps == [5, 10]

    def test_access_denied_stops_polling(self, monkeypatch):
        with pytest.raises(NativeOIDCError, match="was denied"):
            self.poll(monkeypatch, [oauth_error("access_denied")])

    def test_expired_token_stops_polling(self, monkeypatch):
        with pytest.raises(NativeOIDCError, match="expired before it was approved"):
            self.poll(monkeypatch, [oauth_error("expired_token")])

    def test_unknown_oauth_error_stops_polling(self, monkeypatch):
        with pytest.raises(NativeOIDCError, match="invalid_client"):
            self.poll(monkeypatch, [oauth_error("invalid_client")])

    def test_deadline_ends_the_loop(self, monkeypatch):
        with pytest.raises(NativeOIDCError, match="expired before it was approved"):
            self.poll(
                monkeypatch,
                [oauth_error("authorization_pending")] * 3,
                expires_in=10,
                interval=5,
            )


class TestRunDeviceFlow:
    def test_shows_the_user_code_but_never_the_device_code(self, monkeypatch):
        stub_post_form(
            monkeypatch,
            device_flow,
            [
                JsonResponse(
                    status_code=200,
                    payload=device_payload(device_code="device-code-secret"),
                    retry_after=None,
                ),
                token_success(),
            ],
        )
        lines = []
        token = run_device_flow(
            METADATA,
            PROVIDER,
            open_browser=False,
            echo=lines.append,
            sleep=lambda _: None,
            monotonic=lambda: 0.0,
        )
        assert token.access_token == "at"
        output = "\n".join(lines)
        assert "WXYZ-1234" in output
        assert f"{ISSUER}/activate" in output
        assert "device-code-secret" not in output

    def test_unsupported_provider_is_rejected_before_any_request(self, monkeypatch):
        stub_post_form(monkeypatch, device_flow, [])
        provider = ProviderMetadata(**{**PROVIDER.__dict__, "device_authorization_endpoint": None})
        with pytest.raises(NativeOIDCError, match="device_authorization_endpoint"):
            run_device_flow(METADATA, provider, open_browser=False, echo=lambda _: None)

    def run(self, monkeypatch, payload, **kwargs):
        stub_post_form(
            monkeypatch,
            device_flow,
            [
                JsonResponse(status_code=200, payload=payload, retry_after=None),
                token_success(),
            ],
        )
        return run_device_flow(
            METADATA,
            PROVIDER,
            sleep=lambda _: None,
            monotonic=lambda: 0.0,
            **kwargs,
        )

    def test_the_complete_verification_uri_is_offered_and_opened(self, monkeypatch):
        complete = f"{ISSUER}/activate?user_code=WXYZ-1234"
        opened = []
        monkeypatch.setattr(device_flow.webbrowser, "open", opened.append)
        lines = []
        self.run(monkeypatch, device_payload(verification_uri_complete=complete), echo=lines.append)
        assert complete in "\n".join(lines)
        assert opened == [complete]

    def test_the_plain_verification_uri_is_opened_when_there_is_no_complete_one(self, monkeypatch):
        opened = []
        monkeypatch.setattr(device_flow.webbrowser, "open", opened.append)
        self.run(monkeypatch, device_payload(), echo=lambda _: None)
        assert opened == [f"{ISSUER}/activate"]

    def test_browser_launch_failure_is_not_fatal(self, monkeypatch):
        def boom(target):
            raise RuntimeError("no display")

        monkeypatch.setattr(device_flow.webbrowser, "open", boom)
        # The code is already on screen, so a failed launch must not abort.
        assert self.run(monkeypatch, device_payload(), echo=lambda _: None).access_token == "at"


# --------------------------------------------------------------------------- #
# Browser flow
# --------------------------------------------------------------------------- #


class TestBuildAuthorizationUrl:
    def build(self, endpoint=f"{ISSUER}/authorize"):
        challenge = generate_pkce_challenge()
        url = build_authorization_url(
            endpoint,
            client_id=CLIENT_ID,
            redirect_uri="http://127.0.0.1:5555/oauth/callback",
            scopes=("openid", "profile"),
            challenge=challenge,
        )
        return url, challenge

    def test_contains_the_required_parameters(self):
        url, challenge = self.build()
        params = parse_qs(urlsplit(url).query)
        assert params["response_type"] == ["code"]
        assert params["client_id"] == [CLIENT_ID]
        assert params["redirect_uri"] == ["http://127.0.0.1:5555/oauth/callback"]
        assert params["scope"] == ["openid profile"]
        assert params["state"] == [challenge.state]
        assert params["code_challenge"] == [challenge.code_challenge]
        assert params["code_challenge_method"] == ["S256"]

    def test_never_carries_the_verifier(self):
        url, challenge = self.build()
        assert challenge.code_verifier not in url

    def test_preserves_an_existing_query_on_the_endpoint(self):
        url, _ = self.build(endpoint=f"{ISSUER}/authorize?tenant=acme")
        params = parse_qs(urlsplit(url).query)
        assert params["tenant"] == ["acme"]
        assert params["response_type"] == ["code"]

    def test_no_fragment_is_emitted(self):
        url, _ = self.build()
        assert urlsplit(url).fragment == ""


class TestExchangeCodeForToken:
    def test_sends_the_verifier_as_a_public_client(self, monkeypatch):
        calls = stub_post_form(monkeypatch, browser_flow, [token_success()])
        token = exchange_code_for_token(
            f"{ISSUER}/token",
            code="the-code",
            redirect_uri="http://127.0.0.1:5555/oauth/callback",
            client_id=CLIENT_ID,
            code_verifier="the-verifier",
        )
        assert token.access_token == "at"
        assert calls[0]["data"] == {
            "grant_type": "authorization_code",
            "code": "the-code",
            "redirect_uri": "http://127.0.0.1:5555/oauth/callback",
            "client_id": CLIENT_ID,
            "code_verifier": "the-verifier",
        }
        assert "client_secret" not in calls[0]["data"]

    def test_error_response_raises_without_echoing_the_body(self, monkeypatch):
        stub_post_form(
            monkeypatch,
            browser_flow,
            [
                JsonResponse(
                    status_code=400,
                    payload={
                        "error": "invalid_grant",
                        "error_description": "code SECRET",
                    },
                    retry_after=None,
                )
            ],
        )
        with pytest.raises(NativeOIDCError) as excinfo:
            exchange_code_for_token(
                f"{ISSUER}/token",
                code="c",
                redirect_uri="http://127.0.0.1:5555/oauth/callback",
                client_id=CLIENT_ID,
                code_verifier="v",
            )
        assert "invalid_grant" in str(excinfo.value)
        assert "SECRET" not in str(excinfo.value)

    def test_non_json_success_is_rejected(self, monkeypatch):
        stub_post_form(
            monkeypatch,
            browser_flow,
            [JsonResponse(status_code=200, payload=None, retry_after=None)],
        )
        with pytest.raises(NativeOIDCError, match="HTTP 200"):
            exchange_code_for_token(
                f"{ISSUER}/token",
                code="c",
                redirect_uri="http://127.0.0.1:5555/oauth/callback",
                client_id=CLIENT_ID,
                code_verifier="v",
            )


class TestRunBrowserFlow:
    def test_end_to_end_over_a_real_loopback_redirect(self, monkeypatch):
        calls = stub_post_form(monkeypatch, browser_flow, [token_success()])
        lines = []
        url_ready = threading.Event()
        captured = {}

        def echo(line):
            lines.append(line)
            if "code_challenge" in line:
                captured["url"] = line.strip()
                url_ready.set()

        def redirect_back():
            assert url_ready.wait(5)
            query = parse_qs(urlsplit(captured["url"]).query)
            redirect = urlsplit(query["redirect_uri"][0])
            connection = HTTPConnection(redirect.hostname, redirect.port, timeout=5)
            connection.request("GET", f"{redirect.path}?code=auth-code&state={query['state'][0]}")
            captured["status"] = connection.getresponse().status
            connection.close()

        thread = threading.Thread(target=redirect_back)
        thread.start()
        token = run_browser_flow(METADATA, PROVIDER, open_browser=False, timeout=10, echo=echo)
        thread.join(5)

        assert captured["status"] == 200
        assert token.access_token == "at"
        assert calls[0]["data"]["code"] == "auth-code"
        # The verifier matches the challenge that was sent to the authorization endpoint.
        sent_challenge = parse_qs(urlsplit(captured["url"]).query)["code_challenge"][0]
        assert compute_code_challenge(calls[0]["data"]["code_verifier"]) == sent_challenge

    def test_unsupported_provider_is_rejected_before_binding_a_port(self, monkeypatch):
        stub_post_form(monkeypatch, browser_flow, [])
        provider = ProviderMetadata(**{**PROVIDER.__dict__, "code_challenge_methods_supported": ("plain",)})
        with pytest.raises(NativeOIDCError, match="PKCE S256"):
            run_browser_flow(METADATA, provider, open_browser=False, echo=lambda _: None)

    def test_browser_launch_failure_is_not_fatal(self, monkeypatch):
        monkeypatch.setattr(
            browser_flow.webbrowser,
            "open",
            lambda url: (_ for _ in ()).throw(RuntimeError("no display")),
        )
        lines = []
        browser_flow._try_open_browser("https://idp.example.com", echo=lines.append)
        assert any("manually" in line for line in lines)


def test_login_timeout_default_is_bounded():
    assert 0 < browser_flow.DEFAULT_LOGIN_TIMEOUT_SECONDS <= 600
    assert isinstance(time.monotonic(), float)
