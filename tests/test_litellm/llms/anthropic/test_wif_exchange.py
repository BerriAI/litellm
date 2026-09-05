import asyncio
import concurrent.futures
import json
import logging
import time
from collections.abc import Callable, Mapping
from typing import Final

import httpx2
import pytest

from litellm.llms.anthropic.wif import AnthropicWifParams
from litellm.llms.anthropic.wif_exchange import (
    _DETAIL_CAP,
    _METRICS_QUEUE_LIMIT,
    CALL_TYPE_CACHE_HIT,
    EXCHANGE_CONNECT_TIMEOUT_SECONDS,
    EXCHANGE_TIMEOUT_SECONDS,
    MAX_ASSERTION_BYTES,
    AnthropicWifTokenExchange,
    ServiceLoggingMetricsSink,
    TokenExchangeEndpointFailure,
    TokenExchangeTransportFailure,
    _default_assertion_reader,
    _error_summary,
    new_exchange_client,
)
from litellm.llms.base_llm.auth.oauth_endpoint import MAX_RESPONSE_BYTES
from litellm.llms.base_llm.auth.types import (
    AssertionSourceError,
    ExchangeError,
    InsecureTokenUrl,
    MalformedTokenResponse,
    TokenEndpointError,
    TokenTransportError,
)
from litellm.secret_managers.main import OidcPathNotAllowedError, _resolve_oidc_file_path
from litellm.types.services import ServiceTypes

DEFAULT_REF: Final = "oidc/env/TEST_ASSERTION"
DEFAULT_ASSERTION: Final = "test-jwt-assertion"
EXCHANGE_BASE: Final = "https://token.example"
EXCHANGE_URL: Final = "https://token.example/v1/oauth/token"
GRANT_TYPE: Final = "urn:ietf:params:oauth:grant-type:jwt-bearer"
SDK_BETA_HEADER: Final = "oauth-2025-04-20,oidc-federation-2026-04-01"
TOKEN_TTL: Final = 3600


class FakeClock:
    """Starts at the real time: the SDK stamps ``expires_at`` from ``time.time()`` and only its
    cache's refresh decisions read the injected clock."""

    def __init__(self) -> None:
        self.now = time.time()

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordedRequest:
    def __init__(self, request: httpx2.Request) -> None:
        self.url = str(request.url)
        self.content = request.content
        self.headers = dict(request.headers)

    def json_body(self) -> dict:
        return json.loads(self.content)


class ScriptedTokenEndpoint:
    """A token endpoint behind ``httpx2.MockTransport``: answers with the scripted responses in
    order (repeating the last) and records every request the SDK put on the wire."""

    def __init__(
        self,
        responses: list[httpx2.Response],
        on_request: Callable[[RecordedRequest], None] | None = None,
    ) -> None:
        self.requests: list[RecordedRequest] = []
        self._responses = list(responses)
        self._on_request = on_request

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        recorded = RecordedRequest(request)
        self.requests.append(recorded)
        if self._on_request is not None:
            self._on_request(recorded)
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


class RaisingTokenEndpoint:
    def __init__(self, error: Exception) -> None:
        self.calls = 0
        self._error = error

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.calls += 1
        raise self._error


class EchoingUnauthorizedEndpoint:
    """401s every attempt, echoing the submitted assertion back into the error body: a token
    endpoint that reflects the request."""

    def __init__(self) -> None:
        self.requests: list[RecordedRequest] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        recorded = RecordedRequest(request)
        self.requests.append(recorded)
        submitted = recorded.json_body()["assertion"]
        return httpx2.Response(401, json={"error": "invalid_grant", "error_description": f"bad assertion {submitted}"})


class RotatingAssertionSource:
    """A per-call assertion source that mints a fresh value on every read, the shape the
    internal_issuer and keycloak identity sources take."""

    def __init__(self, values: list[str]) -> None:
        self._values = iter(values)
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        return next(self._values)


class RecordingMetricsSink:
    def __init__(self) -> None:
        self.successes: list[tuple[str, float]] = []
        self.failures: list[tuple[str, float, ExchangeError]] = []
        self.cache_hits = 0

    def exchange_success(self, *, call_type: str, duration_seconds: float) -> None:
        self.successes.append((call_type, duration_seconds))

    def exchange_failure(self, *, call_type: str, duration_seconds: float, error: ExchangeError) -> None:
        self.failures.append((call_type, duration_seconds, error))

    def cache_hit(self) -> None:
        self.cache_hits += 1


class RaisingMetricsSink:
    def exchange_success(self, *, call_type: str, duration_seconds: float) -> None:
        raise RuntimeError("metrics sink down")

    def exchange_failure(self, *, call_type: str, duration_seconds: float, error: ExchangeError) -> None:
        raise RuntimeError("metrics sink down")

    def cache_hit(self) -> None:
        raise RuntimeError("metrics sink down")


def token_response(token: str = "sk-ant-oat01-minted", expires_in: int | None = TOKEN_TTL) -> httpx2.Response:
    body: Final[dict[str, str | int]] = {
        "access_token": token,
        "token_type": "Bearer",
        **({} if expires_in is None else {"expires_in": expires_in}),
    }
    return httpx2.Response(200, json=body)


def make_params(
    *,
    federation_rule_id: str = "fdrl_1",
    organization_id: str = "org-1",
    service_account_id: str | None = None,
    workspace_id: str | None = None,
    assertion_ref: str = DEFAULT_REF,
    assertion_source: Callable[[], str | None] | None = None,
) -> AnthropicWifParams:
    return AnthropicWifParams(
        federation_rule_id=federation_rule_id,
        organization_id=organization_id,
        service_account_id=service_account_id,
        workspace_id=workspace_id,
        assertion_ref=assertion_ref,
        assertion_source=assertion_source,
    )


def make_exchange(
    endpoint: Callable[[httpx2.Request], httpx2.Response],
    reader: Mapping[str, str] | Callable[[str], str | None] | None = None,
    clock: FakeClock | None = None,
    max_entries: int = 64,
    metrics_sink=None,
) -> AnthropicWifTokenExchange:
    resolved_reader = reader if callable(reader) else (reader or {DEFAULT_REF: DEFAULT_ASSERTION}).get
    return AnthropicWifTokenExchange(
        http_client=httpx2.Client(transport=httpx2.MockTransport(endpoint)),
        assertion_reader=resolved_reader,
        max_entries=max_entries,
        metrics_sink=metrics_sink if metrics_sink is not None else RecordingMetricsSink(),
        clock=clock if clock is not None else time.time,
    )


def mint(
    exchange: AnthropicWifTokenExchange, params: AnthropicWifParams | None = None, exchange_base: str = EXCHANGE_BASE
) -> str:
    result = exchange.get_token(params if params is not None else make_params(), exchange_base)
    assert isinstance(result, str), result
    return result


class TestFreshMintWireExact:
    def test_url_headers_and_full_body(self):
        endpoint = ScriptedTokenEndpoint([token_response()])
        exchange = make_exchange(endpoint)

        token = mint(exchange, make_params(service_account_id="svcacct_1", workspace_id="wrkspc_1"))

        assert token == "sk-ant-oat01-minted"
        (request,) = endpoint.requests
        assert request.url == EXCHANGE_URL
        assert request.headers["anthropic-beta"] == SDK_BETA_HEADER
        assert request.headers["content-type"] == "application/json"
        assert request.headers["user-agent"].startswith("anthropic-python/")
        assert request.json_body() == {
            "grant_type": GRANT_TYPE,
            "assertion": DEFAULT_ASSERTION,
            "federation_rule_id": "fdrl_1",
            "organization_id": "org-1",
            "service_account_id": "svcacct_1",
            "workspace_id": "wrkspc_1",
        }

    def test_unset_optional_ids_are_absent_from_the_body(self):
        endpoint = ScriptedTokenEndpoint([token_response()])

        mint(make_exchange(endpoint))

        assert set(endpoint.requests[0].json_body()) == {
            "grant_type",
            "assertion",
            "federation_rule_id",
            "organization_id",
        }

    def test_a_second_call_is_served_from_cache_without_a_post(self):
        endpoint = ScriptedTokenEndpoint([token_response()])
        exchange = make_exchange(endpoint)

        first = mint(exchange)
        second = mint(exchange)

        assert first == second == "sk-ant-oat01-minted"
        assert len(endpoint.requests) == 1


class TestUnauthorizedRetry:
    def test_a_401_is_retried_once_with_a_freshly_read_assertion(self):
        assertions = {DEFAULT_REF: "assertion-v1"}
        endpoint = ScriptedTokenEndpoint([httpx2.Response(401, json={"error": "invalid_grant"}), token_response()])
        exchange = make_exchange(endpoint, reader=assertions.get)

        def rotate(_request: RecordedRequest) -> None:
            assertions[DEFAULT_REF] = "assertion-v2"

        endpoint._on_request = rotate

        assert mint(exchange) == "sk-ant-oat01-minted"
        assert [request.json_body()["assertion"] for request in endpoint.requests] == ["assertion-v1", "assertion-v2"]

    def test_401_twice_is_endpoint_error(self):
        endpoint = ScriptedTokenEndpoint([httpx2.Response(401, json={"error": "invalid_grant"})])

        result = make_exchange(endpoint).get_token(make_params(), EXCHANGE_BASE)

        assert isinstance(result, TokenEndpointError)
        assert result.status_code == 401
        assert "invalid_grant" in result.redacted_body
        assert len(endpoint.requests) == 2

    def test_401_retry_redacts_the_assertion_actually_sent_not_a_fresh_reread(self):
        """Regression: with a rotating identity source, the reflection-drop check must match the
        assertion the failing (second) attempt actually sent. Re-reading for the check would mint a
        THIRD value that was never sent, so the reflection probe would miss and the actually-sent,
        actually-reflected second assertion would leak into the error."""
        endpoint = EchoingUnauthorizedEndpoint()
        source = RotatingAssertionSource(["assertion-v1", "assertion-v2", "assertion-v3"])

        result = make_exchange(endpoint).get_token(make_params(assertion_source=source), EXCHANGE_BASE)

        assert isinstance(result, TokenEndpointError)
        assert [request.json_body()["assertion"] for request in endpoint.requests] == ["assertion-v1", "assertion-v2"]
        assert source.calls == 2, "the failing attempt's own assertion must be reused, never re-read a third time"
        assert "assertion-v1" not in result.redacted_body
        assert "assertion-v2" not in result.redacted_body
        assert "assertion-v3" not in result.redacted_body


class TestSdkOutcomeMapping:
    """The SDK folds every failure into one exception class; each shape must land on the typed
    error ``wif.py`` maps onto the public exception contract, with nothing echoed through."""

    def test_error_object_body_is_reduced_to_rfc6749_fields_and_capped(self):
        endpoint = ScriptedTokenEndpoint(
            [
                httpx2.Response(
                    400,
                    json={
                        "error": "invalid_grant",
                        "error_description": "d" * 500,
                        "error_uri": "https://errors.example/e1",
                        "assertion_echo": "LEAKED-ASSERTION",
                    },
                )
            ]
        )

        result = make_exchange(endpoint).get_token(make_params(), EXCHANGE_BASE)

        assert isinstance(result, TokenEndpointError)
        assert result.status_code == 400
        assert "invalid_grant" in result.redacted_body
        assert "d" * 256 in result.redacted_body
        assert "d" * 257 not in result.redacted_body
        assert "https://errors.example/e1" in result.redacted_body
        assert "LEAKED-ASSERTION" not in result.redacted_body

    def test_reflected_assertion_in_an_error_body_is_dropped(self):
        result = make_exchange(EchoingUnauthorizedEndpoint()).get_token(make_params(), EXCHANGE_BASE)

        assert isinstance(result, TokenEndpointError)
        assert DEFAULT_ASSERTION not in result.redacted_body

    def test_plain_text_error_body_is_not_echoed(self):
        endpoint = ScriptedTokenEndpoint([httpx2.Response(502, text="t" * 500)])

        result = make_exchange(endpoint).get_token(make_params(), EXCHANGE_BASE)

        assert isinstance(result, TokenEndpointError)
        assert result.status_code == 502
        assert result.redacted_body == "non-JSON error response omitted"

    def test_oversized_error_body_is_never_parsed(self):
        endpoint = ScriptedTokenEndpoint(
            [httpx2.Response(400, content=b'{"error": "' + b"x" * MAX_RESPONSE_BYTES + b'"}')]
        )

        result = make_exchange(endpoint).get_token(make_params(), EXCHANGE_BASE)

        assert isinstance(result, TokenEndpointError)
        assert result.status_code == 400
        assert "x" * 10 not in result.redacted_body

    def test_oversized_success_body_is_malformed(self):
        endpoint = ScriptedTokenEndpoint(
            [httpx2.Response(200, content=b'{"access_token": "' + b"x" * MAX_RESPONSE_BYTES + b'"}')]
        )

        result = make_exchange(endpoint).get_token(make_params(), EXCHANGE_BASE)

        assert isinstance(result, MalformedTokenResponse)
        assert "x" * 10 not in result.detail

    def test_non_json_success_body_that_echoes_the_assertion_is_scrubbed(self):
        """The SDK quotes up to 256 characters of an unparseable 2xx body in its message, so a
        reflected assertion has to be dropped before that message becomes an error detail."""
        endpoint = ScriptedTokenEndpoint([httpx2.Response(200, text=f"<html>{DEFAULT_ASSERTION}</html>")])

        result = make_exchange(endpoint).get_token(make_params(), EXCHANGE_BASE)

        assert isinstance(result, MalformedTokenResponse)
        assert DEFAULT_ASSERTION not in result.detail

    def test_non_json_success_body_without_an_echo_keeps_the_diagnosis(self):
        endpoint = ScriptedTokenEndpoint([httpx2.Response(200, text="<html>upstream proxy page</html>")])

        result = make_exchange(endpoint).get_token(make_params(), EXCHANGE_BASE)

        assert isinstance(result, MalformedTokenResponse)
        assert "non-JSON" in result.detail

    def test_json_array_success_body_is_malformed(self):
        endpoint = ScriptedTokenEndpoint([httpx2.Response(200, json=["a", "b"])])

        result = make_exchange(endpoint).get_token(make_params(), EXCHANGE_BASE)

        assert isinstance(result, MalformedTokenResponse)

    def test_a_non_bearer_token_type_is_refused(self):
        endpoint = ScriptedTokenEndpoint(
            [httpx2.Response(200, json={"access_token": "tok", "token_type": "mac", "expires_in": 300})]
        )

        result = make_exchange(endpoint).get_token(make_params(), EXCHANGE_BASE)

        assert isinstance(result, MalformedTokenResponse)
        assert "mac" in result.detail

    def test_missing_expires_in_is_malformed(self):
        endpoint = ScriptedTokenEndpoint([token_response(expires_in=None)])

        result = make_exchange(endpoint).get_token(make_params(), EXCHANGE_BASE)

        assert isinstance(result, MalformedTokenResponse)
        assert "expires_in" in result.detail

    @pytest.mark.parametrize("access_token", ["", "   "])
    def test_empty_access_token_is_malformed_and_not_cached(self, access_token: str):
        endpoint = ScriptedTokenEndpoint(
            [
                httpx2.Response(200, json={"access_token": access_token, "token_type": "Bearer", "expires_in": 3600}),
                token_response("reminted"),
            ]
        )
        exchange = make_exchange(endpoint)

        first = exchange.get_token(make_params(), EXCHANGE_BASE)
        second = exchange.get_token(make_params(), EXCHANGE_BASE)

        assert isinstance(first, MalformedTokenResponse)
        assert "empty access_token" in first.detail
        assert second == "reminted"
        assert len(endpoint.requests) == 2

    @pytest.mark.parametrize("expires_in", [0, -5])
    def test_a_token_expired_on_arrival_is_never_cached(self, expires_in: int):
        endpoint = ScriptedTokenEndpoint(
            [token_response("short-lived", expires_in=expires_in), token_response("reminted")]
        )
        exchange = make_exchange(endpoint)

        assert mint(exchange) == "short-lived"
        assert mint(exchange) == "reminted"
        assert len(endpoint.requests) == 2

    def test_transport_failure_names_the_endpoint(self):
        endpoint = RaisingTokenEndpoint(httpx2.ConnectError("connection refused"))

        result = make_exchange(endpoint).get_token(make_params(), EXCHANGE_BASE)

        assert isinstance(result, TokenTransportError)
        assert EXCHANGE_URL in result.detail
        assert "connection refused" in result.detail
        assert endpoint.calls == 1

    def test_a_redirect_is_an_error_not_a_second_post(self):
        """Only the bound base URL passed the host allowlist, so a 3xx must never carry the
        assertion to the location it names."""
        endpoint = ScriptedTokenEndpoint(
            [httpx2.Response(302, headers={"location": "https://elsewhere.example/v1/oauth/token"})]
        )

        result = make_exchange(endpoint).get_token(make_params(), EXCHANGE_BASE)

        assert not isinstance(result, str)
        assert len(endpoint.requests) == 1


def test_sentinel_leak_audit(caplog: pytest.LogCaptureFixture):
    jwt_sentinel = "JWT-SENTINEL-2c9f1e7ab4"
    token_sentinel = "sk-ant-oat01-TOKEN-SENTINEL-90d4c3aa17"
    ref = "oidc/env/SENTINEL_ASSERTION"
    params = make_params(assertion_ref=ref)

    def exchange_with(endpoint, reader: Mapping[str, str] | None = None, clock: FakeClock | None = None):
        return make_exchange(endpoint, reader=reader if reader is not None else {ref: jwt_sentinel}, clock=clock)

    with caplog.at_level(logging.DEBUG):
        clock = FakeClock()
        serving = exchange_with(
            ScriptedTokenEndpoint(
                [token_response(token_sentinel), httpx2.Response(500, json={"error": "server_error"})]
            ),
            clock=clock,
        )
        minted = serving.get_token(params, EXCHANGE_BASE)
        endpoint_error = exchange_with(ScriptedTokenEndpoint([httpx2.Response(400, json={"error": "invalid_grant"})]))
        endpoint_error_result = endpoint_error.get_token(params, EXCHANGE_BASE)
        transport_error = exchange_with(RaisingTokenEndpoint(httpx2.ConnectError("boom"))).get_token(
            params, EXCHANGE_BASE
        )
        malformed_error = exchange_with(ScriptedTokenEndpoint([httpx2.Response(200, json={"unexpected": "shape"})]))
        malformed_result = malformed_error.get_token(params, EXCHANGE_BASE)
        echoed_result = exchange_with(EchoingUnauthorizedEndpoint()).get_token(params, EXCHANGE_BASE)
        oversized_error = exchange_with(
            ScriptedTokenEndpoint([token_response()]), reader={ref: jwt_sentinel + "x" * MAX_ASSERTION_BYTES}
        ).get_token(params, EXCHANGE_BASE)
        insecure_error = exchange_with(ScriptedTokenEndpoint([token_response()])).get_token(
            params, "http://token.example"
        )
        clock.advance(TOKEN_TTL - 100)
        stale = serving.get_token(params, EXCHANGE_BASE)

    assert minted == token_sentinel
    assert stale == token_sentinel, "an advisory refresh failure keeps serving the cached token"
    assert isinstance(oversized_error, AssertionSourceError)
    assert oversized_error.kind == "oversized"
    audited_values = [
        str(endpoint_error_result),
        repr(endpoint_error_result),
        str(transport_error),
        repr(transport_error),
        str(malformed_result),
        repr(malformed_result),
        str(echoed_result),
        repr(echoed_result),
        str(oversized_error),
        repr(oversized_error),
        str(insecure_error),
        repr(insecure_error),
        caplog.text,
    ]
    for value in audited_values:
        assert jwt_sentinel not in value
        assert token_sentinel not in value


class TestAssertionGuards:
    @pytest.mark.parametrize(
        "assertion_value,expected_kind",
        [
            ("x" * (MAX_ASSERTION_BYTES + 1), "oversized"),
            ("   \n\t ", "empty"),
            (None, "missing"),
        ],
    )
    def test_bad_assertion_values(self, assertion_value: str | None, expected_kind: str):
        endpoint = ScriptedTokenEndpoint([token_response()])

        result = make_exchange(endpoint, reader=lambda ref: assertion_value).get_token(make_params(), EXCHANGE_BASE)

        assert isinstance(result, AssertionSourceError)
        assert result.kind == expected_kind
        assert result.source_ref == DEFAULT_REF
        assert len(endpoint.requests) == 0

    @pytest.mark.parametrize(
        "raised,expected_kind",
        [
            (OidcPathNotAllowedError("path outside allowed credential directories"), "disallowed_path"),
            (ValueError("Environment variable ANTHROPIC_IDENTITY_TOKEN not found"), "unreadable"),
            (ImportError("needs PyJWT and cryptography: pip install 'litellm[proxy]'"), "unreadable"),
            (OSError("permission denied"), "unreadable"),
        ],
    )
    def test_raising_reader(self, raised: Exception, expected_kind: str):
        endpoint = ScriptedTokenEndpoint([token_response()])

        def reader(ref: str) -> str | None:
            raise raised

        result = make_exchange(endpoint, reader=reader).get_token(make_params(), EXCHANGE_BASE)

        assert isinstance(result, AssertionSourceError)
        assert result.kind == expected_kind
        assert len(endpoint.requests) == 0

    @pytest.mark.parametrize(
        "raised,expected_detail",
        [
            (
                ValueError("Keycloak token endpoint returned invalid_client"),
                "Keycloak token endpoint returned invalid_client",
            ),
            (ValueError("x" * (_DETAIL_CAP + 100)), "x" * _DETAIL_CAP),
            (ImportError("the internal_issuer identity source needs PyJWT: pip install 'litellm[proxy]'"), None),
        ],
    )
    def test_operator_diagnosable_messages_are_captured_as_detail(self, raised: Exception, expected_detail: str | None):
        def reader(ref: str) -> str | None:
            raise raised

        result = make_exchange(ScriptedTokenEndpoint([token_response()]), reader=reader).get_token(
            make_params(), EXCHANGE_BASE
        )

        assert isinstance(result, AssertionSourceError)
        assert result.detail is not None
        assert result.detail == (expected_detail if expected_detail is not None else str(raised))

    @pytest.mark.parametrize(
        "raised",
        [OidcPathNotAllowedError("path outside allowed credential directories"), OSError("permission denied")],
    )
    def test_non_value_error_never_populates_detail(self, raised: Exception):
        """Only the ValueError and ImportError branches carry operator-diagnosable text; every other
        reader failure stays detail=None."""

        def reader(ref: str) -> str | None:
            raise raised

        result = make_exchange(ScriptedTokenEndpoint([token_response()]), reader=reader).get_token(
            make_params(), EXCHANGE_BASE
        )

        assert isinstance(result, AssertionSourceError)
        assert result.detail is None


class TestAssertionSourceOverridesReader:
    """``AnthropicWifParams.assertion_source`` is how a per-config identity source (internal_issuer,
    keycloak) plugs in; it must win over the exchange-level reader, and failures must still be
    reported against ``assertion_ref``."""

    def test_assertion_source_is_used_instead_of_the_reader(self):
        endpoint = ScriptedTokenEndpoint([token_response()])
        calls: list[str] = []

        def reader(ref: str) -> str | None:
            calls.append(ref)
            return "from-reader"

        token = mint(make_exchange(endpoint, reader=reader), make_params(assertion_source=lambda: "from-source"))

        assert token == "sk-ant-oat01-minted"
        assert endpoint.requests[0].json_body()["assertion"] == "from-source"
        assert calls == []

    def test_assertion_source_failure_is_reported_against_assertion_ref(self):
        endpoint = ScriptedTokenEndpoint([token_response()])

        def raising_source() -> str | None:
            raise ValueError("keycloak token endpoint returned invalid_client")

        result = make_exchange(endpoint, reader=lambda ref: "from-reader").get_token(
            make_params(assertion_source=raising_source, assertion_ref="oidc/keycloak/abc123"), EXCHANGE_BASE
        )

        assert isinstance(result, AssertionSourceError)
        assert result.source_ref == "oidc/keycloak/abc123"
        assert result.detail == "keycloak token endpoint returned invalid_client"
        assert len(endpoint.requests) == 0

    def test_assertion_source_is_re_invoked_on_401_retry(self):
        values = iter(["assertion-v1", "assertion-v2"])
        endpoint = ScriptedTokenEndpoint([httpx2.Response(401, json={"error": "invalid_grant"}), token_response()])

        token = mint(
            make_exchange(endpoint, reader=lambda ref: "from-reader"),
            make_params(assertion_source=lambda: next(values)),
        )

        assert token == "sk-ant-oat01-minted"
        assert [request.json_body()["assertion"] for request in endpoint.requests] == ["assertion-v1", "assertion-v2"]


class TestOidcFilePathAllowlistRaisesTypedError:
    """Assertion-source failures are classified by exception type (see TestAssertionGuards); that
    only works if the real oidc/file allowlist raises OidcPathNotAllowedError, not a bare ValueError."""

    def test_out_of_allowlist_absolute_path(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", raising=False)

        with pytest.raises(OidcPathNotAllowedError):
            _resolve_oidc_file_path("/etc/not-a-credential-dir/token")

    def test_relative_path(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", raising=False)

        with pytest.raises(OidcPathNotAllowedError):
            _resolve_oidc_file_path("relative/token/path")


class TestHttpsEnforcement:
    def test_plain_http_rejected_host_only_zero_posts(self):
        endpoint = ScriptedTokenEndpoint([token_response()])
        sink = RecordingMetricsSink()

        result = make_exchange(endpoint, metrics_sink=sink).get_token(make_params(), "http://token.example")

        assert result == InsecureTokenUrl(host="token.example")
        assert "/v1/oauth/token" not in str(result)
        assert len(endpoint.requests) == 0
        assert [(call_type, error) for call_type, _duration, error in sink.failures] == [("cold_mint", result)]

    @pytest.mark.parametrize("base", ["http://localhost:8080", "http://127.0.0.1", "http://[::1]"])
    def test_localhost_http_allowed(self, base: str):
        endpoint = ScriptedTokenEndpoint([token_response()])

        token = mint(make_exchange(endpoint), exchange_base=base)

        assert token == "sk-ant-oat01-minted"
        assert endpoint.requests[0].url == f"{base}/v1/oauth/token"


class TestCacheIdentity:
    def test_cache_key_semantics(self):
        endpoint = ScriptedTokenEndpoint([token_response()])
        assertions = {DEFAULT_REF: DEFAULT_ASSERTION, "oidc/env/OTHER": "other-assertion"}
        exchange = make_exchange(endpoint, reader=assertions.get)

        mint(exchange)
        mint(exchange, make_params(service_account_id="svc-2"))
        mint(exchange, make_params(workspace_id="wrkspc-2"))
        mint(exchange, make_params(federation_rule_id="fdrl_2"))
        mint(exchange, make_params(organization_id="org-2"))
        mint(exchange, exchange_base="https://other.example")
        mint(exchange, make_params(assertion_ref="oidc/env/OTHER"))
        assert len(endpoint.requests) == 7

        assertions[DEFAULT_REF] = "rotated-assertion"
        assert mint(exchange) == "sk-ant-oat01-minted"
        assert len(endpoint.requests) == 7, "a still-valid token is served without re-reading the assertion"

    def test_bounded_eviction_drops_the_oldest_deployment(self):
        endpoint = ScriptedTokenEndpoint([token_response()])
        exchange = make_exchange(endpoint, max_entries=4)

        def mint_org(index: int) -> None:
            mint(exchange, make_params(organization_id=f"org-{index}"))

        for index in range(5):
            mint_org(index)
        assert len(endpoint.requests) == 5

        mint_org(0)
        assert len(endpoint.requests) == 6, "the oldest entry (index 0) should have been evicted"

        mint_org(2)
        assert len(endpoint.requests) == 6, "a younger entry should still be cached"

        mint_org(1)
        assert len(endpoint.requests) == 7, "re-inserting index 0 should have evicted the next oldest entry"


async def test_aget_token_loop_responsive():
    def sleeping_endpoint(request: httpx2.Request) -> httpx2.Response:
        time.sleep(0.3)
        return token_response()

    exchange = make_exchange(sleeping_endpoint)
    ticks = {"count": 0}
    stop = asyncio.Event()

    async def ticker() -> None:
        while not stop.is_set():
            ticks["count"] += 1
            await asyncio.sleep(0.01)

    ticker_task = asyncio.create_task(ticker())
    result = await exchange.aget_token(make_params(), EXCHANGE_BASE)
    stop.set()
    await ticker_task

    assert result == "sk-ant-oat01-minted"
    assert ticks["count"] >= 5, "the event loop was blocked during aget_token"
    assert exchange.get_token(make_params(), EXCHANGE_BASE) == result


class TestRefreshWindows:
    def test_well_before_expiry_the_cache_serves_without_posting(self):
        clock = FakeClock()
        sink = RecordingMetricsSink()
        endpoint = ScriptedTokenEndpoint([token_response("old"), token_response("new")])
        exchange = make_exchange(endpoint, clock=clock, metrics_sink=sink)

        mint(exchange)
        clock.advance(TOKEN_TTL - 121)
        assert mint(exchange) == "old"

        assert len(endpoint.requests) == 1
        assert sink.cache_hits == 1

    def test_inside_the_advisory_window_the_token_is_refreshed_inline(self):
        clock = FakeClock()
        sink = RecordingMetricsSink()
        endpoint = ScriptedTokenEndpoint([token_response("old"), token_response("new")])
        exchange = make_exchange(endpoint, clock=clock, metrics_sink=sink)

        mint(exchange)
        clock.advance(TOKEN_TTL - 119)
        assert mint(exchange) == "new"

        assert len(endpoint.requests) == 2
        assert [call_type for call_type, _ in sink.successes] == ["cold_mint", "refresh"]
        assert sink.cache_hits == 0

    def test_an_advisory_refresh_failure_serves_the_cached_token_and_backs_off(self):
        clock = FakeClock()
        sink = RecordingMetricsSink()
        endpoint = ScriptedTokenEndpoint([token_response("old"), httpx2.Response(500, json={"error": "server_error"})])
        exchange = make_exchange(endpoint, clock=clock, metrics_sink=sink)

        mint(exchange)
        clock.advance(TOKEN_TTL - 119)
        assert mint(exchange) == "old"
        assert mint(exchange) == "old"

        assert len(endpoint.requests) == 2, "a refresh that just failed is not retried on the very next call"
        ((call_type, _duration, error),) = sink.failures
        assert call_type == "refresh"
        assert isinstance(error, TokenEndpointError)
        assert error.status_code == 500

    def test_inside_the_mandatory_window_a_failed_refresh_is_the_error(self):
        clock = FakeClock()
        sink = RecordingMetricsSink()
        endpoint = ScriptedTokenEndpoint([token_response("old"), httpx2.Response(503, json={"error": "unavailable"})])
        exchange = make_exchange(endpoint, clock=clock, metrics_sink=sink)

        mint(exchange)
        clock.advance(TOKEN_TTL - 29)
        result = exchange.get_token(make_params(), EXCHANGE_BASE)

        assert isinstance(result, TokenEndpointError)
        assert result.status_code == 503
        assert [call_type for call_type, _duration, _error in sink.failures] == ["refresh"]

    def test_a_refresh_reads_the_assertion_again(self):
        clock = FakeClock()
        assertions = {DEFAULT_REF: "first-assertion"}
        endpoint = ScriptedTokenEndpoint([token_response("old"), token_response("new")])
        exchange = make_exchange(endpoint, reader=assertions.get, clock=clock)

        mint(exchange)
        assertions[DEFAULT_REF] = "second-assertion"
        clock.advance(TOKEN_TTL - 29)
        assert mint(exchange) == "new"

        assert endpoint.requests[1].json_body()["assertion"] == "second-assertion"


class TestMetricsEmission:
    def test_cold_mint_emits_success_with_duration(self):
        sink = RecordingMetricsSink()
        endpoint = ScriptedTokenEndpoint([token_response()], on_request=lambda _request: time.sleep(0.05))

        mint(make_exchange(endpoint, metrics_sink=sink))

        ((call_type, duration),) = sink.successes
        assert call_type == "cold_mint"
        assert duration >= 0.05
        assert sink.failures == []
        assert sink.cache_hits == 0

    def test_cache_hit_emits_counter_not_a_mint(self):
        sink = RecordingMetricsSink()
        exchange = make_exchange(ScriptedTokenEndpoint([token_response()]), metrics_sink=sink)

        mint(exchange)
        mint(exchange)

        assert sink.cache_hits == 1
        assert len(sink.successes) == 1

    def test_every_failed_cold_mint_attempt_emits_a_failure(self):
        sink = RecordingMetricsSink()
        exchange = make_exchange(
            ScriptedTokenEndpoint([httpx2.Response(503, json={"error": "unavailable"})]), metrics_sink=sink
        )

        first = exchange.get_token(make_params(), EXCHANGE_BASE)
        second = exchange.get_token(make_params(), EXCHANGE_BASE)

        assert isinstance(first, TokenEndpointError)
        assert isinstance(second, TokenEndpointError)
        assert [call_type for call_type, _duration, _error in sink.failures] == ["cold_mint", "cold_mint"]
        assert all(isinstance(error, TokenEndpointError) and error.status_code == 503 for _, _, error in sink.failures)
        assert sink.successes == []

    def test_a_401_retry_reports_the_failed_attempt_and_the_cold_mint(self):
        sink = RecordingMetricsSink()
        endpoint = ScriptedTokenEndpoint([httpx2.Response(401, json={"error": "invalid_grant"}), token_response()])

        mint(make_exchange(endpoint, metrics_sink=sink))

        assert [call_type for call_type, _duration, _error in sink.failures] == ["cold_mint"]
        assert [call_type for call_type, _duration in sink.successes] == ["cold_mint"]

    def test_failure_payload_carries_no_assertion_material(self):
        sink = RecordingMetricsSink()

        result = make_exchange(EchoingUnauthorizedEndpoint(), metrics_sink=sink).get_token(make_params(), EXCHANGE_BASE)

        assert isinstance(result, TokenEndpointError)
        assert len(sink.failures) == 2
        for failure in sink.failures:
            assert DEFAULT_ASSERTION not in repr(failure)
            assert DEFAULT_ASSERTION not in _error_summary(failure[2])

    def test_raising_sink_never_breaks_mint_serve_or_failure(self):
        exchange = make_exchange(ScriptedTokenEndpoint([token_response()]), metrics_sink=RaisingMetricsSink())

        assert mint(exchange) == mint(exchange)

        failing = make_exchange(RaisingTokenEndpoint(httpx2.ConnectError("boom")), metrics_sink=RaisingMetricsSink())
        assert isinstance(failing.get_token(make_params(), EXCHANGE_BASE), TokenTransportError)


class TestDefaultAssertionReader:
    def test_reads_through_litellm_secret_resolution(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("WIF_ASSERTION_FOR_DEFAULT_READER", "header.payload.signature")

        assert _default_assertion_reader("os.environ/WIF_ASSERTION_FOR_DEFAULT_READER") == "header.payload.signature"

    def test_an_unset_reference_reads_as_none(self):
        assert _default_assertion_reader("os.environ/DEFINITELY_NOT_SET_WIF_ASSERTION_REF") is None


class TestErrorSummary:
    def test_every_error_variant_summarises_without_carrying_a_secret(self):
        summaries: Final = {
            _error_summary(AssertionSourceError(kind="unreadable", source_ref="oidc/file/x")),
            _error_summary(InsecureTokenUrl(host="token.internal")),
            _error_summary(TokenEndpointError(status_code=401, redacted_body="invalid_grant")),
            _error_summary(TokenTransportError(detail="ConnectError: refused")),
            _error_summary(MalformedTokenResponse(detail="empty access_token")),
        }

        assert {s.split(":")[0] for s in summaries} == {
            "AssertionSourceError",
            "InsecureTokenUrl",
            "TokenEndpointError",
            "TokenTransportError",
            "MalformedTokenResponse",
        }, "each variant names itself so a log line says which stage failed"


class TestExchangeClient:
    def test_redirects_are_not_followed_and_timeouts_are_bounded(self):
        """Only the initial token URL is validated, so a 3xx must not be allowed to replay the
        assertion to an origin that was never checked; and a stalled endpoint must not pin a
        worker for longer than the exchange budget."""
        client = new_exchange_client()

        assert client.follow_redirects is False
        assert client.timeout.connect == EXCHANGE_CONNECT_TIMEOUT_SECONDS
        assert client.timeout.read == EXCHANGE_TIMEOUT_SECONDS


class RecordingServiceHooks:
    def __init__(self) -> None:
        self.successes: list[tuple[ServiceTypes, str, float]] = []
        self.failures: list[tuple[ServiceTypes, float, str | Exception, str]] = []

    async def async_service_success_hook(self, service: ServiceTypes, call_type: str, duration: float) -> None:
        self.successes.append((service, call_type, duration))

    async def async_service_failure_hook(
        self, service: ServiceTypes, duration: float, error: str | Exception, call_type: str
    ) -> None:
        self.failures.append((service, duration, error, call_type))


class RaisingServiceHooks:
    """Every hook raises, and each call is recorded first so a test can prove the sink kept
    calling through rather than bailing after the first failure."""

    def __init__(self) -> None:
        self.attempts: list[str] = []  # mutable-ok: a test spy accumulating calls in order

    async def async_service_success_hook(self, service: ServiceTypes, call_type: str, duration: float) -> None:
        self.attempts.append(f"success:{call_type}")
        raise RuntimeError("hook down")

    async def async_service_failure_hook(
        self, service: ServiceTypes, duration: float, error: str | Exception, call_type: str
    ) -> None:
        self.attempts.append(f"failure:{call_type}")
        raise RuntimeError("hook down")


class InlineExecutor(concurrent.futures.Executor):
    def submit(self, fn, /, *args, **kwargs):
        future: concurrent.futures.Future = concurrent.futures.Future()
        future.set_result(fn(*args, **kwargs))
        return future


class NeverRunsExecutor(concurrent.futures.Executor):
    """Accepts work and never runs it, standing in for a telemetry backend that has stalled, so a
    test can show the backlog stops growing instead of consuming memory for as long as traffic lasts."""

    def __init__(self) -> None:
        self.submitted = 0  # mutable-ok: a test spy counting accepted work

    def submit(self, fn, /, *args, **kwargs):
        self.submitted += 1
        return concurrent.futures.Future()


class TestServiceLoggingMetricsSink:
    def _sink(self, hooks) -> ServiceLoggingMetricsSink:
        return ServiceLoggingMetricsSink(service_logging_factory=lambda: hooks, executor=InlineExecutor())

    def test_success_maps_to_anthropic_wif_service(self):
        hooks = RecordingServiceHooks()

        self._sink(hooks).exchange_success(call_type="cold_mint", duration_seconds=0.2)

        assert hooks.successes == [(ServiceTypes.ANTHROPIC_WIF, "cold_mint", 0.2)]

    def test_a_stalled_backend_stops_accepting_work_instead_of_queueing_without_bound(self):
        stalled: Final = NeverRunsExecutor()
        sink: Final = ServiceLoggingMetricsSink(service_logging_factory=RecordingServiceHooks, executor=stalled)

        for _ in range(_METRICS_QUEUE_LIMIT + 500):
            sink.cache_hit()

        assert stalled.submitted == _METRICS_QUEUE_LIMIT, (
            "once the backlog is full further events are dropped, so request volume cannot grow it"
        )

    def test_a_drained_backlog_accepts_work_again(self):
        hooks: Final = RecordingServiceHooks()
        sink: Final = ServiceLoggingMetricsSink(service_logging_factory=lambda: hooks, executor=InlineExecutor())

        for _ in range(_METRICS_QUEUE_LIMIT + 10):
            sink.cache_hit()

        assert len(hooks.successes) == _METRICS_QUEUE_LIMIT + 10, (
            "an executor that actually runs releases each slot, so nothing is dropped"
        )

    def test_failure_maps_variant_and_redacted_summary(self):
        hooks = RecordingServiceHooks()
        error = TokenEndpointError(status_code=503, redacted_body="error: unavailable")

        self._sink(hooks).exchange_failure(call_type="refresh", duration_seconds=0.1, error=error)

        ((service, duration, emitted, call_type),) = hooks.failures
        assert service is ServiceTypes.ANTHROPIC_WIF
        assert duration == 0.1
        assert call_type == "refresh"
        assert isinstance(emitted, TokenExchangeEndpointFailure)
        assert str(emitted) == _error_summary(error)

    def test_transport_failure_gets_its_own_error_class(self):
        hooks = RecordingServiceHooks()

        self._sink(hooks).exchange_failure(
            call_type="refresh", duration_seconds=0.05, error=TokenTransportError(detail="ConnectError: boom")
        )

        ((_service, _duration, emitted, _call_type),) = hooks.failures
        assert isinstance(emitted, TokenExchangeTransportFailure)

    def test_cache_hit_maps_to_cache_service_with_zero_duration(self):
        hooks = RecordingServiceHooks()

        self._sink(hooks).cache_hit()

        assert hooks.successes == [(ServiceTypes.ANTHROPIC_WIF_CACHE, CALL_TYPE_CACHE_HIT, 0.0)]

    def test_end_to_end_reflected_assertion_never_reaches_the_hook(self):
        hooks = RecordingServiceHooks()
        exchange = make_exchange(EchoingUnauthorizedEndpoint(), metrics_sink=self._sink(hooks))

        result = exchange.get_token(make_params(), EXCHANGE_BASE)

        assert isinstance(result, TokenEndpointError)
        assert [call_type for _service, _duration, _emitted, call_type in hooks.failures] == ["cold_mint", "cold_mint"]
        for _service, _duration, emitted, _call_type in hooks.failures:
            assert DEFAULT_ASSERTION not in str(emitted)
            assert DEFAULT_ASSERTION not in repr(emitted)

    def test_raising_hooks_are_swallowed(self):
        hooks: Final = RaisingServiceHooks()
        sink: Final = self._sink(hooks)

        sink.exchange_success(call_type="cold_mint", duration_seconds=0.2)
        sink.cache_hit()
        sink.exchange_failure(call_type="cold_mint", duration_seconds=0.1, error=TokenTransportError(detail="boom"))

        assert hooks.attempts == ["success:cold_mint", "success:cache_hit", "failure:cold_mint"], (
            "every event is still handed to the hooks, and one raising hook does not stop the next"
        )
