import asyncio
import concurrent.futures
import json
import logging
import threading
import time
from collections.abc import Callable, Mapping
from typing import Final
from urllib.parse import parse_qsl

import httpx
import pytest
from pydantic import SecretStr

from litellm.llms.base_llm.auth.token_exchange import (
    _REDACTION_CAP,
    ADVISORY_REFRESH_BACKOFF_SECONDS,
    FALLBACK_TOKEN_TTL_SECONDS,
    MAX_ASSERTION_BYTES,
    MAX_RESPONSE_BYTES,
    JwtBearerTokenExchangeEngine,
    redact_oauth_error_body,
)
from litellm.llms.base_llm.auth.types import (
    AssertionSourceError,
    ExchangeResult,
    InsecureTokenUrl,
    MalformedTokenResponse,
    MintedToken,
    TokenEndpointError,
    TokenExchangeSpec,
    TokenTransportError,
)
from litellm.secret_managers.main import OidcPathNotAllowedError, _resolve_oidc_file_path

DEFAULT_REF: Final = "oidc/env/TEST_ASSERTION"
DEFAULT_ASSERTION: Final = "test-jwt-assertion"


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordedRequest:
    def __init__(self, url: str, content: bytes, headers: Mapping[str, str], timeout: float) -> None:
        self.url = url
        self.content = content
        self.headers = dict(headers)
        self.timeout = timeout

    def json_body(self) -> dict:
        return json.loads(self.content)


class ScriptedPoster:
    """Returns scripted responses in order (repeating the last one); records requests."""

    def __init__(
        self,
        responses: list[httpx.Response],
        on_request: Callable[[RecordedRequest], None] | None = None,
    ) -> None:
        self.requests: list[RecordedRequest] = []
        self._responses = list(responses)
        self._on_request = on_request

    def post(self, url: str, *, content: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response:
        recorded = RecordedRequest(url, content, headers, timeout)
        self.requests.append(recorded)
        if self._on_request is not None:
            self._on_request(recorded)
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


class RaisingPoster:
    def __init__(self, error: Exception) -> None:
        self.calls = 0
        self._error = error

    def post(self, url: str, *, content: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response:
        self.calls += 1
        raise self._error


class ManualExecutor(concurrent.futures.Executor):
    """Records submissions; runs them only when the test says so."""

    def __init__(self) -> None:
        self.pending: list[Callable[[], None]] = []

    def submit(self, fn, /, *args, **kwargs):
        future: concurrent.futures.Future = concurrent.futures.Future()
        self.pending.append(lambda: fn(*args, **kwargs))
        return future

    def run_all(self) -> None:
        drained = list(self.pending)
        self.pending.clear()
        for job in drained:
            job()


class InlineExecutor(concurrent.futures.Executor):
    def submit(self, fn, /, *args, **kwargs):
        future: concurrent.futures.Future = concurrent.futures.Future()
        future.set_result(fn(*args, **kwargs))
        return future


def token_response(token: str = "sk-ant-oat01-minted", expires_in: int | None = 3600) -> httpx.Response:
    body = {"access_token": token, "token_type": "Bearer"}
    if expires_in is not None:
        body["expires_in"] = expires_in
    return httpx.Response(200, json=body)


def make_spec(**overrides) -> TokenExchangeSpec:
    base = {
        "token_url": "https://token.example/v1/oauth/token",
        "assertion_ref": DEFAULT_REF,
        "assertion_field": "assertion",
        "static_body": {
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "federation_rule_id": "fdrl_1",
            "organization_id": "org-1",
        },
        "body_encoding": "json",
        "request_headers": {"anthropic-beta": "oauth-2025-04-20,oidc-federation-2026-04-01"},
        "cache_key_identity": ("fdrl_1", "org-1", "", ""),
        "timeout_seconds": 2.0,
    }
    base.update(overrides)
    return TokenExchangeSpec(**base)


def make_engine(
    poster,
    reader: Mapping[str, str] | Callable[[str], str | None] | None = None,
    clock: FakeClock | None = None,
    executor: concurrent.futures.Executor | None = None,
    max_entries: int = 64,
) -> JwtBearerTokenExchangeEngine:
    resolved_reader = reader if callable(reader) else (reader or {DEFAULT_REF: DEFAULT_ASSERTION}).get
    return JwtBearerTokenExchangeEngine(
        poster=poster,
        assertion_reader=resolved_reader,
        clock=clock if clock is not None else FakeClock(),
        refresh_executor=executor if executor is not None else ManualExecutor(),
        max_entries=max_entries,
    )


def mint(engine: JwtBearerTokenExchangeEngine, spec: TokenExchangeSpec) -> MintedToken:
    result = engine.get_token(spec)
    assert isinstance(result, MintedToken)
    return result


class TestFreshMintWireExact:
    def test_json_body_and_headers(self):
        poster = ScriptedPoster([token_response(expires_in=3600)])
        clock = FakeClock(start=1_000.0)
        engine = make_engine(poster, clock=clock)
        spec = make_spec()

        result = mint(engine, spec)

        assert result.access_token.get_secret_value() == "sk-ant-oat01-minted"
        assert result.expires_at == 1_000.0 + 3600
        assert len(poster.requests) == 1
        request = poster.requests[0]
        assert request.url == "https://token.example/v1/oauth/token"
        assert request.timeout == 2.0
        assert request.headers == {
            "content-type": "application/json",
            "anthropic-beta": "oauth-2025-04-20,oidc-federation-2026-04-01",
        }
        assert request.json_body() == {
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "federation_rule_id": "fdrl_1",
            "organization_id": "org-1",
            "assertion": DEFAULT_ASSERTION,
        }

    def test_form_body_and_content_type(self):
        poster = ScriptedPoster([token_response()])
        engine = make_engine(poster)
        spec = make_spec(body_encoding="form")

        mint(engine, spec)

        request = poster.requests[0]
        assert request.headers["content-type"] == "application/x-www-form-urlencoded"
        assert dict(parse_qsl(request.content.decode())) == {
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "federation_rule_id": "fdrl_1",
            "organization_id": "org-1",
            "assertion": DEFAULT_ASSERTION,
        }


def test_cache_hit_zero_posts():
    poster = ScriptedPoster([token_response(expires_in=3600)])
    clock = FakeClock()
    engine = make_engine(poster, clock=clock)
    spec = make_spec()

    first = mint(engine, spec)
    clock.advance(100.0)
    second = mint(engine, spec)

    assert len(poster.requests) == 1
    assert second.access_token.get_secret_value() == first.access_token.get_secret_value()


@pytest.mark.parametrize(
    "remaining,expect_advisory_submit,expect_new_token",
    [
        (121.0, False, False),
        (120.0, True, False),
        (119.0, True, False),
        (31.0, True, False),
        (30.0, False, True),
        (29.0, False, True),
    ],
)
def test_window_boundaries(remaining: float, expect_advisory_submit: bool, expect_new_token: bool):
    poster = ScriptedPoster([token_response("old-token", expires_in=3600), token_response("new-token")])
    clock = FakeClock(start=1_000.0)
    executor = ManualExecutor()
    engine = make_engine(poster, clock=clock, executor=executor)
    spec = make_spec()

    mint(engine, spec)
    expires_at = 1_000.0 + 3600
    clock.now = expires_at - remaining
    result = mint(engine, spec)

    assert len(executor.pending) == (1 if expect_advisory_submit else 0)
    expected_token = "new-token" if expect_new_token else "old-token"
    assert result.access_token.get_secret_value() == expected_token
    assert len(poster.requests) == (2 if expect_new_token else 1)


def test_advisory_serve_stale_single_flight_backoff(caplog: pytest.LogCaptureFixture):
    poster = ScriptedPoster(
        [
            token_response("stale-token", expires_in=3600),
            httpx.Response(500, json={"error": "server_error"}),
            httpx.Response(500, json={"error": "server_error"}),
        ]
    )
    clock = FakeClock(start=1_000.0)
    executor = ManualExecutor()
    engine = make_engine(poster, clock=clock, executor=executor)
    spec = make_spec()

    mint(engine, spec)
    clock.now = 1_000.0 + 3600 - 100.0

    first = mint(engine, spec)
    second = mint(engine, spec)
    assert first.access_token.get_secret_value() == "stale-token"
    assert second.access_token.get_secret_value() == "stale-token"
    assert len(executor.pending) == 1

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        executor.run_all()
    assert len(poster.requests) == 2
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Advisory token refresh" in r.getMessage() for r in warning_records)
    assert "server_error" in caplog.text
    assert DEFAULT_ASSERTION not in caplog.text
    assert "stale-token" not in caplog.text

    within_backoff = mint(engine, spec)
    assert within_backoff.access_token.get_secret_value() == "stale-token"
    assert len(executor.pending) == 0

    clock.advance(ADVISORY_REFRESH_BACKOFF_SECONDS)
    after_backoff = mint(engine, spec)
    assert after_backoff.access_token.get_secret_value() == "stale-token"
    assert len(executor.pending) == 1
    executor.run_all()
    assert len(poster.requests) == 3


class GatedPoster:
    """Blocks the leader inside post() until the test releases it."""

    def __init__(self, response: httpx.Response) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self._calls_lock = threading.Lock()
        self._response = response

    def post(self, url: str, *, content: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response:
        with self._calls_lock:
            self.calls += 1
        self.entered.set()
        assert self.release.wait(timeout=10)
        return self._response


def _run_concurrent_get_token(
    engine: JwtBearerTokenExchangeEngine, spec: TokenExchangeSpec, poster: GatedPoster, thread_count: int
) -> list[ExchangeResult]:
    results: list[ExchangeResult] = []
    results_lock = threading.Lock()
    start_barrier = threading.Barrier(thread_count)

    def worker() -> None:
        start_barrier.wait()
        result = engine.get_token(spec)
        with results_lock:
            results.append(result)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(thread_count)]
    for thread in threads:
        thread.start()
    assert poster.entered.wait(timeout=10)
    time.sleep(0.3)
    poster.release.set()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    return results


def test_mandatory_single_leader():
    poster = GatedPoster(token_response("leader-token"))
    engine = make_engine(poster)
    spec = make_spec()

    results = _run_concurrent_get_token(engine, spec, poster, thread_count=5)

    assert poster.calls == 1
    assert len(results) == 5
    for result in results:
        assert isinstance(result, MintedToken)
        assert result.access_token.get_secret_value() == "leader-token"


def test_mandatory_failure_is_value():
    poster = GatedPoster(httpx.Response(500, json={"error": "server_error"}))
    engine = make_engine(poster)
    spec = make_spec()

    results = _run_concurrent_get_token(engine, spec, poster, thread_count=3)

    assert len(results) == 3
    for result in results:
        assert isinstance(result, TokenEndpointError)
        assert result.status_code == 500
        assert "server_error" in result.redacted_body


def test_lock_released_around_io():
    inner_spec = make_spec(
        token_url="https://inner.example/v1/oauth/token",
        cache_key_identity=("fdrl_inner", "org-1", "", ""),
    )
    engine_holder: dict[str, JwtBearerTokenExchangeEngine] = {}
    inner_results: list[ExchangeResult] = []

    class ReentrantPoster:
        def post(self, url: str, *, content: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response:
            if url == "https://token.example/v1/oauth/token":
                inner_results.append(engine_holder["engine"].get_token(inner_spec))
            return token_response()

    engine = make_engine(ReentrantPoster())
    engine_holder["engine"] = engine

    outcome: list[ExchangeResult] = []
    thread = threading.Thread(target=lambda: outcome.append(engine.get_token(make_spec())), daemon=True)
    thread.start()
    thread.join(timeout=10)

    assert not thread.is_alive(), "engine held its lock across poster I/O and deadlocked"
    assert len(outcome) == 1
    assert isinstance(outcome[0], MintedToken)
    assert len(inner_results) == 1
    assert isinstance(inner_results[0], MintedToken)


def test_401_retry_once_with_reread():
    assertions = {DEFAULT_REF: "assertion-v1"}

    def rotate_on_first_request(request: RecordedRequest) -> None:
        assertions[DEFAULT_REF] = "assertion-v2"

    poster = ScriptedPoster(
        [httpx.Response(401, json={"error": "invalid_grant"}), token_response()],
        on_request=rotate_on_first_request,
    )
    engine = make_engine(poster, reader=assertions.get)

    result = mint(engine, make_spec())

    assert result.access_token.get_secret_value() == "sk-ant-oat01-minted"
    assert len(poster.requests) == 2
    assert poster.requests[0].json_body()["assertion"] == "assertion-v1"
    assert poster.requests[1].json_body()["assertion"] == "assertion-v2"


def test_401_twice_is_endpoint_error():
    poster = ScriptedPoster([httpx.Response(401, json={"error": "invalid_grant"})])
    engine = make_engine(poster)

    result = engine.get_token(make_spec())

    assert isinstance(result, TokenEndpointError)
    assert result.status_code == 401
    assert "invalid_grant" in result.redacted_body
    assert len(poster.requests) == 2


class TestRedactionAndCaps:
    def test_object_body_reduced_to_rfc6749_fields(self):
        poster = ScriptedPoster(
            [
                httpx.Response(
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
        result = make_engine(poster).get_token(make_spec())

        assert isinstance(result, TokenEndpointError)
        assert result.status_code == 400
        assert "invalid_grant" in result.redacted_body
        assert "d" * 256 in result.redacted_body
        assert "d" * 257 not in result.redacted_body
        assert "https://errors.example/e1" in result.redacted_body
        assert "LEAKED-ASSERTION" not in result.redacted_body

    def test_nested_error_envelope_renders_readable_text(self):
        body = {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "federation_rule_id is not a well-formed fdrl_ tagged ID",
            },
        }
        result = redact_oauth_error_body(400, json.dumps(body))

        assert "invalid_request_error" in result.redacted_body
        assert "federation_rule_id is not a well-formed fdrl_ tagged ID" in result.redacted_body
        assert "{'" not in result.redacted_body

    def test_flat_rfc6749_shape_still_renders(self):
        body = {"error": "invalid_grant", "error_description": "bad request"}
        result = redact_oauth_error_body(400, json.dumps(body))

        assert result.redacted_body == "error: invalid_grant; error_description: bad request"

    def test_nested_error_message_is_capped_at_256_chars(self):
        body = {"error": {"type": "invalid_request_error", "message": "m" * 500}}
        result = redact_oauth_error_body(400, json.dumps(body))

        assert "m" * 256 in result.redacted_body
        assert "m" * 257 not in result.redacted_body

    def test_json_string_body_is_not_echoed(self):
        """A free-text body can carry back whatever was sent, so only structured OAuth fields are
        ever rendered into an error an operator or caller will see."""
        result = redact_oauth_error_body(400, json.dumps("s" * 500))
        assert result.redacted_body == "non-object error response omitted"
        assert "s" * 32 not in result.redacted_body

    def test_plain_text_body_is_not_echoed(self):
        result = redact_oauth_error_body(502, "t" * 500)
        assert result.redacted_body == "non-JSON error response omitted"
        assert "t" * 32 not in result.redacted_body

    def test_reflected_assertion_is_dropped(self):
        """An endpoint that echoes the submitted assertion must not put it in the log or the error."""
        assertion = SecretStr("eyJhbGciOiJSUzI1NiJ9.REFLECTEDPAYLOAD.signature")
        body = {"error": "invalid_grant", "error_description": f"bad assertion {assertion.get_secret_value()}"}

        result = redact_oauth_error_body(400, json.dumps(body), assertion)

        assert assertion.get_secret_value() not in result.redacted_body
        assert "REFLECTEDPAYLOAD" not in result.redacted_body

    def test_json_array_body_constant_message(self):
        result = redact_oauth_error_body(400, json.dumps(["a", "b"]))
        assert result.redacted_body == "non-object error response omitted"

    def test_oversized_body_never_parsed(self):
        poster = ScriptedPoster([httpx.Response(400, content=b'{"error": "' + b"x" * MAX_RESPONSE_BYTES + b'"}')])
        result = make_engine(poster).get_token(make_spec())

        assert isinstance(result, TokenEndpointError)
        assert result.redacted_body == "oversized error response omitted"

    def test_oversized_success_body_is_malformed(self):
        poster = ScriptedPoster(
            [httpx.Response(200, content=b'{"access_token": "' + b"x" * MAX_RESPONSE_BYTES + b'"}')]
        )
        result = make_engine(poster).get_token(make_spec())

        assert not isinstance(result, MintedToken)
        assert b"x" * 10 not in str(result).encode()


@pytest.mark.parametrize("access_token", ["", "   "])
def test_empty_access_token_is_malformed(access_token: str):
    poster = ScriptedPoster(
        [httpx.Response(200, json={"access_token": access_token, "token_type": "Bearer", "expires_in": 3600})]
    )
    result = make_engine(poster).get_token(make_spec())

    assert isinstance(result, MalformedTokenResponse)
    assert "empty access_token" in result.detail


def test_sentinel_leak_audit(caplog: pytest.LogCaptureFixture):
    jwt_sentinel = "JWT-SENTINEL-2c9f1e7ab4"
    token_sentinel = "sk-ant-oat01-TOKEN-SENTINEL-90d4c3aa17"
    ref = "oidc/env/SENTINEL_ASSERTION"

    with caplog.at_level(logging.DEBUG):
        success_poster = ScriptedPoster([token_response(token_sentinel, expires_in=3600)])
        success_clock = FakeClock()
        success_executor = ManualExecutor()
        engine = make_engine(success_poster, reader={ref: jwt_sentinel}, clock=success_clock, executor=success_executor)
        spec = make_spec(assertion_ref=ref)
        minted = mint(engine, spec)

        endpoint_error = make_engine(
            ScriptedPoster([httpx.Response(400, json={"error": "invalid_grant"})]), reader={ref: jwt_sentinel}
        ).get_token(spec)
        transport_error = make_engine(RaisingPoster(RuntimeError("boom")), reader={ref: jwt_sentinel}).get_token(spec)
        malformed_error = make_engine(
            ScriptedPoster([httpx.Response(200, json={"unexpected": "shape"})]), reader={ref: jwt_sentinel}
        ).get_token(spec)
        oversized_error = make_engine(
            ScriptedPoster([token_response()]), reader={ref: jwt_sentinel + "x" * MAX_ASSERTION_BYTES}
        ).get_token(spec)
        insecure_error = make_engine(ScriptedPoster([token_response()]), reader={ref: jwt_sentinel}).get_token(
            make_spec(assertion_ref=ref, token_url="http://token.example/v1/oauth/token")
        )

        success_poster._responses = [httpx.Response(500, json={"error": "server_error"})]
        success_clock.now = success_clock.now + 3600 - 100.0
        stale = engine.get_token(spec)
        success_executor.run_all()

    audited_values = [
        str(minted),
        repr(minted),
        str(minted.access_token),
        repr(minted.access_token),
        str(endpoint_error),
        repr(endpoint_error),
        str(transport_error),
        repr(transport_error),
        str(malformed_error),
        repr(malformed_error),
        str(oversized_error),
        repr(oversized_error),
        str(insecure_error),
        repr(insecure_error),
        str(stale),
        repr(stale),
        caplog.text,
    ]
    assert isinstance(oversized_error, AssertionSourceError)
    assert oversized_error.kind == "oversized"
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
        poster = ScriptedPoster([token_response()])
        engine = make_engine(poster, reader=lambda ref: assertion_value)

        result = engine.get_token(make_spec())

        assert isinstance(result, AssertionSourceError)
        assert result.kind == expected_kind
        assert result.source_ref == DEFAULT_REF
        assert len(poster.requests) == 0

    @pytest.mark.parametrize(
        "raised,expected_kind",
        [
            (OidcPathNotAllowedError("path outside allowed credential directories"), "disallowed_path"),
            (ValueError("Environment variable ANTHROPIC_IDENTITY_TOKEN not found"), "unreadable"),
            (OSError("permission denied"), "unreadable"),
        ],
    )
    def test_raising_reader(self, raised: Exception, expected_kind: str):
        poster = ScriptedPoster([token_response()])

        def reader(ref: str) -> str | None:
            raise raised

        result = make_engine(poster, reader=reader).get_token(make_spec())

        assert isinstance(result, AssertionSourceError)
        assert result.kind == expected_kind
        assert len(poster.requests) == 0

    def test_value_error_message_is_captured_as_detail(self):
        poster = ScriptedPoster([token_response()])

        def reader(ref: str) -> str | None:
            raise ValueError("Keycloak token endpoint returned invalid_client")

        result = make_engine(poster, reader=reader).get_token(make_spec())

        assert isinstance(result, AssertionSourceError)
        assert result.detail == "Keycloak token endpoint returned invalid_client"

    @pytest.mark.parametrize(
        "raised",
        [OidcPathNotAllowedError("path outside allowed credential directories"), OSError("permission denied")],
    )
    def test_non_value_error_never_populates_detail(self, raised: Exception):
        """Only the ValueError branch carries operator-diagnosable text; every other reader failure
        stays detail=None, matching today's file/env behavior byte-for-byte."""
        poster = ScriptedPoster([token_response()])

        def reader(ref: str) -> str | None:
            raise raised

        result = make_engine(poster, reader=reader).get_token(make_spec())

        assert isinstance(result, AssertionSourceError)
        assert result.detail is None

    def test_value_error_detail_is_capped(self):
        poster = ScriptedPoster([token_response()])
        overlong_message = "x" * (_REDACTION_CAP + 100)

        def reader(ref: str) -> str | None:
            raise ValueError(overlong_message)

        result = make_engine(poster, reader=reader).get_token(make_spec())

        assert isinstance(result, AssertionSourceError)
        assert result.detail == overlong_message[:_REDACTION_CAP]
        assert len(result.detail) == _REDACTION_CAP


class TestAssertionSourceOverridesEngineReader:
    """``TokenExchangeSpec.assertion_source`` is the dispatch mechanism a per-config identity
    source (internal_issuer, keycloak) plugs into the shared engine with -- it must win over the
    engine-level reader, and failures must still be reported against ``assertion_ref``."""

    def test_assertion_source_is_used_instead_of_the_reader(self):
        poster = ScriptedPoster([token_response()])
        engine = make_engine(poster, reader=lambda ref: "from-engine-reader")
        spec = make_spec(assertion_source=lambda: "from-assertion-source")

        result = mint(engine, spec)

        assert result.access_token.get_secret_value() == "sk-ant-oat01-minted"
        assert poster.requests[0].json_body()["assertion"] == "from-assertion-source"

    def test_reader_is_never_called_when_assertion_source_is_set(self):
        poster = ScriptedPoster([token_response()])
        calls: list[str] = []

        def reader(ref: str) -> str | None:
            calls.append(ref)
            return "from-engine-reader"

        engine = make_engine(poster, reader=reader)
        spec = make_spec(assertion_source=lambda: "from-assertion-source")

        mint(engine, spec)

        assert calls == []

    def test_assertion_source_failure_is_reported_against_assertion_ref(self):
        poster = ScriptedPoster([token_response()])
        engine = make_engine(poster, reader=lambda ref: "from-engine-reader")

        def raising_source() -> str | None:
            raise ValueError("keycloak token endpoint returned invalid_client")

        spec = make_spec(assertion_source=raising_source, assertion_ref="oidc/keycloak/abc123")

        result = engine.get_token(spec)

        assert isinstance(result, AssertionSourceError)
        assert result.source_ref == "oidc/keycloak/abc123"
        assert result.detail == "keycloak token endpoint returned invalid_client"
        assert len(poster.requests) == 0

    def test_assertion_source_is_re_invoked_on_401_retry(self):
        """The 401-retry re-read (``_reread_assertion``) must also prefer ``assertion_source``,
        not silently fall back to the engine reader for the reflected-assertion check."""
        values = iter(["assertion-v1", "assertion-v2"])
        poster = ScriptedPoster([httpx.Response(401, json={"error": "invalid_grant"}), token_response()])
        engine = make_engine(poster, reader=lambda ref: "from-engine-reader")
        spec = make_spec(assertion_source=lambda: next(values))

        result = mint(engine, spec)

        assert result.access_token.get_secret_value() == "sk-ant-oat01-minted"
        assert poster.requests[0].json_body()["assertion"] == "assertion-v1"
        assert poster.requests[1].json_body()["assertion"] == "assertion-v2"


class TestOidcFilePathAllowlistRaisesTypedError:
    """The engine classifies assertion-source failures by exception type (see
    TestAssertionGuards.test_raising_reader); that classification only works if the real
    oidc/file allowlist actually raises OidcPathNotAllowedError rather than a bare ValueError."""

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
        poster = ScriptedPoster([token_response()])
        engine = make_engine(poster)

        result = engine.get_token(make_spec(token_url="http://token.example/v1/oauth/token"))

        assert result == InsecureTokenUrl(host="token.example")
        assert "/v1/oauth/token" not in str(result)
        assert len(poster.requests) == 0

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:8080/v1/oauth/token",
            "http://127.0.0.1/v1/oauth/token",
            "http://[::1]/v1/oauth/token",
        ],
    )
    def test_localhost_http_allowed(self, url: str):
        poster = ScriptedPoster([token_response()])
        engine = make_engine(poster)

        result = engine.get_token(make_spec(token_url=url))

        assert isinstance(result, MintedToken)
        assert len(poster.requests) == 1


def test_cache_key_semantics():
    poster = ScriptedPoster([token_response()])
    assertions = {DEFAULT_REF: DEFAULT_ASSERTION, "oidc/env/OTHER": "other-assertion"}
    engine = make_engine(poster, reader=assertions.get)
    base_spec = make_spec()

    mint(engine, base_spec)
    mint(engine, make_spec(cache_key_identity=("fdrl_1", "org-1", "svc-2", "")))
    mint(engine, make_spec(token_url="https://other.example/v1/oauth/token"))
    mint(engine, make_spec(assertion_ref="oidc/env/OTHER"))
    assert len(poster.requests) == 4

    assertions[DEFAULT_REF] = "rotated-assertion"
    cached = mint(engine, base_spec)
    assert len(poster.requests) == 4
    assert cached.access_token.get_secret_value() == "sk-ant-oat01-minted"


def test_bounded_eviction():
    clock = FakeClock()

    class PerCallPoster:
        def __init__(self) -> None:
            self.calls = 0

        def post(self, url: str, *, content: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response:
            self.calls += 1
            body = json.loads(content)
            expires_in = 3600 + int(body["organization_id"].split("-")[1])
            return token_response(f"token-{body['organization_id']}", expires_in=expires_in)

    poster = PerCallPoster()
    engine = make_engine(poster, clock=clock, max_entries=64)

    def spec_for(index: int) -> TokenExchangeSpec:
        return make_spec(
            static_body={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "federation_rule_id": "fdrl_1",
                "organization_id": f"org-{index}",
            },
            cache_key_identity=("fdrl_1", f"org-{index}", "", ""),
        )

    for index in range(65):
        mint(engine, spec_for(index))
    assert poster.calls == 65

    mint(engine, spec_for(0))
    assert poster.calls == 66, "the earliest-expiring entry (index 0) should have been evicted"

    mint(engine, spec_for(2))
    assert poster.calls == 66, "a later-expiring entry should still be cached"

    mint(engine, spec_for(1))
    assert poster.calls == 67, "re-inserting index 0 should have evicted the next earliest-expiring entry"


@pytest.mark.parametrize("expires_in", [None, 0, -5])
def test_missing_or_nonsense_expires_in_gets_fallback_ttl(expires_in: int | None):
    poster = ScriptedPoster(
        [token_response("short-lived", expires_in=expires_in), token_response("reminted", expires_in=3600)]
    )
    clock = FakeClock(start=1_000.0)
    engine = make_engine(poster, clock=clock)
    spec = make_spec()

    first = mint(engine, spec)
    assert first.expires_at == 1_000.0 + FALLBACK_TOKEN_TTL_SECONDS

    clock.advance(FALLBACK_TOKEN_TTL_SECONDS + 1.0)
    second = mint(engine, spec)

    assert second.access_token.get_secret_value() == "reminted"
    assert len(poster.requests) == 2, "a token without a sane expires_in must never be cached forever"


async def test_aget_token_loop_responsive():
    class SleepingPoster:
        def post(self, url: str, *, content: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response:
            time.sleep(0.3)
            return token_response()

    engine = make_engine(SleepingPoster())
    spec = make_spec()
    ticks = {"count": 0}
    stop = asyncio.Event()

    async def ticker() -> None:
        while not stop.is_set():
            ticks["count"] += 1
            await asyncio.sleep(0.01)

    ticker_task = asyncio.create_task(ticker())
    result = await engine.aget_token(spec)
    stop.set()
    await ticker_task

    assert isinstance(result, MintedToken)
    assert result.access_token.get_secret_value() == "sk-ant-oat01-minted"
    assert ticks["count"] >= 5, "the event loop was blocked during aget_token"
    sync_result = engine.get_token(spec)
    assert sync_result == result


def test_invalidate_forces_refresh():
    poster = ScriptedPoster([token_response("token-1", expires_in=3600), token_response("token-2", expires_in=3600)])
    engine = make_engine(poster)
    spec = make_spec()

    first = mint(engine, spec)
    assert first.access_token.get_secret_value() == "token-1"

    engine.invalidate(spec)
    second = mint(engine, spec)
    assert second.access_token.get_secret_value() == "token-2"
    assert len(poster.requests) == 2

    third = mint(engine, spec)
    assert third.access_token.get_secret_value() == "token-2"
    assert len(poster.requests) == 2, "force_refresh must be one-shot"


def test_invalidate_unknown_spec_is_noop():
    poster = ScriptedPoster([token_response()])
    engine = make_engine(poster)

    engine.invalidate(make_spec())

    assert len(poster.requests) == 0


def test_advisory_failure_wakes_expired_follower_to_re_lead():
    poster = ScriptedPoster(
        [
            token_response("initial-token", expires_in=3600),
            httpx.Response(500, json={"error": "server_error"}),
            token_response("recovered-token", expires_in=3600),
        ]
    )
    clock = FakeClock(start=1_000.0)
    executor = ManualExecutor()
    engine = make_engine(poster, clock=clock, executor=executor)
    spec = make_spec()

    mint(engine, spec)
    clock.now = 1_000.0 + 3600 - 100.0
    mint(engine, spec)
    assert len(executor.pending) == 1

    clock.advance(200.0)
    results: list[ExchangeResult] = []
    follower = threading.Thread(target=lambda: results.append(engine.get_token(spec)), daemon=True)
    follower.start()
    time.sleep(0.3)
    executor.run_all()
    follower.join(timeout=10)

    assert not follower.is_alive()
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, MintedToken), f"follower was handed {result!r} instead of re-leading a fresh mint"
    assert result.access_token.get_secret_value() == "recovered-token"
    assert len(poster.requests) == 3


class TwoAttemptGatedPoster:
    """401 on the first attempt, then blocks the leader's retry until released."""

    def __init__(self) -> None:
        self.entered_second = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def post(self, url: str, *, content: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response:
        self.calls += 1
        if self.calls == 1:
            return httpx.Response(401, json={"error": "invalid_grant"})
        self.entered_second.set()
        assert self.release.wait(timeout=30)
        return token_response("slow-leader-token")


def test_follower_budget_outlasts_slow_two_attempt_leader():
    poster = TwoAttemptGatedPoster()
    engine = make_engine(poster)
    spec = make_spec(timeout_seconds=1.0)

    leader_results: list[ExchangeResult] = []
    leader = threading.Thread(target=lambda: leader_results.append(engine.get_token(spec)), daemon=True)
    leader.start()
    assert poster.entered_second.wait(timeout=10)

    follower_results: list[ExchangeResult] = []
    follower = threading.Thread(target=lambda: follower_results.append(engine.get_token(spec)), daemon=True)
    follower.start()
    time.sleep(6.5)
    poster.release.set()
    leader.join(timeout=10)
    follower.join(timeout=10)

    assert leader_results and isinstance(leader_results[0], MintedToken)
    assert follower_results, "follower never returned"
    follower_result = follower_results[0]
    assert isinstance(follower_result, MintedToken), (
        f"follower gave up before the leader's two-attempt worst case: {follower_result!r}"
    )
    assert follower_result.access_token.get_secret_value() == "slow-leader-token"


class FailThenGatePoster:
    """500 on the first call, then blocks until released before succeeding."""

    def __init__(self) -> None:
        self.entered_gate = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def post(self, url: str, *, content: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response:
        self.calls += 1
        if self.calls == 1:
            return httpx.Response(500, json={"error": "server_error"})
        self.entered_gate.set()
        assert self.release.wait(timeout=30)
        return token_response("round-two-token")


def test_new_round_timed_out_follower_never_returns_previous_rounds_error():
    poster = FailThenGatePoster()
    clock = FakeClock()
    engine = make_engine(poster, clock=clock)
    spec = make_spec(timeout_seconds=0.05)

    first = engine.get_token(spec)
    assert isinstance(first, TokenEndpointError)

    clock.advance(ADVISORY_REFRESH_BACKOFF_SECONDS + 1.0)
    leader = threading.Thread(target=lambda: engine.get_token(spec), daemon=True)
    leader.start()
    assert poster.entered_gate.wait(timeout=10)

    follower_result = engine.get_token(spec)

    assert isinstance(follower_result, TokenTransportError), (
        f"timed-out follower returned the previous round's error: {follower_result!r}"
    )
    assert "timed out" in follower_result.detail
    poster.release.set()
    leader.join(timeout=10)


def test_lead_backoff_fails_fast_within_window_and_expires_after():
    poster = ScriptedPoster([httpx.Response(500, json={"error": "server_error"})])
    clock = FakeClock()
    engine = make_engine(poster, clock=clock)
    spec = make_spec()

    first = engine.get_token(spec)
    assert isinstance(first, TokenEndpointError)
    assert len(poster.requests) == 1

    clock.advance(ADVISORY_REFRESH_BACKOFF_SECONDS - 1.0)
    second = engine.get_token(spec)
    assert second == first
    assert len(poster.requests) == 1, "a request inside the backoff window must make zero POSTs"

    clock.advance(1.0)
    third = engine.get_token(spec)
    assert isinstance(third, TokenEndpointError)
    assert len(poster.requests) == 2


def test_invalidate_bypasses_lead_backoff():
    poster = ScriptedPoster(
        [httpx.Response(500, json={"error": "server_error"}), token_response("post-invalidate", expires_in=3600)]
    )
    clock = FakeClock()
    engine = make_engine(poster, clock=clock)
    spec = make_spec()

    first = engine.get_token(spec)
    assert isinstance(first, TokenEndpointError)

    engine.invalidate(spec)
    second = engine.get_token(spec)

    assert isinstance(second, MintedToken)
    assert second.access_token.get_secret_value() == "post-invalidate"
    assert len(poster.requests) == 2
