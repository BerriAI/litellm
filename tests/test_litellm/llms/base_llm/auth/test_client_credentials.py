import base64
import logging
from collections.abc import Mapping
from typing import Final
from urllib.parse import parse_qsl, unquote

import httpx
import pytest

from litellm.llms.base_llm.auth.client_credentials import (
    fetch_keycloak_assertion,
    keycloak_assertion_source,
)
from litellm.llms.base_llm.auth.identity_source import KeycloakSource, identity_source_ref

TOKEN_URL: Final = "https://keycloak.example/realms/litellm/protocol/openid-connect/token"
CLIENT_ID: Final = "litellm"
CLIENT_SECRET_REF: Final = "oidc/env/KEYCLOAK_CLIENT_SECRET"
CLIENT_SECRET: Final = "s3cr3t-client-value"


class RecordedRequest:
    def __init__(self, url: str, content: bytes, headers: Mapping[str, str], timeout: float) -> None:
        self.url = url
        self.content = content
        self.headers = dict(headers)
        self.timeout = timeout

    def form_body(self) -> dict[str, str]:
        return dict(parse_qsl(self.content.decode()))


class ScriptedPoster:
    """Returns one scripted response per call; records every request it receives."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self.requests: list[RecordedRequest] = []
        self._responses = list(responses)

    def post(self, url: str, *, content: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response:
        self.requests.append(RecordedRequest(url, content, headers, timeout))
        return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]


class RaisingPoster:
    def __init__(self, error: Exception) -> None:
        self.calls = 0
        self._error = error

    def post(self, url: str, *, content: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response:
        self.calls += 1
        raise self._error


def make_config(
    auth_method: str = "client_secret_basic",
    scope: str | None = None,
    token_url: str = TOKEN_URL,
    client_secret_ref: str = CLIENT_SECRET_REF,
    client_id: str = CLIENT_ID,
) -> KeycloakSource:
    return KeycloakSource(
        token_url=token_url,
        client_id=client_id,
        client_secret_ref=client_secret_ref,
        auth_method=auth_method,  # pyright: ignore[reportArgumentType]  # test-only string widened for parametrization
        scope=scope,
    )


def secret_reader_returning(secret: str | None):
    def reader(ref: str) -> str | None:
        assert ref == CLIENT_SECRET_REF
        return secret

    return reader


DEFAULT_SECRET_READER: Final = secret_reader_returning(CLIENT_SECRET)


def token_response(access_token: str = "keycloak-minted-token") -> httpx.Response:
    return httpx.Response(200, json={"access_token": access_token, "token_type": "Bearer", "expires_in": 300})


class TestClientSecretBasic:
    def test_sends_basic_auth_header_and_no_secret_in_body(self):
        poster = ScriptedPoster([token_response("minted-1")])

        token = fetch_keycloak_assertion(
            make_config(auth_method="client_secret_basic"), poster=poster, secret_reader=DEFAULT_SECRET_READER
        )

        assert token == "minted-1"
        request = poster.requests[0]
        assert request.url == TOKEN_URL
        expected_auth = "Basic " + base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode("ascii")
        assert request.headers["authorization"] == expected_auth
        assert request.headers["content-type"] == "application/x-www-form-urlencoded"
        body = request.form_body()
        assert body["grant_type"] == "client_credentials"
        assert "client_secret" not in body
        assert "client_id" not in body

    def test_reserved_characters_are_form_encoded_before_basic(self):
        """RFC 6749 2.3.1 requires the client id and secret be application/x-www-form-urlencoded
        (Appendix B) before being base64'd into the Basic header; a raw join lets a reserved
        character in either value corrupt the ':'-joined pair Keycloak decodes back out."""
        client_id = "id:with+reserved% chars"
        client_secret = "secret:with+reserved% chars"
        poster = ScriptedPoster([token_response("minted-reserved")])

        fetch_keycloak_assertion(
            make_config(auth_method="client_secret_basic", client_id=client_id),
            poster=poster,
            secret_reader=secret_reader_returning(client_secret),
        )

        header = poster.requests[0].headers["authorization"]
        assert header.startswith("Basic ")
        decoded = base64.b64decode(header.removeprefix("Basic ")).decode("ascii")
        encoded_id, _, encoded_secret = decoded.partition(":")
        assert unquote(encoded_id) == client_id
        assert unquote(encoded_secret) == client_secret

    def test_scope_included_only_when_set(self):
        poster = ScriptedPoster([token_response()])
        fetch_keycloak_assertion(
            make_config(scope="openid profile"), poster=poster, secret_reader=DEFAULT_SECRET_READER
        )

        assert poster.requests[0].form_body()["scope"] == "openid profile"

        poster_no_scope = ScriptedPoster([token_response()])
        fetch_keycloak_assertion(make_config(scope=None), poster=poster_no_scope, secret_reader=DEFAULT_SECRET_READER)

        assert "scope" not in poster_no_scope.requests[0].form_body()


class TestClientSecretPost:
    def test_sends_client_id_and_secret_in_body_with_no_basic_header(self):
        poster = ScriptedPoster([token_response("minted-2")])

        token = fetch_keycloak_assertion(
            make_config(auth_method="client_secret_post"), poster=poster, secret_reader=DEFAULT_SECRET_READER
        )

        assert token == "minted-2"
        request = poster.requests[0]
        assert "authorization" not in request.headers
        body = request.form_body()
        assert body["grant_type"] == "client_credentials"
        assert body["client_id"] == CLIENT_ID
        assert body["client_secret"] == CLIENT_SECRET


class TestOnePostPerExchange:
    def test_exactly_one_post_per_call_no_cache(self):
        poster = ScriptedPoster([token_response("first"), token_response("second")])

        first = fetch_keycloak_assertion(make_config(), poster=poster, secret_reader=DEFAULT_SECRET_READER)
        second = fetch_keycloak_assertion(make_config(), poster=poster, secret_reader=DEFAULT_SECRET_READER)

        assert first == "first"
        assert second == "second"
        assert len(poster.requests) == 2


class TestInvalidClient:
    def test_400_invalid_client_surfaces_redacted_detail(self):
        poster = ScriptedPoster(
            [httpx.Response(400, json={"error": "invalid_client", "error_description": "unauthorized client"})]
        )

        with pytest.raises(ValueError, match="invalid_client") as exc_info:
            fetch_keycloak_assertion(make_config(), poster=poster, secret_reader=DEFAULT_SECRET_READER)

        assert "unauthorized client" in str(exc_info.value)
        assert "400" in str(exc_info.value)
        assert CLIENT_SECRET not in str(exc_info.value)

    def test_echoed_client_secret_is_never_reflected_into_the_error(self):
        """A misbehaving Keycloak that echoes the submitted client_secret back in its error body
        must never leak it into the exception the caller sees."""
        long_secret: Final = "reflectable-secret-0123456789"
        poster = ScriptedPoster(
            [httpx.Response(400, json={"error": "invalid_client", "error_description": f"got {long_secret} in body"})]
        )

        with pytest.raises(ValueError, match="keycloak") as exc_info:
            fetch_keycloak_assertion(make_config(), poster=poster, secret_reader=secret_reader_returning(long_secret))

        assert long_secret not in str(exc_info.value)
        assert "redacted" in str(exc_info.value)

    def test_echoed_short_client_secret_is_never_reflected_into_the_error(self):
        """Real Keycloak client secrets are often shorter than a JWT: the reflection probe must
        not silently stop protecting a secret just because it is under the probe's usual length."""
        short_secret: Final = "hand-set-14ch"
        poster = ScriptedPoster(
            [httpx.Response(400, json={"error": "invalid_client", "error_description": f"got {short_secret} in body"})]
        )

        with pytest.raises(ValueError, match="keycloak") as exc_info:
            fetch_keycloak_assertion(make_config(), poster=poster, secret_reader=secret_reader_returning(short_secret))

        assert short_secret not in str(exc_info.value)
        assert "redacted" in str(exc_info.value)


class TestUnreachable:
    def test_transport_failure_raises_diagnosable_value_error(self):
        poster = RaisingPoster(httpx.ConnectError("connection refused"))

        with pytest.raises(ValueError, match="ConnectError") as exc_info:
            fetch_keycloak_assertion(make_config(), poster=poster, secret_reader=DEFAULT_SECRET_READER)

        assert poster.calls == 1
        assert CLIENT_SECRET not in str(exc_info.value)


class TestNon2xx:
    def test_500_raises_value_error_with_status_code(self):
        poster = ScriptedPoster([httpx.Response(500, json={"error": "server_error"})])

        with pytest.raises(ValueError, match="500"):
            fetch_keycloak_assertion(make_config(), poster=poster, secret_reader=DEFAULT_SECRET_READER)


class TestResponseValidation:
    def test_missing_access_token_is_a_value_error(self):
        poster = ScriptedPoster([httpx.Response(200, json={"token_type": "Bearer"})])

        with pytest.raises(ValueError, match="schema validation"):
            fetch_keycloak_assertion(make_config(), poster=poster, secret_reader=DEFAULT_SECRET_READER)

    def test_empty_access_token_is_a_value_error(self):
        poster = ScriptedPoster([httpx.Response(200, json={"access_token": "   "})])

        with pytest.raises(ValueError, match="empty access_token"):
            fetch_keycloak_assertion(make_config(), poster=poster, secret_reader=DEFAULT_SECRET_READER)


class TestInsecureTokenUrl:
    def test_http_url_is_rejected_before_any_post(self):
        poster = ScriptedPoster([token_response()])

        with pytest.raises(ValueError, match="https"):
            fetch_keycloak_assertion(
                make_config(token_url="http://keycloak.example/token"),
                poster=poster,
                secret_reader=DEFAULT_SECRET_READER,
            )

        assert poster.requests == []


class TestMissingClientSecret:
    def test_unresolvable_secret_ref_raises_value_error_naming_the_ref_not_a_secret(self):
        poster = ScriptedPoster([token_response()])

        with pytest.raises(ValueError, match=CLIENT_SECRET_REF):
            fetch_keycloak_assertion(make_config(), poster=poster, secret_reader=secret_reader_returning(None))

        assert poster.requests == []


class TestKeycloakAssertionSource:
    def test_returns_a_callable_that_fetches_fresh_each_call(self):
        poster = ScriptedPoster([token_response("first"), token_response("second")])
        source = keycloak_assertion_source(make_config(), poster=poster, secret_reader=DEFAULT_SECRET_READER)

        assert source() == "first"
        assert source() == "second"
        assert len(poster.requests) == 2

    def test_propagates_the_underlying_fetch_failure(self):
        poster = ScriptedPoster([httpx.Response(400, json={"error": "invalid_client"})])
        source = keycloak_assertion_source(make_config(), poster=poster, secret_reader=DEFAULT_SECRET_READER)

        with pytest.raises(ValueError, match="invalid_client"):
            source()


class TestClientSecretNeverLeaks:
    """Regression coverage for the load-bearing property: a Keycloak client_secret must never
    surface in the assertion_ref, in any error message, or in a log record, however it fails."""

    def test_never_in_the_assertion_ref(self):
        config = make_config(client_secret_ref=CLIENT_SECRET_REF)

        ref = identity_source_ref(config)

        assert CLIENT_SECRET not in ref
        assert CLIENT_SECRET_REF not in ref

    def test_never_in_any_raised_error_message_across_every_failure_mode(self):
        config = make_config()
        failures = [
            lambda: fetch_keycloak_assertion(
                config,
                poster=ScriptedPoster([httpx.Response(400, json={"error": "invalid_client"})]),
                secret_reader=DEFAULT_SECRET_READER,
            ),
            lambda: fetch_keycloak_assertion(
                config, poster=RaisingPoster(httpx.ConnectError("boom")), secret_reader=DEFAULT_SECRET_READER
            ),
            lambda: fetch_keycloak_assertion(
                config,
                poster=ScriptedPoster([httpx.Response(500, json={"error": "server_error"})]),
                secret_reader=DEFAULT_SECRET_READER,
            ),
            lambda: fetch_keycloak_assertion(
                config, poster=ScriptedPoster([token_response()]), secret_reader=secret_reader_returning(None)
            ),
        ]
        for fail in failures:
            with pytest.raises(ValueError, match="keycloak") as exc_info:
                fail()
            assert CLIENT_SECRET not in str(exc_info.value)

    def test_never_in_a_log_record(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.DEBUG):
            poster = ScriptedPoster(
                [httpx.Response(400, json={"error": "invalid_client", "error_description": CLIENT_SECRET})]
            )
            with pytest.raises(ValueError, match="keycloak"):
                fetch_keycloak_assertion(make_config(), poster=poster, secret_reader=DEFAULT_SECRET_READER)
            fetch_keycloak_assertion(
                make_config(), poster=ScriptedPoster([token_response()]), secret_reader=DEFAULT_SECRET_READER
            )

        assert CLIENT_SECRET not in caplog.text
