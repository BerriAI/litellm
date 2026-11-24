from collections.abc import Callable
from dataclasses import replace
from typing import Final
from urllib.parse import parse_qs

import httpx
import pytest

from litellm.litellm_core_utils import oauth_token_exchange
from litellm.litellm_core_utils.oauth_token_exchange import (
    OAuthTokenExchangeConfig,
    OAuthTokenExchangeError,
    TokenExchangeProfile,
    aexchange_token,
    build_token_exchange_form,
    exchange_token,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


class _AsyncCaptureTransport(httpx.AsyncBaseTransport):
    def __init__(self, responder: Callable[[httpx.Request], httpx.Response]) -> None:
        self._responder = responder

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self._responder(request)


def _config(
    *,
    token_endpoint: str = "https://identity.example.com/oauth2/token",
    client_id: str = "client-9306",
    client_secret: str = "secret-9306",
    profile: TokenExchangeProfile = "rfc8693",
    scope: str | None = "graph.scope",
    audience: str | None = "graph-api",
) -> OAuthTokenExchangeConfig:
    return OAuthTokenExchangeConfig(
        token_endpoint=token_endpoint,
        client_id=client_id,
        client_secret=client_secret,
        profile=profile,
        scope=scope,
        audience=audience,
    )


def _sync_client(
    status_code: int,
    payload: object,
) -> tuple[HTTPHandler, list[httpx.Request]]:
    requests: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code=status_code, json=payload, request=request)

    return HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond))), requests


def _decode_form(request: httpx.Request) -> dict[str, str]:
    return {name: values[0] for name, values in parse_qs(request.content.decode("utf-8")).items()}


def test_rfc8693_form_contains_exact_fields_and_optional_scope_and_audience() -> None:
    config: Final = _config()

    form: Final = build_token_exchange_form(config, "subject-token-9306")

    assert form == {
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "client_id": "client-9306",
        "client_secret": "secret-9306",
        "subject_token": "subject-token-9306",
        "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        "scope": "graph.scope",
        "audience": "graph-api",
    }


@pytest.mark.parametrize(
    ("scope", "audience", "expected_optional_fields"),
    [
        (None, None, {}),
        ("", "", {}),
        ("graph.scope", None, {"scope": "graph.scope"}),
        (None, "graph-api", {"audience": "graph-api"}),
    ],
)
def test_rfc8693_form_omits_empty_or_unset_optional_fields(
    scope: str | None,
    audience: str | None,
    expected_optional_fields: dict[str, str],
) -> None:
    config: Final = _config(scope=scope, audience=audience)

    form: Final = build_token_exchange_form(config, "subject-token-9306")

    assert form == {
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "client_id": "client-9306",
        "client_secret": "secret-9306",
        "subject_token": "subject-token-9306",
        "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        **expected_optional_fields,
    }


def test_jwt_bearer_obo_form_requires_scope_and_ignores_audience() -> None:
    config: Final = _config(profile="jwt_bearer_obo", audience="ignored-audience")

    form: Final = build_token_exchange_form(config, "subject-token-9306")

    assert form == {
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "client_id": "client-9306",
        "client_secret": "secret-9306",
        "assertion": "subject-token-9306",
        "scope": "graph.scope",
        "requested_token_use": "on_behalf_of",
    }


@pytest.mark.parametrize("scope", [None, "", "   "])
def test_jwt_bearer_obo_rejects_empty_scope(scope: str | None) -> None:
    with pytest.raises(OAuthTokenExchangeError) as error:
        _config(profile="jwt_bearer_obo", scope=scope)

    assert error.value.status_code == 400
    assert error.value.message == "jwt_bearer_obo requires a non-empty token exchange scope"


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://identity.example.com/token",
        "https:///token",
        "https://identity.example.com:invalid/token",
        "not-a-url",
    ],
)
def test_config_rejects_invalid_token_endpoint(endpoint: str) -> None:
    with pytest.raises(OAuthTokenExchangeError) as error:
        _config(token_endpoint=endpoint)

    assert error.value.status_code == 400
    assert error.value.message == "token_exchange_endpoint must be an HTTPS URL with a host"


def test_cache_separates_each_config_field_and_subject_token() -> None:
    base_config: Final = _config(
        token_endpoint="https://identity.example.com/cache-separation-9306",
        client_id="cache-client-9306",
        client_secret="cache-secret-9306",
        profile="rfc8693",
        scope="cache-scope-9306",
        audience="cache-audience-9306",
    )
    config_and_subjects: Final = (
        (base_config, "cache-subject-9306"),
        (replace(base_config, token_endpoint="https://identity.example.com/other"), "cache-subject-9306"),
        (replace(base_config, client_id="other-client"), "cache-subject-9306"),
        (replace(base_config, client_secret="other-secret"), "cache-subject-9306"),
        (replace(base_config, profile="jwt_bearer_obo"), "cache-subject-9306"),
        (replace(base_config, scope="other-scope"), "cache-subject-9306"),
        (replace(base_config, audience="other-audience"), "cache-subject-9306"),
        (base_config, "other-subject"),
    )
    response_count: Final[list[int]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        response_count.append(len(response_count) + 1)
        return httpx.Response(
            status_code=200,
            json={"access_token": f"cached-token-{len(response_count)}", "expires_in": 3600},
            request=request,
        )

    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    returned_tokens: Final = tuple(
        exchange_token(client, config, subject_token) for config, subject_token in config_and_subjects
    )
    repeated_tokens: Final = tuple(
        exchange_token(client, config, subject_token) for config, subject_token in config_and_subjects
    )

    assert len(response_count) == len(config_and_subjects)
    assert returned_tokens == repeated_tokens


def test_exchange_applies_cache_safety_margin() -> None:
    cache_margin: Final = oauth_token_exchange.OAUTH_TOKEN_EXCHANGE_CACHE_SAFETY_MARGIN_SECONDS
    no_cache_client, no_cache_requests = _sync_client(
        200,
        {"access_token": "ttl-boundary-token-9306", "expires_in": cache_margin},
    )
    no_cache_config: Final = _config(token_endpoint="https://identity.example.com/ttl-boundary-9306")
    cache_client, cache_requests = _sync_client(
        200,
        {"access_token": "ttl-cached-token-9306", "expires_in": cache_margin + 60},
    )
    cache_config: Final = _config(token_endpoint="https://identity.example.com/ttl-cached-9306")

    exchange_token(no_cache_client, no_cache_config, "ttl-boundary-subject-9306")
    exchange_token(no_cache_client, no_cache_config, "ttl-boundary-subject-9306")
    exchange_token(cache_client, cache_config, "ttl-cached-subject-9306")
    exchange_token(cache_client, cache_config, "ttl-cached-subject-9306")

    assert len(no_cache_requests) == 2
    assert len(cache_requests) == 1


@pytest.mark.parametrize(
    ("status_code", "expected_status"),
    [(400, 401), (429, 429), (503, 503)],
)
def test_sync_exchange_maps_error_status_and_redacts_secrets(status_code: int, expected_status: int) -> None:
    subject_token: Final = "header.secret.subject"
    client_secret: Final = "secret-error-9306"
    client, _ = _sync_client(
        status_code,
        {
            "error": "invalid_grant",
            "error_description": f"subject={subject_token}, secret={client_secret}",
        },
    )

    with pytest.raises(OAuthTokenExchangeError) as error:
        exchange_token(client, _config(client_secret=client_secret), subject_token)

    assert error.value.status_code == expected_status
    assert error.value.message == "OAuth token exchange failed: invalid_grant: subject=[REDACTED], secret=[REDACTED]"
    assert client_secret not in error.value.message
    assert subject_token not in error.value.message


def test_sync_exchange_posts_to_endpoint_with_exact_form() -> None:
    config: Final = _config(token_endpoint="https://identity.example.com/sync-9306")
    client, requests = _sync_client(200, {"access_token": "sync-token-9306", "expires_in": 3600})

    token: Final = exchange_token(client, config, "sync-subject-9306")

    assert str(requests[0].url) == config.token_endpoint
    assert _decode_form(requests[0]) == build_token_exchange_form(config, "sync-subject-9306")
    assert token == "sync-token-9306"


@pytest.mark.asyncio
async def test_async_exchange_posts_to_endpoint_with_exact_form() -> None:
    config: Final = _config(token_endpoint="https://identity.example.com/async-9306")
    requests: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status_code=200,
            json={"access_token": "async-token-9306", "expires_in": 3600},
            request=request,
        )

    client: Final = AsyncHTTPHandler(transport=_AsyncCaptureTransport(respond))
    async with client.client:
        token: Final = await aexchange_token(client, config, "async-subject-9306")

    assert str(requests[0].url) == config.token_endpoint
    assert _decode_form(requests[0]) == build_token_exchange_form(config, "async-subject-9306")
    assert token == "async-token-9306"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_status"),
    [(400, 401), (429, 429), (503, 503)],
)
async def test_async_exchange_maps_error_status(status_code: int, expected_status: int) -> None:
    client: Final = AsyncHTTPHandler(
        transport=_AsyncCaptureTransport(
            lambda request: httpx.Response(
                status_code=status_code,
                json={"error": "invalid_grant", "error_description": "request failed"},
                request=request,
            )
        )
    )
    async with client.client:
        with pytest.raises(OAuthTokenExchangeError) as error:
            await aexchange_token(client, _config(), "async-status-subject-9306")

    assert error.value.status_code == expected_status
