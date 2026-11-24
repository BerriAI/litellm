from typing import Final
from urllib.parse import parse_qs

import httpx
import pytest
import respx

from litellm.exceptions import AuthenticationError
from litellm.llms.base_llm.auth.provider_oauth import provider_oauth_token

_TOKEN_URL: Final = "https://idp.example.com/oauth2/token"
_PROVIDER_URL: Final = "https://provider.example.com/v1"
_PARAMS: Final = {
    "oauth_client_credentials": True,
    "oauth_token_url": _TOKEN_URL,
    "oauth_client_id": "test-client",
    "oauth_client_secret": "test-secret",
    "oauth_scope": "inference",
}


def test_oauth_posts_credentials_and_caches_only_for_the_same_connection() -> None:
    with respx.mock as upstream:
        token_route = upstream.post(_TOKEN_URL).mock(
            side_effect=(
                httpx.Response(200, json={"access_token": "initial-token", "expires_in": "300"}),
                httpx.Response(200, json={"access_token": "rotated-token", "expires_in": 300}),
                httpx.Response(200, json={"access_token": "other-origin-token", "expires_in": 300}),
            )
        )
        initial = provider_oauth_token(_PARAMS, api_base=_PROVIDER_URL, provider="openai", model="test")
        cached = provider_oauth_token(_PARAMS, api_base=_PROVIDER_URL, provider="anthropic", model="other")
        rotated = provider_oauth_token(
            {**_PARAMS, "oauth_client_secret": "rotated-secret"},
            api_base=_PROVIDER_URL,
            provider="openai",
            model="test",
        )
        other = provider_oauth_token(_PARAMS, api_base="https://other.example.com/v1", provider="openai", model="test")

    assert (initial, cached, rotated, other) == (
        "initial-token",
        "initial-token",
        "rotated-token",
        "other-origin-token",
    )
    assert token_route.call_count == 3
    assert parse_qs(token_route.calls[0].request.content.decode()) == {
        "grant_type": ["client_credentials"],
        "client_id": ["test-client"],
        "client_secret": ["test-secret"],
        "scope": ["inference"],
    }


@pytest.mark.parametrize(
    "response",
    (
        httpx.Response(302, headers={"Location": "https://redirect.example.com/token"}),
        httpx.Response(403, text="test-secret"),
        httpx.Response(200, json={"access_token": "test-secret\r\n"}),
        httpx.Response(200, json={"access_token": "test-secret", "token_type": "MAC"}),
    ),
    ids=("redirect", "error", "malformed-bearer", "unsupported-type"),
)
def test_oauth_does_not_forward_client_credentials_or_expose_endpoint_response(response: httpx.Response) -> None:
    token_url = "https://idp.example.com/invalid"
    with respx.mock as upstream:
        token_route = upstream.post(token_url).mock(return_value=response)
        redirect = upstream.post("https://redirect.example.com/token").mock(return_value=httpx.Response(204))
        with pytest.raises(AuthenticationError, match="OAuth client credentials") as error:
            provider_oauth_token(
                {**_PARAMS, "oauth_token_url": token_url}, api_base=_PROVIDER_URL, provider="openai", model="test"
            )

    assert token_route.called
    assert not redirect.called
    assert "test-secret" not in str(error.value)
