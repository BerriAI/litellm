from collections.abc import Iterator
from typing import Final
from urllib.parse import parse_qs

import httpx
import openai
import pytest
import respx

import litellm
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.exceptions import AuthenticationError
from litellm.llms.anthropic.messages.native.handler import anthropic_messages
from litellm.llms.base_llm.auth import provider_oauth
from litellm.llms.base_llm.auth.provider_oauth import provider_oauth_token
from litellm.llms.base_llm.auth.token_exchange import MAX_RESPONSE_BYTES
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from tests.unit.llms.base_llm.auth.test_token_exchange import FakeClock

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
        token_route: Final = upstream.post(_TOKEN_URL).mock(
            side_effect=(
                httpx.Response(200, json={"access_token": "initial-token", "expires_in": "300"}),
                httpx.Response(200, json={"access_token": "rotated-token", "expires_in": 300}),
                httpx.Response(200, json={"access_token": "other-origin-token", "expires_in": 300}),
            )
        )
        initial: Final = provider_oauth_token(_PARAMS, api_base=_PROVIDER_URL, provider="openai", model="test")
        cached: Final = provider_oauth_token(_PARAMS, api_base=_PROVIDER_URL, provider="anthropic", model="other")
        rotated: Final = provider_oauth_token(
            {**_PARAMS, "oauth_client_secret": "rotated-secret"},
            api_base=_PROVIDER_URL,
            provider="openai",
            model="test",
        )
        other: Final = provider_oauth_token(
            _PARAMS, api_base="https://other.example.com/v1", provider="openai", model="test"
        )

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
    token_url: Final = "https://idp.example.com/invalid"
    with respx.mock as upstream:
        token_route: Final = upstream.post(token_url).mock(return_value=response)
        redirect: Final = upstream.post("https://redirect.example.com/token").mock(return_value=httpx.Response(204))
        with pytest.raises(AuthenticationError, match="OAuth client credentials") as error:
            provider_oauth_token(
                {**_PARAMS, "oauth_token_url": token_url}, api_base=_PROVIDER_URL, provider="openai", model="test"
            )

    assert token_route.called
    assert not redirect.called
    assert "test-secret" not in str(error.value)


def test_oauth_stops_reading_an_oversized_token_response() -> None:
    class OversizedResponse(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            for _ in range(MAX_RESPONSE_BYTES // 65536 + 1):
                yield b" " * 65536
            pytest.fail("Read past the first chunk above the response limit")

    token_url: Final = "https://idp.example.com/oversized"
    with respx.mock as upstream:
        upstream.post(token_url).mock(return_value=httpx.Response(200, stream=OversizedResponse()))
        with pytest.raises(AuthenticationError, match="valid Bearer access token"):
            provider_oauth_token(
                {**_PARAMS, "oauth_token_url": token_url}, api_base=_PROVIDER_URL, provider="openai", model="test"
            )


@pytest.mark.parametrize("provider", ("openai", "custom_openai"))
@pytest.mark.parametrize("expires_in", (30, None), ids=("issuer-expiry", "fallback-expiry"))
def test_openai_compatible_completion_sends_refreshed_bearer(
    provider: str, expires_in: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock: Final = FakeClock(start=0.0)
    expiry_field: Final = {} if expires_in is None else {"expires_in": expires_in}
    after_expiry: Final = 61.0 if expires_in is None else 31.0
    monkeypatch.setattr(provider_oauth, "monotonic", clock)
    monkeypatch.setattr(provider_oauth, "_TOKENS", InMemoryCache(clock=clock))
    with respx.mock as upstream:
        token_route: Final = upstream.post(_TOKEN_URL).mock(
            side_effect=(
                httpx.Response(200, json={"access_token": "first-token", **expiry_field}),
                httpx.Response(200, json={"access_token": "fresh-token", **expiry_field}),
            ),
        )
        inference: Final = upstream.post(f"{_PROVIDER_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-auth-test",
                    "object": "chat.completion",
                    "created": 123,
                    "model": "gpt-6-luna",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}
                    ],
                },
            )
        )
        for elapsed in (0.0, 5.0, after_expiry - 5.0):
            clock.advance(elapsed)
            litellm.completion(
                model=f"{provider}/gpt-6-luna",
                messages=[{"role": "user", "content": "Hello"}],
                api_base=_PROVIDER_URL,
                api_key="static-key",
                extra_headers={"Authorization": "Bearer caller-key", "X-API-Key": "caller-key"},
                **_PARAMS,
            )

    assert token_route.call_count == 2
    assert [call.request.headers.get("authorization") for call in inference.calls] == [
        "Bearer first-token",
        "Bearer first-token",
        "Bearer fresh-token",
    ]
    assert [call.request.headers.get("x-api-key") for call in inference.calls] == [None, None, None]


def test_oauth_completion_preserves_the_supplied_openai_transport() -> None:
    token_url: Final = "https://idp.example.com/sdk-token"

    def infer(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{_PROVIDER_URL}/chat/completions"
        assert request.headers["authorization"] == "Bearer sdk-token"
        assert request.headers["x-connection-marker"] == "custom-client"
        assert "x-api-key" not in request.headers
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-auth-test",
                "object": "chat.completion",
                "created": 123,
                "model": "gpt-6-luna",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
            },
        )

    with (
        respx.mock as upstream,
        openai.OpenAI(
            api_key="sdk-key",
            base_url="https://previous.example.com/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(infer)),
            default_headers={"X-API-Key": "sdk-key", "X-Connection-Marker": "custom-client"},
        ) as client,
    ):
        token_route: Final = upstream.post(token_url).mock(
            return_value=httpx.Response(200, json={"access_token": "sdk-token"})
        )
        response: Final = litellm.completion(
            model="openai/gpt-6-luna",
            messages=[{"role": "user", "content": "Hello"}],
            client=client,
            api_base=_PROVIDER_URL,
            **{**_PARAMS, "oauth_token_url": token_url},
        )

    assert token_route.called
    assert response.choices[0].message.content == "OK"
    assert client.api_key == "sdk-key"


@pytest.mark.asyncio
async def test_native_messages_sends_bearer_without_static_api_key() -> None:
    token_url: Final = "https://idp.example.com/messages-token"
    api_base: Final = "https://provider.example.com"
    client: Final = AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport())
    async with client.client:
        with respx.mock as upstream:
            upstream.post(token_url).mock(return_value=httpx.Response(200, json={"access_token": "messages-token"}))
            inference: Final = upstream.post(f"{api_base}/v1/messages").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "id": "msg-auth-test",
                        "type": "message",
                        "model": "claude-sonnet-5",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "OK"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 3, "output_tokens": 1},
                    },
                )
            )
            await anthropic_messages(
                model="claude-sonnet-5",
                custom_llm_provider="anthropic",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=16,
                api_base=api_base,
                api_key="static-key",
                client=client,
                extra_headers={"Authorization": "Bearer caller-key", "X-API-Key": "caller-key"},
                **{**_PARAMS, "oauth_token_url": token_url},
            )

    assert inference.calls[0].request.headers["authorization"] == "Bearer messages-token"
    assert "x-api-key" not in inference.calls[0].request.headers
