import json

import httpx
import pytest

from litellm.exceptions import APIConnectionError, AuthenticationError, Timeout
from litellm.llms.chatgpt.common_utils import GetAccessTokenError
from litellm.llms.chatgpt.search.handler import ChatGPTSearchHandler


class StubAuthenticator:
    def get_access_token(self) -> str:
        return "oauth-token"

    def get_account_id(self) -> str:
        return "account-123"

    def get_api_base(self) -> str:
        return "https://default.chatgpt.test/backend-api/codex"


@pytest.mark.asyncio
async def test_search_forwards_payload_oauth_and_codex_headers() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status_code=429,
            headers={"content-type": "application/json", "x-codex-primary-used-percent": "100"},
            json={"error": {"message": "search limit reached"}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        response = await ChatGPTSearchHandler(
            authenticator=StubAuthenticator(),
            client=client,
        ).search(
            payload=b'{"future":{"preserved":true}}',
            model="gpt-5.6-sol",
            session_id="session-123",
            api_base="https://custom.chatgpt.test/backend-api/codex/",
            extra_headers={
                "authorization": "Bearer untrusted-token",
                "originator": "codex_vscode",
                "x-codex-turn-metadata": '{"turn_id":"turn-123"}',
            },
            timeout=12,
        )

    assert response.status_code == 429
    assert response.json() == {"error": {"message": "search limit reached"}}
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://custom.chatgpt.test/backend-api/codex/alpha/search"
    assert request.content == b'{"future":{"preserved":true}}'
    assert request.headers["authorization"] == "Bearer oauth-token"
    assert request.headers["chatgpt-account-id"] == "account-123"
    assert request.headers["session_id"] == "session-123"
    assert request.headers["accept"] == "application/json"
    assert request.headers["originator"] == "codex_vscode"
    assert json.loads(request.headers["x-codex-turn-metadata"]) == {"turn_id": "turn-123"}


@pytest.mark.asyncio
async def test_search_maps_device_login_failure_to_authentication_error() -> None:
    class FailingAuthenticator(StubAuthenticator):
        def get_access_token(self) -> str:
            raise GetAccessTokenError(message="device login required", status_code=401)

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500))) as client:
        with pytest.raises(AuthenticationError, match="device login required"):
            await ChatGPTSearchHandler(
                authenticator=FailingAuthenticator(),
                client=client,
            ).search(
                payload=b"{}",
                model="gpt-5.6-sol",
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transport_error", "expected_error"),
    [
        (httpx.ReadTimeout("upstream timed out"), Timeout),
        (httpx.ConnectError("upstream unavailable"), APIConnectionError),
    ],
)
async def test_search_maps_transport_failures(
    transport_error: httpx.RequestError,
    expected_error: type[Exception],
) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        transport_error.request = request
        raise transport_error

    async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(expected_error):
            await ChatGPTSearchHandler(
                authenticator=StubAuthenticator(),
                client=client,
            ).search(
                payload=b"{}",
                model="gpt-5.6-sol",
            )
