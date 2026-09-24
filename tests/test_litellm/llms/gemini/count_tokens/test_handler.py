import json

import httpx
import pytest

from litellm.llms.gemini.count_tokens.handler import GoogleAIStudioTokenCounter

COUNT_TOKENS_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:countTokens"


@pytest.mark.asyncio
async def test_acount_tokens_sends_generate_content_request_when_system_or_tools_present():
    recorded: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        return httpx.Response(200, json={"totalTokens": 42})

    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

    result = await GoogleAIStudioTokenCounter().acount_tokens(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": "hello world"}]}],
        api_key="test-key",
        system_instruction={"parts": [{"text": "You are a helpful assistant"}]},
        tools=[{"function_declarations": [{"name": "get_weather"}]}],
        client=client,
    )

    assert result == {"totalTokens": 42}
    request = recorded[-1]
    assert request.url == COUNT_TOKENS_URL
    body = json.loads(request.content)
    assert "contents" not in body
    generate_content_request = body["generateContentRequest"]
    assert generate_content_request["model"] == "models/gemini-2.5-flash"
    assert generate_content_request["contents"] == [{"role": "user", "parts": [{"text": "hello world"}]}]
    assert generate_content_request["systemInstruction"] == {"parts": [{"text": "You are a helpful assistant"}]}
    assert generate_content_request["tools"][0]["function_declarations"][0]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_acount_tokens_keeps_contents_body_without_system_or_tools():
    recorded: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        return httpx.Response(200, json={"totalTokens": 4})

    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

    await GoogleAIStudioTokenCounter().acount_tokens(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": "hi"}]}],
        api_key="test-key",
        client=client,
    )

    body = json.loads(recorded[-1].content)
    assert body == {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}


@pytest.mark.asyncio
async def test_acount_tokens_wraps_unexpected_error_in_api_error():
    import litellm

    def _handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("transport exploded")

    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

    with pytest.raises(litellm.APIError) as excinfo:
        await GoogleAIStudioTokenCounter().acount_tokens(
            model="gemini-2.5-flash",
            contents=[{"role": "user", "parts": [{"text": "hello"}]}],
            api_key="test-key",
            client=client,
        )

    assert excinfo.value.status_code == 500


@pytest.mark.asyncio
async def test_acount_tokens_wraps_malformed_contents_error_in_api_error():
    import litellm

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))

    with pytest.raises(litellm.APIError):
        await GoogleAIStudioTokenCounter().acount_tokens(
            model="gemini-2.5-flash",
            contents=5,  # pyright: ignore[reportArgumentType]  # malformed caller input exercises the error boundary
            api_key="test-key",
            client=client,
        )
