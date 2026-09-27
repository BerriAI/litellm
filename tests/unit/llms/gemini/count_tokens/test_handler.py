import json
import sys

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
async def test_acount_tokens_ignores_gemini_api_base_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_BASE", "https://gateway.example.com")
    recorded: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        return httpx.Response(200, json={"totalTokens": 4})

    await GoogleAIStudioTokenCounter().acount_tokens(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": "hi"}]}],
        api_key="test-key",
        client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )

    assert recorded[-1].url == COUNT_TOKENS_URL


@pytest.mark.asyncio
async def test_acount_tokens_uses_deployment_api_base():
    recorded: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        return httpx.Response(200, json={"totalTokens": 4})

    await GoogleAIStudioTokenCounter().acount_tokens(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": "hi"}]}],
        api_key="test-key",
        api_base="https://explicit.example.com",
        client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )

    assert recorded[-1].url == "https://explicit.example.com/v1beta/models/gemini-2.5-flash:countTokens"


@pytest.mark.asyncio
async def test_acount_tokens_sends_generate_content_request_with_system_only():
    recorded: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        return httpx.Response(200, json={"totalTokens": 9})

    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

    await GoogleAIStudioTokenCounter().acount_tokens(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": "hi"}]}],
        api_key="test-key",
        system_instruction={"parts": [{"text": "be terse"}]},
        client=client,
    )

    body = json.loads(recorded[-1].content)
    assert body == {
        "generateContentRequest": {
            "model": "models/gemini-2.5-flash",
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "systemInstruction": {"parts": [{"text": "be terse"}]},
        }
    }


@pytest.mark.asyncio
async def test_acount_tokens_sends_generate_content_request_with_tools_only():
    recorded: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        return httpx.Response(200, json={"totalTokens": 9})

    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

    await GoogleAIStudioTokenCounter().acount_tokens(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": "hi"}]}],
        api_key="test-key",
        tools=[{"function_declarations": [{"name": "get_weather"}]}],
        client=client,
    )

    body = json.loads(recorded[-1].content)
    assert body == {
        "generateContentRequest": {
            "model": "models/gemini-2.5-flash",
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "tools": [{"function_declarations": [{"name": "get_weather"}]}],
        }
    }


@pytest.mark.asyncio
async def test_acount_tokens_reaches_gemini_without_google_genai_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "google.genai.types", None)
    recorded: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        return httpx.Response(200, json={"totalTokens": 7})

    contents = [
        {"role": "model", "parts": [{"function_call": {"name": "Bash", "args": {"command": "ls"}}}]},
        {"role": "user", "parts": [{"function_response": {"name": "Bash", "response": {"content": "a.txt"}}}]},
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "id": "toolu_1",
                        "name": "Bash",
                        "response": {"content": "ok"},
                        "willContinue": None,
                    }
                },
                "not-a-dict-part",
            ],
        },
        {"role": "user", "parts": [{"functionResponse": "malformed"}]},
        "not-a-dict-content",
    ]

    result = await GoogleAIStudioTokenCounter().acount_tokens(
        model="gemini-2.5-flash",
        contents=contents,
        api_key="test-key",
        client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )

    assert result == {"totalTokens": 7}
    assert json.loads(recorded[-1].content) == {
        "contents": [
            contents[0],
            contents[1],
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": "Bash", "response": {"content": "ok"}}},
                    "not-a-dict-part",
                ],
            },
            {"role": "user", "parts": [{"functionResponse": "malformed"}]},
            "not-a-dict-content",
        ]
    }
