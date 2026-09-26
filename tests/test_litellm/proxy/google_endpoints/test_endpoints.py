"""
Test for google_endpoints/endpoints.py
"""

import pytest
import sys, os
from dotenv import load_dotenv


from litellm.proxy.google_endpoints.endpoints import google_count_tokens
from litellm.types.llms.vertex_ai import TokenCountDetailsResponse
from starlette.requests import Request

load_dotenv()



@pytest.mark.asyncio
async def test_proxy_gemini_to_openai_like_model_token_counting():
    """
    Test the token counting endpoint for proxing gemini to openai-like models.
    """
    response: TokenCountDetailsResponse = await google_count_tokens(
        request=Request(
            scope={
                "type": "http",
                "parsed_body": (
                    ["contents"],
                    {"contents": [{"parts": [{"text": "Hello, how are you?"}]}]},
                ),
            }
        ),
        model_name="volcengine/foo",
    )

    assert response.get("totalTokens") > 0


@pytest.mark.asyncio
async def test_google_count_tokens_forwards_system_instruction_and_tools_to_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    import httpx
    import respx

    import litellm
    from litellm import Router
    from litellm.proxy import proxy_server

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "gemini-count",
                    "litellm_params": {
                        "model": "gemini/gemini-2.5-flash",
                        "api_key": "test-key",
                        "api_base": "https://gemini.test",
                    },
                }
            ]
        ),
    )
    system_instruction = {"parts": [{"text": "Your name is Doodle."}]}
    tools = [{"functionDeclarations": [{"name": "get_weather", "parameters": {"type": "object"}}]}]

    with respx.mock(assert_all_called=True) as upstream:
        count_route = upstream.post("https://gemini.test/v1beta/models/gemini-2.5-flash:countTokens").mock(
            return_value=httpx.Response(200, json={"totalTokens": 31, "promptTokensDetails": []})
        )
        response = await google_count_tokens(
            request=Request(
                scope={
                    "type": "http",
                    "parsed_body": (
                        ["contents", "systemInstruction", "tools"],
                        {
                            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                            "systemInstruction": system_instruction,
                            "tools": tools,
                        },
                    ),
                }
            ),
            model_name="gemini-count",
        )

    sent = json.loads(count_route.calls.last.request.content)["generateContentRequest"]
    assert response.get("totalTokens") == 31
    assert sent["systemInstruction"] == system_instruction
    assert sent["tools"] == tools


@pytest.mark.asyncio
async def test_google_count_tokens_unwraps_generate_content_request_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    import httpx
    import respx

    import litellm
    from litellm import Router
    from litellm.proxy import proxy_server

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "gemini-count",
                    "litellm_params": {
                        "model": "gemini/gemini-2.5-flash",
                        "api_key": "test-key",
                        "api_base": "https://gemini.test",
                    },
                }
            ]
        ),
    )
    inner_body = {
        "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
        "systemInstruction": {"parts": [{"text": "Your name is Doodle."}]},
        "tools": [{"functionDeclarations": [{"name": "get_weather", "parameters": {"type": "object"}}]}],
    }

    with respx.mock(assert_all_called=True) as upstream:
        count_route = upstream.post("https://gemini.test/v1beta/models/gemini-2.5-flash:countTokens").mock(
            return_value=httpx.Response(200, json={"totalTokens": 31, "promptTokensDetails": []})
        )
        response = await google_count_tokens(
            request=Request(
                scope={
                    "type": "http",
                    "parsed_body": (
                        ["generateContentRequest"],
                        {"generateContentRequest": inner_body},
                    ),
                }
            ),
            model_name="gemini-count",
        )

    sent = json.loads(count_route.calls.last.request.content)["generateContentRequest"]
    assert response.get("totalTokens") == 31
    assert sent["contents"] == inner_body["contents"]
    assert sent["systemInstruction"] == inner_body["systemInstruction"]
    assert sent["tools"] == inner_body["tools"]


@pytest.mark.asyncio
async def test_google_count_tokens_rejects_non_dict_tools_with_400() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await google_count_tokens(
            request=Request(
                scope={
                    "type": "http",
                    "parsed_body": (
                        ["contents", "tools"],
                        {
                            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                            "tools": ["just-a-string"],
                        },
                    ),
                }
            ),
            model_name="gemini-count",
        )

    assert exc_info.value.status_code == 400
