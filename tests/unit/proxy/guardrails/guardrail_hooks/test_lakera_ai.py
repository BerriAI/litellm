import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.lakera_ai import lakeraAI_Moderation


@pytest.mark.asyncio
async def test_prompt_check_appends_replayed_server_tool_call_arguments_to_system_message(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "add_function_to_prompt", False)
    route: Final = respx_mock.post("https://api.lakera.ai/v1/prompt_injection").mock(
        return_value=httpx.Response(200, json={"results": [{"flagged": False}]})
    )
    moderation: Final = lakeraAI_Moderation(api_key="test-key", moderation_check="pre_call")

    await moderation._check(
        data={
            "messages": [
                {"role": "system", "content": "You are a shopping assistant."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_01",
                            "type": "function",
                            "function": {"name": "lookup_order", "arguments": '{"order_id": "A-17"}'},
                        }
                    ],
                    "provider_specific_fields": {
                        "server_tool_calls": [
                            {
                                "id": "srvtoolu_01",
                                "type": "function",
                                "function": {"name": "web_fetch", "arguments": '{"url": "https://example.com"}'},
                            }
                        ]
                    },
                },
            ]
        },
        user_api_key_dict=UserAPIKeyAuth(),
        call_type="completion",
    )

    assert json.loads(route.calls.last.request.content)["input"][0] == {
        "role": "system",
        "content": 'You are a shopping assistant. Function Input: {"url": "https://example.com"} {"order_id": "A-17"}',
    }


@pytest.mark.asyncio
async def test_prompt_check_scans_a_replayed_turn_that_only_has_server_tool_calls(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "add_function_to_prompt", False)
    route: Final = respx_mock.post("https://api.lakera.ai/v1/prompt_injection").mock(
        return_value=httpx.Response(200, json={"results": [{"flagged": False}]})
    )
    moderation: Final = lakeraAI_Moderation(api_key="test-key", moderation_check="pre_call")

    await moderation._check(
        data={
            "messages": [
                {"role": "system", "content": "You are a shopping assistant."},
                {
                    "role": "assistant",
                    "content": "Searching.",
                    "tool_calls": None,
                    "provider_specific_fields": {
                        "server_tool_calls": [
                            {
                                "id": "srvtoolu_01",
                                "type": "function",
                                "function": {"name": "web_search", "arguments": '{"query": "order A-17"}'},
                            }
                        ]
                    },
                },
            ]
        },
        user_api_key_dict=UserAPIKeyAuth(),
        call_type="completion",
    )

    assert json.loads(route.calls.last.request.content)["input"][0] == {
        "role": "system",
        "content": 'You are a shopping assistant. Function Input: {"query": "order A-17"}',
    }
