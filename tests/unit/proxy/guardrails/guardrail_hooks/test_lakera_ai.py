"""The legacy Lakera guardrail reads client JSON that request setup hasn't normalized on pass-through routes."""

import json
from typing import Final
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.lakera_ai import lakeraAI_Moderation

VENDOR: Final = "http://vendor.invalid"
CLEAN: Final = httpx.Response(200, json={"results": [{"flagged": False}]}, request=httpx.Request("POST", VENDOR))


async def _sent_to_lakera(data: dict[str, object], call_type: str = "pass_through_endpoint") -> tuple[str, ...]:
    guardrail: Final = lakeraAI_Moderation(api_key="key", api_base=VENDOR)
    with patch.object(guardrail.async_handler, "post", new=AsyncMock(return_value=CLEAN)) as post:
        await guardrail._check(data=data, user_api_key_dict=UserAPIKeyAuth(api_key="key"), call_type=call_type)
    return tuple(call.kwargs["data"] for call in post.await_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("token_arrays", [[1, 2, 3], [[1, 2], [3, 4]]])
async def test_token_arrays_have_no_text_to_send(token_arrays: list[object]) -> None:
    assert await _sent_to_lakera({"input": token_arrays}) == ()


@pytest.mark.asyncio
async def test_string_entries_of_a_list_input_are_sent() -> None:
    sent: Final = await _sent_to_lakera({"input": ["hello", "there"]})

    assert [json.loads(body)["input"] for body in sent] == ["hello\nthere"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    ["card 4111", [{"type": "input_text", "text": "card 4111"}, {"type": "input_image", "image_url": "http://x"}]],
)
async def test_a_responses_message_in_a_list_input_is_sent(content: object) -> None:
    sent: Final = await _sent_to_lakera({"input": [{"role": "user", "content": content}]})

    assert [json.loads(body)["input"] for body in sent] == ["card 4111"]


@pytest.mark.asyncio
async def test_a_pass_through_body_with_no_messages_or_input_sends_nothing() -> None:
    contents: Final = {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]}

    assert await _sent_to_lakera(contents) == ()


@pytest.mark.asyncio
async def test_a_native_body_with_no_messages_or_input_still_reaches_lakera() -> None:
    assert await _sent_to_lakera({"prompt": "hello"}, call_type="text_completion") == ("",)


@pytest.mark.asyncio
@pytest.mark.parametrize(("strings", "text"), [(["a", "", "b"], "a\n\nb"), (["a", ""], "a\n")])
async def test_empty_strings_of_a_list_input_keep_their_place(strings: list[object], text: str) -> None:
    sent: Final = await _sent_to_lakera({"input": strings})

    assert [json.loads(body)["input"] for body in sent] == [text]


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
