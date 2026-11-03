"""The legacy Lakera guardrail reads client JSON that request setup hasn't normalized on pass-through routes."""

import json
from typing import Final
from unittest.mock import AsyncMock, patch

import httpx
import pytest

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
