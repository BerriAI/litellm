from typing import Final

import pytest
from fastapi import HTTPException

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.realtime_guardrails import (
    RealtimeUserText,
    realtime_guarded_item,
    realtime_guardrail_block,
)
from litellm.types.guardrails import GuardrailEventHooks


class RaisingGuardrail(CustomGuardrail):
    def __init__(self, error: Exception):
        super().__init__(guardrail_name="raising", event_hook=GuardrailEventHooks.pre_call, default_on=True)
        self.error = error

    async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
        raise self.error


def _item_create(item: object) -> dict:
    return {"type": "conversation.item.create", "item": item}


def test_only_the_typed_text_of_a_user_message_is_screened():
    client_event: Final = _item_create(
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_audio", "audio": "AAAA"}, {"type": "input_text", "text": "hello"}],
        }
    )

    assert realtime_guarded_item(client_event) == RealtimeUserText(text="hello")


@pytest.mark.parametrize(
    "client_event",
    [
        pytest.param(
            _item_create({"type": "message", "role": "user", "content": [{"type": "input_audio", "audio": "AAAA"}]}),
            id="audio-only user message",
        ),
        pytest.param(
            _item_create(
                {"type": "message", "role": "assistant", "content": [{"type": "input_text", "text": "hello"}]}
            ),
            id="assistant message",
        ),
        pytest.param(
            _item_create({"type": "function_call_output", "call_id": "call_1", "output": ""}), id="empty tool output"
        ),
        pytest.param(_item_create("hello"), id="item that is not an object"),
    ],
)
def test_items_without_client_text_are_not_screened(client_event):
    assert realtime_guarded_item(client_event) is None


@pytest.mark.asyncio
async def test_a_plain_string_block_reason_becomes_the_violation_message():
    guardrail: Final = RaisingGuardrail(HTTPException(status_code=400, detail="hello is not allowed"))

    block: Final = await realtime_guardrail_block("hello", (GuardrailEventHooks.pre_call,), {}, None, (guardrail,))

    assert block is not None
    assert block.message == "hello is not allowed"


@pytest.mark.asyncio
async def test_a_guardrail_that_crashes_is_not_treated_as_a_block():
    guardrail: Final = RaisingGuardrail(RuntimeError("guardrail backend down"))

    with pytest.raises(RuntimeError, match="guardrail backend down"):
        await realtime_guardrail_block("hello", (GuardrailEventHooks.pre_call,), {}, None, (guardrail,))
