import json
from typing import Final, Literal

import pytest
from fastapi import HTTPException

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.litellm_core_utils.realtime_guardrails import (
    RealtimeGuardrailBlock,
    RealtimeGuardrailRewrite,
    RealtimeUserText,
    realtime_guarded_item,
    realtime_guardrail_verdict,
    rewritten_item,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs


class RaisingGuardrail(CustomGuardrail):
    def __init__(self, error: Exception, realtime_violation_message: str | None = None) -> None:
        super().__init__(
            guardrail_name="raising",
            event_hook=GuardrailEventHooks.pre_call,
            default_on=True,
            realtime_violation_message=realtime_violation_message,
        )
        self.error = error

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLogging | None = None,
    ) -> GenericGuardrailAPIInputs:
        raise self.error


class MaskingGuardrail(CustomGuardrail):
    def __init__(self, word: str) -> None:
        super().__init__(guardrail_name=f"mask-{word}", event_hook=GuardrailEventHooks.pre_call, default_on=True)
        self.word = word
        self.screened: list[tuple[str, ...]] = []

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLogging | None = None,
    ) -> GenericGuardrailAPIInputs:
        texts: Final = inputs.get("texts", [])
        self.screened.append(tuple(texts))
        return {**inputs, "texts": [text.replace(self.word, "[MASKED]") for text in texts]}


def _item_create(item: object) -> dict:
    return {"type": "conversation.item.create", "item": item}


@pytest.mark.parametrize("role", ["user", "assistant", "system"])
def test_only_the_typed_text_of_a_message_is_screened_whatever_its_role(role: str) -> None:
    client_event: Final = _item_create(
        {
            "type": "message",
            "role": role,
            "content": [{"type": "input_audio", "audio": "AAAA"}, {"type": "input_text", "text": "hello"}],
        }
    )

    assert realtime_guarded_item(client_event) == RealtimeUserText(texts=("hello",))


@pytest.mark.parametrize(
    "client_event",
    [
        pytest.param(
            _item_create({"type": "message", "role": "user", "content": [{"type": "input_audio", "audio": "AAAA"}]}),
            id="audio-only user message",
        ),
        pytest.param(
            _item_create({"type": "function_call_output", "call_id": "call_1", "output": ""}), id="empty tool output"
        ),
        pytest.param(_item_create("hello"), id="item that is not an object"),
    ],
)
def test_items_without_client_text_are_not_screened(client_event: dict) -> None:
    assert realtime_guarded_item(client_event) is None


@pytest.mark.asyncio
async def test_a_plain_string_block_reason_becomes_the_violation_message() -> None:
    guardrail: Final = RaisingGuardrail(HTTPException(status_code=400, detail="hello is not allowed"))

    block: Final = await realtime_guardrail_verdict(("hello",), (GuardrailEventHooks.pre_call,), {}, None, (guardrail,))

    assert isinstance(block, RealtimeGuardrailBlock)
    assert block.message == "hello is not allowed"


@pytest.mark.asyncio
async def test_the_spoken_notice_never_repeats_the_block_reason_unless_one_is_configured() -> None:
    guardrail: Final = RaisingGuardrail(HTTPException(status_code=400, detail={"error": "keyword 'hello' detected"}))
    configured: Final = RaisingGuardrail(
        HTTPException(status_code=400, detail="hello is not allowed"), realtime_violation_message="Please rephrase"
    )

    block: Final = await realtime_guardrail_verdict(("hello",), (GuardrailEventHooks.pre_call,), {}, None, (guardrail,))
    configured_block: Final = await realtime_guardrail_verdict(
        ("hello",), (GuardrailEventHooks.pre_call,), {}, None, (configured,)
    )

    assert isinstance(block, RealtimeGuardrailBlock)
    assert block.message == "keyword 'hello' detected"
    assert "hello" not in block.spoken_message
    assert isinstance(configured_block, RealtimeGuardrailBlock)
    assert (configured_block.message, configured_block.spoken_message) == ("Please rephrase", "Please rephrase")


@pytest.mark.asyncio
async def test_masked_texts_come_back_as_a_rewrite_and_reach_later_guardrails_masked() -> None:
    first: Final = MaskingGuardrail("hunter2")
    second: Final = MaskingGuardrail("swordfish")

    verdict: Final = await realtime_guardrail_verdict(
        ("my hunter2", "and swordfish", "hi"), (GuardrailEventHooks.pre_call,), {}, None, (first, second)
    )

    assert verdict == RealtimeGuardrailRewrite(texts=("my [MASKED]", "and [MASKED]", "hi"))
    assert second.screened == [("my [MASKED]", "and swordfish", "hi")]


@pytest.mark.asyncio
async def test_texts_a_guardrail_leaves_unchanged_need_no_rewrite() -> None:
    verdict: Final = await realtime_guardrail_verdict(
        ("hello",), (GuardrailEventHooks.pre_call,), {}, None, (MaskingGuardrail("hunter2"),)
    )

    assert verdict is None


def test_a_rewrite_replaces_only_the_typed_text_parts_in_order() -> None:
    audio: Final = {"type": "input_audio", "audio": "AAAA"}
    client_event: Final = _item_create(
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "a"}, audio, {"type": "input_text", "text": "b"}],
        }
    )

    assert rewritten_item(client_event, ("A", "B")) == {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "A"}, audio, {"type": "input_text", "text": "B"}],
    }


def test_a_rewrite_replaces_the_tool_output() -> None:
    client_event: Final = _item_create({"type": "function_call_output", "call_id": "call_1", "output": "hunter2"})

    assert rewritten_item(client_event, ("[MASKED]",)) == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "[MASKED]",
    }


@pytest.mark.asyncio
async def test_a_guardrail_that_crashes_is_not_treated_as_a_block() -> None:
    guardrail: Final = RaisingGuardrail(RuntimeError("guardrail backend down"))

    with pytest.raises(RuntimeError, match="guardrail backend down"):
        await realtime_guardrail_verdict(("hello",), (GuardrailEventHooks.pre_call,), {}, None, (guardrail,))
