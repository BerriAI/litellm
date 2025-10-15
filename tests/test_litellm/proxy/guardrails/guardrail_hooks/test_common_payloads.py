import copy

import pytest

from litellm.proxy.guardrails.guardrail_hooks.common_payloads import (
    GuardrailPayload,
    get_guardrail_input_payload,
)


@pytest.mark.parametrize("call_type", ["completion", "image_generation"])
def test_payload_prefers_existing_messages(call_type):
    original_data = {
        "messages": [{"role": "user", "content": "hello"}],
        "prompt": "ignored",
    }
    data = copy.deepcopy(original_data)

    payload = get_guardrail_input_payload(data=data, call_type=call_type)

    assert payload is not None
    assert payload.kind == "messages"
    assert payload.had_messages_key is True
    assert payload.to_message_list() == original_data["messages"]

    masked_messages = [{"role": "user", "content": "redacted"}]
    payload.apply_masked_messages(data=data, masked_messages=masked_messages)

    assert data["messages"] == masked_messages
    assert data["prompt"] == original_data["prompt"]


def test_image_generation_prompt_extraction_and_masking():
    data = {"prompt": "draw a cat"}

    payload = get_guardrail_input_payload(data=data, call_type="image_generation")

    assert payload is not None
    assert payload.kind == "text"
    assert payload.text_values == ["draw a cat"]
    assert payload.had_messages_key is False

    synthetic_messages = payload.to_message_list()
    assert synthetic_messages == [{"role": "user", "content": "draw a cat"}]

    masked_messages = [{"role": "user", "content": "safe prompt"}]
    payload.apply_masked_messages(data=data, masked_messages=masked_messages)

    assert data["prompt"] == "safe prompt"
    assert "messages" not in data


def test_image_generation_masking_handles_structured_message_content():
    data = {"prompt": "original"}
    payload = get_guardrail_input_payload(data=data, call_type="image_generation")

    assert payload is not None

    masked_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "masked"},
                {"type": "text", "text": " prompt"},
            ],
        }
    ]

    payload.apply_masked_messages(data=data, masked_messages=masked_messages)

    assert data["prompt"] == "masked prompt"
    assert "messages" not in data


def test_apply_masked_messages_without_masked_values_keeps_original_prompt():
    data = {"prompt": "keep me"}
    payload = GuardrailPayload(
        kind="text",
        text_values=["keep me"],
        data_pointer="prompt",
        metadata={"text_value_format": "single"},
        had_messages_key=False,
    )

    payload.apply_masked_messages(data=data, masked_messages=[])

    assert data["prompt"] == "keep me"
    assert "messages" not in data
