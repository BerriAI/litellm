"""
Tests for issue #25595:
GroqChatConfig._transform_messages must strip `provider_specific_fields` from
assistant messages before the request reaches the Groq API.

Groq is strict about extra properties on messages and rejects them with
"provider_specific_fields is unsupported".
"""
import pytest

from litellm.llms.groq.chat.transformation import GroqChatConfig

_GROQ = GroqChatConfig()


def _make_messages(with_psf: bool):
    """Build a simple multi-turn messages list.

    When with_psf=True, the assistant message carries provider_specific_fields
    as litellm would set them after processing a streaming tool-call response.
    """
    assistant_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"},
            }
        ],
    }
    if with_psf:
        assistant_msg["provider_specific_fields"] = {"some_internal_key": "value"}

    return [
        {"role": "user", "content": "What is the weather?"},
        assistant_msg,
        {"role": "tool", "tool_call_id": "call_abc", "content": "Sunny, 25°C"},
    ]


def test_provider_specific_fields_stripped_from_assistant_message():
    """Core regression test: provider_specific_fields must not survive _transform_messages."""
    messages = _make_messages(with_psf=True)
    transformed = _GROQ._transform_messages(messages, model="llama3-8b-8192")

    assistant = next(m for m in transformed if m.get("role") == "assistant")
    assert "provider_specific_fields" not in assistant, (
        f"provider_specific_fields must be stripped before sending to Groq, "
        f"but found: {assistant.get('provider_specific_fields')}"
    )


def test_messages_without_provider_specific_fields_unchanged():
    """Verify that messages without the field are unaffected."""
    messages = _make_messages(with_psf=False)
    transformed = _GROQ._transform_messages(messages, model="llama3-8b-8192")

    assistant = next(m for m in transformed if m.get("role") == "assistant")
    assert "provider_specific_fields" not in assistant
    # Tool calls should be preserved
    assert assistant.get("tool_calls") is not None


def test_non_assistant_messages_not_modified():
    """user / tool messages must pass through without provider_specific_fields being injected."""
    messages = _make_messages(with_psf=True)
    transformed = _GROQ._transform_messages(messages, model="llama3-8b-8192")

    for msg in transformed:
        if msg.get("role") != "assistant":
            assert "provider_specific_fields" not in msg


def test_null_fields_stripped_from_assistant_message():
    """Existing null-field-stripping behaviour is not regressed (issue #5839)."""
    messages = [
        {"role": "assistant", "content": "Hello", "function_call": None}
    ]
    transformed = _GROQ._transform_messages(messages, model="llama3-8b-8192")
    assistant = transformed[0]
    assert assistant.get("function_call") is None or "function_call" not in assistant
