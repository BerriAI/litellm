"""Tests for the AIM guardrail's inspection-payload construction."""

from litellm.proxy.guardrails.guardrail_hooks.aim.aim import AimGuardrail


def test_aim_inspection_messages_coerces_chat_completions_tool_role_to_user():
    """LIT-4294: A valid chat-completions ``role: "tool"`` message carries a
    ``tool_call_id``, but the inspection flatten drops every field except
    ``role`` and ``content``. A bare ``tool`` message without ``tool_call_id``
    is schema-invalid per the OpenAI chat schema, and the customer's writeup
    reproduced AIM's ``/fw/v1/analyze`` returning 422 on exactly that shape.
    The AIM POST collapses the role to ``user``; the outbound request to the
    LLM is untouched."""
    data = {
        "messages": [
            {"role": "user", "content": "weather in SF"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        ]
    }
    assert AimGuardrail._build_aim_inspection_messages(data) == [
        {"role": "user", "content": "weather in SF"},
        {"role": "user", "content": "sunny"},
    ]


def test_aim_inspection_messages_coerces_non_standard_caller_role_to_user():
    """LIT-4294: A caller-supplied role outside {system, user, assistant}
    (e.g. ``developer``, ``function``) is coerced to ``user`` for the AIM
    POST, since AIM validates the payload against the OpenAI chat schema
    and rejects unknown roles the same way it rejects bare ``tool``."""
    data = {
        "messages": [
            {"role": "developer", "content": "system-ish instruction"},
            {"role": "user", "content": "normal user text"},
        ]
    }
    assert AimGuardrail._build_aim_inspection_messages(data) == [
        {"role": "user", "content": "system-ish instruction"},
        {"role": "user", "content": "normal user text"},
    ]


def test_aim_inspection_messages_coerces_responses_function_call_output_role():
    """LIT-4294: the shared helper synthesises ``role: "tool"`` for a
    Responses ``function_call_output`` item (semantic equivalent of
    chat-completions tool messages). AIM's schema-validating POST cannot
    carry ``tool_call_id`` in the flat inspection payload, so AIM collapses
    that ``tool`` role to ``user`` locally before POSTing."""
    data = {
        "input": [
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [{"type": "input_text", "text": "sunny"}],
            },
        ]
    }
    assert AimGuardrail._build_aim_inspection_messages(data) == [
        {"role": "user", "content": "sunny"},
    ]


def test_aim_inspection_messages_preserves_safe_roles():
    """Safe roles pass through untouched — the coercion only fires for
    roles the OpenAI chat schema flatten cannot represent standalone."""
    data = {
        "messages": [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    }
    assert AimGuardrail._build_aim_inspection_messages(data) == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
