"""Tests for the shared guardrail content extraction helpers."""

from litellm.proxy.guardrails._content_utils import (
    apply_redacted_messages_back,
    build_inspection_messages,
    has_non_string_content,
    iter_message_text,
    walk_user_text,
)

# ── iter_message_text ────────────────────────────────────────────────────────────


def test_iter_message_text_string_messages():
    data = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    }
    assert list(iter_message_text(data)) == ["hello", "hi"]


def test_iter_message_text_multimodal_list_content():
    """VERIA-11: list-format content must be inspected, not silently skipped."""
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "AWS_KEY=AKIA..."},
                    {"type": "image_url", "image_url": {"url": "..."}},
                    {"type": "text", "text": "more secrets"},
                ],
            }
        ]
    }
    assert list(iter_message_text(data)) == ["AWS_KEY=AKIA...", "more secrets"]


def test_iter_message_text_responses_api_string_input():
    """fniVO9-F: Responses-API ``input`` must be inspectable when ``messages`` absent."""
    data = {"input": "tell me a secret"}
    assert list(iter_message_text(data)) == ["tell me a secret"]


def test_iter_message_text_responses_api_list_input_messages():
    data = {
        "input": [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
    }
    assert list(iter_message_text(data)) == ["first", "second"]


def test_iter_message_text_responses_api_list_input_content_parts():
    data = {
        "input": [
            {"type": "text", "text": "alpha"},
            {"type": "image_url", "image_url": {"url": "..."}},
            {"type": "text", "text": "beta"},
        ]
    }
    assert list(iter_message_text(data)) == ["alpha", "beta"]


def test_iter_message_text_responses_api_list_input_mixed_dicts_and_strings():
    """Greptile P2: mixed-list ``input`` with content-part dicts AND bare
    strings must yield every text fragment — read helpers used to truncate
    the bare strings."""
    data = {
        "input": [
            {"type": "text", "text": "from-dict"},
            "from-bare-string",
            {"type": "image_url", "image_url": {"url": "..."}},
            "another-bare-string",
        ]
    }
    assert list(iter_message_text(data)) == [
        "from-dict",
        "from-bare-string",
        "another-bare-string",
    ]


def test_iter_message_text_walks_messages_and_input_independently():
    """When both are present (rare), every fragment from either field is
    inspected — a stricter guarantee than "first one wins"."""
    data = {
        "messages": [{"role": "user", "content": "msg-content"}],
        "input": "input-content",
    }
    assert list(iter_message_text(data)) == ["msg-content", "input-content"]


def test_iter_message_text_empty_data():
    assert list(iter_message_text({})) == []
    assert list(iter_message_text({"messages": []})) == []
    assert list(iter_message_text({"input": ""})) == []


def test_iter_message_text_responses_api_input_text_and_output_text_parts():
    """LIT-4294: Responses-API content parts use ``input_text`` (request) and
    ``output_text`` (assistant); reading only ``type == "text"`` skipped every
    ``/v1/responses`` body and every text guardrail was a no-op on that path."""
    data = {
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "user text"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "assistant text"}],
            },
        ]
    }
    assert list(iter_message_text(data)) == ["user text", "assistant text"]


def test_iter_message_text_responses_api_tool_call_taxonomy():
    """LIT-4294: a Responses ``input`` list freely mixes message items,
    ``function_call`` (no ``role``), and ``function_call_output`` items. The
    old ``all(item has 'role')`` gate wrapped the whole list as one blob and
    yielded nothing; every text fragment must be visited independently."""
    data = {
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            },
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "get_weather",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [{"type": "input_text", "text": "sunny"}],
            },
        ]
    }
    assert list(iter_message_text(data)) == ["hello", "sunny"]


# ── walk_user_text ────────────────────────────────────────────────────────────


def test_walk_user_text_redacts_string_messages_in_place():
    data = {
        "messages": [
            {"role": "user", "content": "leak: AKIAEXAMPLE"},
            {"role": "assistant", "content": "ok"},
        ]
    }
    visited = walk_user_text(data, lambda s: s.replace("AKIAEXAMPLE", "[REDACTED]"))
    assert visited == 2
    assert data["messages"][0]["content"] == "leak: [REDACTED]"
    assert data["messages"][1]["content"] == "ok"


def test_walk_user_text_redacts_multimodal_text_parts():
    """VERIA-11: list-content text parts must be mutable for in-place redaction."""
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "AKIAEXAMPLE here"},
                    {"type": "image_url", "image_url": {"url": "..."}},
                    {"type": "text", "text": "no secret"},
                ],
            }
        ]
    }
    visited = walk_user_text(data, lambda s: s.replace("AKIAEXAMPLE", "[REDACTED]"))
    assert visited == 2
    parts = data["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "[REDACTED] here"}
    # Non-text part must be left untouched.
    assert parts[1] == {"type": "image_url", "image_url": {"url": "..."}}
    assert parts[2] == {"type": "text", "text": "no secret"}


def test_walk_user_text_redacts_responses_api_string_input():
    data = {"input": "leak AKIAEXAMPLE"}
    visited = walk_user_text(data, lambda s: s.replace("AKIAEXAMPLE", "[REDACTED]"))
    assert visited == 1
    assert data["input"] == "leak [REDACTED]"


def test_walk_user_text_redacts_responses_api_list_input():
    data = {
        "input": [
            {"type": "text", "text": "AKIAEXAMPLE"},
            {"type": "image_url", "image_url": {"url": "..."}},
        ]
    }
    visited = walk_user_text(data, lambda s: f"[redacted]{s}[/]")
    assert visited == 1
    assert data["input"][0] == {"type": "text", "text": "[redacted]AKIAEXAMPLE[/]"}
    assert data["input"][1] == {"type": "image_url", "image_url": {"url": "..."}}


def test_walk_user_text_redacts_responses_input_text_and_output_text_parts():
    """LIT-4294: ``walk_user_text`` must recognise the Responses text-part
    variants so masking guardrails (secret detection, PII) actually redact
    ``/v1/responses`` bodies instead of no-op'ing on them."""
    data = {
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "AKIAEXAMPLE"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "AKIAEXAMPLE too"}],
            },
        ]
    }
    visited = walk_user_text(data, lambda s: s.replace("AKIAEXAMPLE", "[REDACTED]"))
    assert visited == 2
    assert data["input"][0]["content"][0] == {
        "type": "input_text",
        "text": "[REDACTED]",
    }
    assert data["input"][1]["content"][0] == {
        "type": "output_text",
        "text": "[REDACTED] too",
    }


def test_walk_user_text_redacts_function_call_output_text():
    """LIT-4294: tool-call round-trips carry secrets in
    ``function_call_output.output``; the redact walker must descend into it
    while leaving ``function_call`` items (call_id, arguments) untouched."""
    data = {
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "AKIAEXAMPLE user"}],
            },
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "get_weather",
                "arguments": '{"AKIAEXAMPLE": 1}',
            },
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [{"type": "input_text", "text": "AKIAEXAMPLE tool"}],
            },
        ]
    }
    visited = walk_user_text(data, lambda s: s.replace("AKIAEXAMPLE", "[REDACTED]"))
    assert visited == 2
    assert data["input"][0]["content"][0]["text"] == "[REDACTED] user"
    assert data["input"][1] == {
        "type": "function_call",
        "call_id": "c1",
        "name": "get_weather",
        "arguments": '{"AKIAEXAMPLE": 1}',
    }
    assert data["input"][2]["output"][0]["text"] == "[REDACTED] tool"


def test_walk_user_text_redacts_function_call_output_string_output():
    """LIT-4294: ``function_call_output.output`` is also a plain string in
    OpenAI's Responses spec; the redact walker must handle both forms."""
    data = {
        "input": [
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": "AKIAEXAMPLE tool",
            },
        ]
    }
    visited = walk_user_text(data, lambda s: s.replace("AKIAEXAMPLE", "[REDACTED]"))
    assert visited == 1
    assert data["input"][0]["output"] == "[REDACTED] tool"


def test_walk_user_text_redacts_mixed_list_input():
    """Read and write helpers must agree on coverage — bare strings inside
    a mixed ``input`` list are inspected by both."""
    data = {
        "input": [
            {"type": "text", "text": "secret-one"},
            "secret-two",
            {"type": "image_url", "image_url": {"url": "..."}},
        ]
    }
    visited = walk_user_text(data, lambda s: f"<{s}>")
    assert visited == 2
    assert data["input"][0] == {"type": "text", "text": "<secret-one>"}
    assert data["input"][1] == "<secret-two>"
    assert data["input"][2] == {"type": "image_url", "image_url": {"url": "..."}}


# ── build_inspection_messages ─────────────────────────────────────────────────


def test_build_inspection_messages_chat_completion_passthrough():
    data = {
        "messages": [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ]
    }
    assert build_inspection_messages(data) == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
    ]


def test_build_inspection_messages_joins_multimodal_text_parts():
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first part"},
                    {"type": "image_url", "image_url": {"url": "..."}},
                    {"type": "text", "text": "second part"},
                ],
            }
        ]
    }
    assert build_inspection_messages(data) == [{"role": "user", "content": "first part\nsecond part"}]


def test_build_inspection_messages_lifts_responses_api_input():
    """fniVO9-F: ``input`` must be visible to hooks that POST messages to a remote API."""
    data = {"input": "responses-api content"}
    assert build_inspection_messages(data) == [{"role": "user", "content": "responses-api content"}]


def test_build_inspection_messages_drops_messages_with_no_text():
    data = {
        "messages": [
            {"role": "user", "content": ""},
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "..."}}],
            },
            {"role": "user", "content": "kept"},
        ]
    }
    assert build_inspection_messages(data) == [{"role": "user", "content": "kept"}]


def test_build_inspection_messages_responses_api_tool_call_taxonomy():
    """LIT-4294: mixed Responses ``input`` (message + function_call +
    function_call_output) must produce a non-empty inspection list. The
    customer's writeup reproduced a 422 from AIM's ``/fw/v1/analyze``
    (``No messages in the request``) when this synthesised list came back
    empty; every other guardrail silently scanned nothing on the same
    input."""
    data = {
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            },
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "get_weather",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [{"type": "input_text", "text": "sunny"}],
            },
        ]
    }
    assert build_inspection_messages(data) == [
        {"role": "user", "content": "hello"},
        {"role": "tool", "content": "sunny"},
    ]


def test_build_inspection_messages_function_call_output_defaults_to_tool():
    """LIT-4294: a Responses ``function_call_output`` item is the semantic
    equivalent of a chat-completions ``role: "tool"`` message, so the shared
    helper synthesises ``role: "tool"`` when the item has no explicit role.
    AIM's schema-safe coercion happens at the AIM call site, not here."""
    data = {
        "input": [
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [{"type": "input_text", "text": "tool text"}],
            },
        ]
    }
    assert build_inspection_messages(data) == [{"role": "tool", "content": "tool text"}]


def test_build_inspection_messages_function_call_output_preserves_explicit_role():
    """When ``function_call_output`` carries a caller-supplied ``role`` the
    shared helper preserves it rather than synthesising ``tool``."""
    data = {
        "input": [
            {
                "type": "function_call_output",
                "role": "assistant",
                "call_id": "c1",
                "output": [{"type": "input_text", "text": "tool text"}],
            },
        ]
    }
    assert build_inspection_messages(data) == [{"role": "assistant", "content": "tool text"}]


def test_build_inspection_messages_bare_content_part_preserves_explicit_role():
    """A bare content-part dict with an explicit ``role`` keeps it. Only
    absent roles get defaulted to ``user``."""
    data = {
        "input": [
            {"type": "input_text", "text": "no role"},
            {"type": "output_text", "role": "assistant", "text": "with role"},
        ]
    }
    assert build_inspection_messages(data) == [
        {"role": "user", "content": "no role"},
        {"role": "assistant", "content": "with role"},
    ]


def test_build_inspection_messages_message_item_preserves_role():
    """Responses message items carry a role explicitly; the shared helper
    passes it through untouched."""
    data = {
        "input": [
            {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "sys"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "asst"}]},
        ]
    }
    assert build_inspection_messages(data) == [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "asst"},
    ]


def test_build_inspection_messages_empty_data():
    assert build_inspection_messages({}) == []
    assert build_inspection_messages({"messages": []}) == []
    assert build_inspection_messages({"input": ""}) == []


def test_build_inspection_messages_drops_tool_and_function_roles():
    """AIM (and other remote guardrail APIs) validate against an
    OpenAI-compatible schema that requires ``tool_call_id`` on ``tool`` role
    messages. ``build_inspection_messages`` flattens messages to plain
    ``{role, content}`` and discards that metadata, so passing tool/function
    role messages through produced a 422 "Tool Call ID required on tool calls"
    and made AIM non-functional for any conversation with tool calls. They
    must be filtered out entirely."""
    data = {
        "messages": [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "show me milk"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search_products", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "milk $3.99"},
            {"role": "function", "name": "search_products", "content": "eggs $2.50"},
            {"role": "assistant", "content": "Here is the milk you asked for"},
        ]
    }
    assert build_inspection_messages(data) == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "show me milk"},
        {"role": "assistant", "content": "Here is the milk you asked for"},
    ]


# ── has_non_string_content ────────────────────────────────────────────────────


def test_has_non_string_content_string_messages():
    data = {"messages": [{"role": "user", "content": "hello"}]}
    assert has_non_string_content(data) is False


def test_has_non_string_content_multimodal_messages():
    data = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    assert has_non_string_content(data) is True


def test_has_non_string_content_responses_api_string_input():
    assert has_non_string_content({"input": "plain string"}) is False


def test_has_non_string_content_responses_api_list_input():
    assert has_non_string_content({"input": ["a", "b"]}) is True


def test_has_non_string_content_empty_data():
    assert has_non_string_content({}) is False
    assert has_non_string_content({"messages": []}) is False
    assert has_non_string_content({"input": ""}) is False


# ── apply_redacted_messages_back ──────────────────────────────────────────────


def test_apply_redacted_messages_back_chat_completion():
    data = {"messages": [{"role": "user", "content": "secret"}]}
    apply_redacted_messages_back(data, [{"role": "user", "content": "[REDACTED]"}])
    assert data["messages"] == [{"role": "user", "content": "[REDACTED]"}]
    assert "input" not in data


def test_apply_redacted_messages_back_responses_api_string_input():
    """A Responses-API request reads ``data["input"]``; writing only to
    ``messages`` would let unredacted text reach the LLM."""
    data = {"input": "secret payload"}
    apply_redacted_messages_back(data, [{"role": "user", "content": "[REDACTED]"}])
    assert data["input"] == "[REDACTED]"


def test_apply_redacted_messages_back_both_fields():
    """Defensive: when both fields are present, both are updated."""
    data = {
        "messages": [{"role": "user", "content": "old"}],
        "input": "old",
    }
    apply_redacted_messages_back(data, [{"role": "user", "content": "[REDACTED]"}])
    assert data["messages"] == [{"role": "user", "content": "[REDACTED]"}]
    assert data["input"] == "[REDACTED]"


def test_apply_redacted_messages_back_skips_input_when_not_string():
    """List ``input`` (multimodal Responses-API) is left alone — the
    multimodal-degrades-to-block guard runs upstream."""
    data = {"input": [{"type": "text", "text": "leak"}]}
    apply_redacted_messages_back(data, [{"role": "user", "content": "[REDACTED]"}])
    assert data["input"] == [{"type": "text", "text": "leak"}]


# -------------------------------------------------------------------
# LIT-4302: custom_tool_call_output walking
# -------------------------------------------------------------------

def test_iter_message_text_walks_custom_tool_call_output():
    """custom_tool_call_output items should yield their output text."""
    data = {
        "input": [
            {"type": "custom_tool_call_output", "output": "tool-secret"},
        ]
    }
    from litellm.proxy.guardrails._content_utils import iter_message_text
    texts = list(iter_message_text(data))
    assert "tool-secret" in texts


def test_walk_user_text_redacts_custom_tool_call_output():
    """walk_user_text should rewrite text inside custom_tool_call_output."""
    data = {
        "input": [
            {"type": "custom_tool_call_output", "output": "PII-data"},
        ]
    }
    count = walk_user_text(data, lambda t: t.replace("PII-data", "[MASKED]"))
    assert count >= 1
    assert data["input"][0]["output"] == "[MASKED]"


def test_build_inspection_messages_custom_tool_call_output():
    """build_inspection_messages should include custom_tool_call_output text."""
    data = {
        "input": [
            {"type": "custom_tool_call_output", "output": "custom-tool-leak"},
        ]
    }
    msgs = build_inspection_messages(data)
    assert any("custom-tool-leak" in m["content"] for m in msgs)
