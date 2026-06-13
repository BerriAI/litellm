"""Differential parity for the /v1/responses forward path.

The Responses inbound parser maps a Responses request onto the chat IR. v1's
oracle is ``LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request``,
which produces an OpenAI chat-completion request dict. There is no ``responses``
provider wire (the schema always rides the IR), so the round-trip used for
anthropic_messages does not apply; instead this gate runs v1 in-process at HEAD
and asserts the v2 IR is field-for-field equivalent to v1's chat request:
the messages (role + content + tool_calls/tool results), the system message,
the tools, tool_choice, response_format, reasoning_effort and the sampling
params. A mutation in the parser (a dropped field, a wrong tool_choice arm, a
lost consecutive-function_call merge, a mis-mapped reasoning effort) diverges
from v1.

The fail-closed rows assert every shape the chat IR cannot carry returns a
typed ``unsupported`` so dispatch falls back to v1 instead of silently dropping
it (researcher-6 §2.4/§2.5).
"""

import copy
import json

import pytest
from expression import Option

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig as V1,
)

from litellm.translation.inbound.responses import parse_request

MODEL = "gpt-4o"


def _v1_chat(request: dict) -> dict:
    body = dict(request)
    model = body.pop("model")
    input_param = body.pop("input")
    stream = body.pop("stream", None)
    return V1.transform_responses_api_request_to_chat_completion_request(
        model=model,
        input=input_param,
        responses_api_request=body,
        stream=stream,
    )


def _opt(value: Option) -> object:
    return value.some if value.is_some() else None


# ---- forward equivalence over the SERVE corpus ----------------------------

_SERVE: dict[str, dict] = {
    "string_input": {"model": MODEL, "input": "hello"},
    "message_text": {
        "model": MODEL,
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "hi there"}]}
        ],
    },
    "instructions_system": {
        "model": MODEL,
        "instructions": "be terse",
        "input": "hello",
    },
    "function_tool_and_choice": {
        "model": MODEL,
        "input": "weather?",
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "weather",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        "tool_choice": {"type": "function", "name": "get_weather"},
    },
    "tool_choice_cursor_tool": {
        "model": MODEL,
        "input": "x",
        "tools": [{"type": "function", "name": "f", "parameters": {"type": "object"}}],
        "tool_choice": {"type": "tool"},
    },
    "tool_choice_required_string": {
        "model": MODEL,
        "input": "x",
        "tools": [{"type": "function", "name": "f", "parameters": {"type": "object"}}],
        "tool_choice": "required",
    },
    "function_call_then_output": {
        "model": MODEL,
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "go"}]},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get",
                "arguments": '{"q": 1}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "result text",
            },
        ],
        "tools": [
            {"type": "function", "name": "get", "parameters": {"type": "object"}}
        ],
    },
    "consecutive_function_calls_merge": {
        "model": MODEL,
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "go"}]},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "a",
                "arguments": "{}",
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "b",
                "arguments": "{}",
            },
        ],
        "tools": [
            {"type": "function", "name": "a", "parameters": {"type": "object"}},
            {"type": "function", "name": "b", "parameters": {"type": "object"}},
        ],
    },
    "reasoning_effort": {
        "model": MODEL,
        "input": "think",
        "reasoning": {"effort": "high"},
    },
    "max_output_tokens_and_sampling": {
        "model": MODEL,
        "input": "x",
        "max_output_tokens": 256,
        "temperature": 0.5,
        "top_p": 0.9,
        "parallel_tool_calls": True,
        "user": "u1",
    },
    "text_format_json_object": {
        "model": MODEL,
        "input": "x",
        "text": {"format": {"type": "json_object"}},
    },
}


def _ir_messages(ir) -> list:
    out = []
    for message in ir.messages:
        blocks = []
        for block in message.content:
            if block.tag == "text":
                blocks.append({"type": "text", "text": block.text.text})
            elif block.tag == "tool_use":
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.tool_use.id,
                        "name": block.tool_use.name,
                        "args": block.tool_use.arguments.value,
                    }
                )
            elif block.tag == "tool_result":
                result = block.tool_result
                blocks.append(
                    {
                        "type": "tool_result",
                        "id": result.tool_use_id,
                        "text": (
                            result.content.text
                            if result.content.tag == "text"
                            else None
                        ),
                    }
                )
            else:
                blocks.append({"type": block.tag})
        out.append({"role": message.role, "blocks": blocks})
    return out


def _v1_messages(chat: dict) -> list:
    """Project v1's chat messages onto the same role/block shape the IR carries:
    system rides ChatRequest.system (excluded here), tool messages become a
    tool_result block in a user turn, assistant tool_calls become tool_use
    blocks. This is the inverse of the inbound mapping the IR encodes."""
    out: list = []
    for message in chat.get("messages", []):
        role = message.get("role")
        if role == "system":
            continue
        if role == "tool":
            out.append(
                {
                    "role": "user",
                    "blocks": [
                        {
                            "type": "tool_result",
                            "id": message.get("tool_call_id"),
                            "text": message.get("content"),
                        }
                    ],
                }
            )
            continue
        blocks: list = []
        content = message.get("content")
        if isinstance(content, str) and content:
            blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if part.get("type") in ("text", "input_text", "output_text"):
                    blocks.append({"type": "text", "text": part.get("text")})
        for call in message.get("tool_calls") or []:
            fn = call.get("function", {})
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.get("id"),
                    "name": fn.get("name"),
                    "args": json.loads(fn.get("arguments") or "{}"),
                }
            )
        out.append({"role": role, "blocks": blocks})
    return out


@pytest.mark.parametrize("name", sorted(_SERVE))
def test_forward_ir_matches_v1_chat_request(name: str) -> None:
    request = copy.deepcopy(_SERVE[name])
    parsed = parse_request(copy.deepcopy(request))
    assert (
        parsed.is_ok()
    ), f"{name}: parse failed: {parsed.error.summary if parsed.is_error() else ''}"
    ir = parsed.ok
    chat = _v1_chat(request)

    assert _ir_messages(ir) == _v1_messages(chat), f"{name}: messages diverge"

    v1_system = next(
        (
            m.get("content")
            for m in chat.get("messages", [])
            if m.get("role") == "system"
        ),
        None,
    )
    ir_system = ir.system[0].text if len(ir.system) > 0 else None
    assert ir_system == v1_system, f"{name}: system diverges"

    ir_tools = [t.name for t in ir.tools]
    v1_tools = [
        t.get("function", {}).get("name")
        for t in chat.get("tools", [])
        if t.get("type") == "function"
    ]
    assert ir_tools == v1_tools, f"{name}: tools diverge"

    assert _ir_tool_choice(ir) == chat.get(
        "tool_choice"
    ), f"{name}: tool_choice diverges"
    assert _opt(ir.reasoning_effort) == chat.get("reasoning_effort"), name
    assert _opt(ir.params.max_tokens) == chat.get("max_tokens"), name
    assert _opt(ir.params.temperature) == chat.get("temperature"), name
    assert _opt(ir.params.top_p) == chat.get("top_p"), name
    assert _opt(ir.parallel_tool_calls) == chat.get("parallel_tool_calls"), name
    assert _opt(ir.user) == chat.get("user"), name


def _ir_tool_choice(ir) -> object:
    match ir.tool_choice:
        case Option(tag="some", some=choice):
            if choice.tag == "auto":
                return "auto"
            if choice.tag == "required":
                return "required"
            if choice.tag == "none":
                return "none"
            return {"type": "function", "function": {"name": choice.specific}}
        case _:
            return None


# ---- fail-closed rows -----------------------------------------------------

_FALLBACK: dict[str, tuple[dict, str]] = {
    "previous_response_id": (
        {"model": MODEL, "input": "x", "previous_response_id": "resp_1"},
        "previous_response_id",
    ),
    "store": ({"model": MODEL, "input": "x", "store": True}, "store"),
    "background": ({"model": MODEL, "input": "x", "background": True}, "background"),
    "include": ({"model": MODEL, "input": "x", "include": ["a"]}, "include"),
    "metadata": ({"model": MODEL, "input": "x", "metadata": {"k": "v"}}, "metadata"),
    "service_tier": (
        {"model": MODEL, "input": "x", "service_tier": "flex"},
        "service_tier",
    ),
    "hosted_web_search_tool": (
        {"model": MODEL, "input": "x", "tools": [{"type": "web_search_preview"}]},
        "hosted tools",
    ),
    "hosted_mcp_tool": (
        {
            "model": MODEL,
            "input": "x",
            "tools": [{"type": "mcp", "server_label": "m", "server_url": "https://m"}],
        },
        "hosted tools",
    ),
    "reasoning_summary": (
        {
            "model": MODEL,
            "input": "x",
            "reasoning": {"effort": "high", "summary": "detailed"},
        },
        "reasoning.summary",
    ),
    "reasoning_item": (
        {
            "model": MODEL,
            "input": [{"type": "reasoning", "id": "rs_1", "summary": [{"text": "t"}]}],
        },
        "reasoning input items",
    ),
    "item_reference": (
        {"model": MODEL, "input": [{"type": "item_reference", "id": "msg_1"}]},
        "item_reference",
    ),
    "input_file": (
        {
            "model": MODEL,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_file", "file_id": "f1"}],
                }
            ],
        },
        "input_file",
    ),
    "image_by_file_id": (
        {
            "model": MODEL,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_image", "file_id": "f1"}],
                }
            ],
        },
        "input_image by file_id",
    ),
    "function_call_output_without_call": (
        {
            "model": MODEL,
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "orphan",
                    "output": "r",
                }
            ],
        },
        "without a matching function_call",
    ),
    "tool_extra_defer_loading": (
        {
            "model": MODEL,
            "input": "x",
            "tools": [
                {
                    "type": "function",
                    "name": "f",
                    "parameters": {"type": "object"},
                    "defer_loading": True,
                }
            ],
        },
        "defer_loading",
    ),
    # A mid-conversation message item whose role is system/developer: v1
    # FORWARDS the role verbatim (transformation.py:995 ``role or "user"``),
    # so the instruction reaches the model as a system turn. The chat IR's
    # Message.role is Literal[user, assistant] with no home for it, so v2 must
    # FAIL CLOSED (the seam serves it on v1) rather than silently demote it to
    # a user turn -- a #30138-class semantic mutation (critic-inbound BLOCKER-1).
    "message_role_system_fails_closed": (
        {
            "model": MODEL,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": "be terse"}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
            ],
        },
        "role 'system'",
    ),
    "message_role_developer_fails_closed": (
        {
            "model": MODEL,
            "input": [
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "be terse"}],
                }
            ],
        },
        "role 'developer'",
    ),
}


@pytest.mark.parametrize("name", sorted(_FALLBACK))
def test_unported_shapes_fail_closed(name: str) -> None:
    request, reason_substring = _FALLBACK[name]
    parsed = parse_request(copy.deepcopy(request))
    assert parsed.is_error(), f"{name}: should fall back, but parse succeeded"
    assert (
        reason_substring in parsed.error.summary
    ), f"{name}: error {parsed.error.summary!r} does not name {reason_substring!r}"


@pytest.mark.parametrize("role", ["system", "developer"])
def test_v1_forwards_a_message_item_role_so_the_fallback_is_real(role: str) -> None:
    """The fail-closed rows above only matter if v1 actually SERVES the role; if
    v1 dropped it too the divergence would be invented. Probe v1 in-process at
    HEAD: it forwards the role verbatim into the chat request (no demotion to
    user), which is the byte v2 would silently lose if it collapsed the role."""
    item = {"role": role, "content": [{"type": "input_text", "text": "be terse"}]}
    message = V1._transform_responses_api_input_item_to_chat_completion_message(
        input_item=item
    )
    assert message == [
        {"role": role, "content": [{"type": "text", "text": "be terse"}]}
    ]
