"""Differential parity for the /v1/messages forward path.

The inbound parser maps an Anthropic Messages request onto the chat IR; the
already-differential-green anthropic provider serializer is its inverse, so a
round-trip ``anthropic-in -> IR -> anthropic-out`` over a corpus authored in
the serializer's canonical form (list-form content, ``type: "custom"`` tools,
tools present whenever tool history is) must reproduce the input byte-for-byte
under canonical JSON. A mutation in the parser (a dropped field, a mis-renamed
``stop_sequences``, a wrong ``tool_choice`` arm, a lost ``cache_control``,
a swapped block order) breaks the round-trip; that is the mutation-kill bar.

String content / string system are tested separately (they parse to the same
IR as their list form, which the serializer canonicalizes to list output).

The fail-closed rows assert every shape the chat IR cannot carry returns a
typed ``unsupported`` so dispatch falls back to v1, instead of silently
dropping it (researcher-6 §1.4/§1.5).
"""

import copy
import json

import pytest

from litellm.translation.inbound.anthropic_messages import parse_request
from litellm.translation.providers.anthropic import serialize_request

from .conftest import build_real_deps

MODEL = "claude-sonnet-4-5"


def _canonical(body: object) -> str:
    return json.dumps(body, sort_keys=True, default=str)


# Authored in the anthropic serializer's canonical output form so the
# round-trip is an identity. The serializer is the proven inverse, so equality
# proves the parser captured every field faithfully (researcher-6 §1.4 SERVE).
_SERVE: dict[str, dict] = {
    "text": {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    },
    "system_block_list_with_cache": {
        "model": MODEL,
        "max_tokens": 64,
        "system": [
            {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "b"},
        ],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    },
    "image_base64_and_text_with_cache": {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "look",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "AAAA",
                        },
                    },
                ],
            }
        ],
    },
    "image_url": {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "url", "url": "https://x/y.png"},
                    }
                ],
            }
        ],
    },
    "assistant_thinking_text_tool_use": {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                    {"type": "text", "text": "calling"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get",
                        "input": {"q": 1},
                    },
                ],
            },
        ],
        "tools": [
            {"name": "get", "input_schema": {"type": "object"}, "type": "custom"}
        ],
    },
    "redacted_thinking": {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "redacted_thinking", "data": "ENC"},
                    {"type": "text", "text": "ok"},
                ],
            },
        ],
    },
    "tool_result_string_with_tools": {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}
                ],
            }
        ],
        "tools": [
            {"name": "get", "input_schema": {"type": "object"}, "type": "custom"}
        ],
    },
    "tool_result_parts_multi_with_tools": {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [
                            {"type": "text", "text": "one"},
                            {"type": "text", "text": "two"},
                        ],
                    }
                ],
            }
        ],
        "tools": [
            {"name": "get", "input_schema": {"type": "object"}, "type": "custom"}
        ],
    },
    "tools_and_tool_choice_tool": {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "weather"}]}
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": "weather",
                "input_schema": {"type": "object", "properties": {}},
                "type": "custom",
            }
        ],
        "tool_choice": {"type": "tool", "name": "get_weather"},
    },
    "tool_choice_any": {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}],
        "tools": [{"name": "f", "input_schema": {"type": "object"}, "type": "custom"}],
        "tool_choice": {"type": "any"},
    },
    "thinking_enabled_budget": {
        "model": MODEL,
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 2000},
        "messages": [{"role": "user", "content": [{"type": "text", "text": "think"}]}],
    },
    "sampling_and_stop_and_user": {
        "model": MODEL,
        "max_tokens": 64,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 40,
        "stop_sequences": ["END", "STOP"],
        "metadata": {"user_id": "u1"},
        "messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}],
    },
}


@pytest.mark.parametrize("name", sorted(_SERVE))
def test_request_round_trips_through_the_anthropic_serializer(name: str) -> None:
    request = copy.deepcopy(_SERVE[name])
    parsed = parse_request(request)
    assert parsed.is_ok(), f"{name}: parse failed: {parsed.error.summary}"
    serialized = serialize_request(parsed.ok, build_real_deps())
    assert serialized.is_ok(), f"{name}: serialize failed: {serialized.error.summary}"
    assert _canonical(serialized.ok) == _canonical(request), name


def test_string_content_parses_like_its_text_block_form() -> None:
    string_form = {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
    }
    block_form = {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
    deps = build_real_deps()
    string_body = serialize_request(parse_request(string_form).ok, deps)
    block_body = serialize_request(parse_request(block_form).ok, deps)
    assert string_body.is_ok() and block_body.is_ok()
    assert _canonical(string_body.ok) == _canonical(block_body.ok)


def test_string_system_parses_like_its_text_block_form() -> None:
    string_form = {
        "model": MODEL,
        "max_tokens": 64,
        "system": "be terse",
        "messages": [{"role": "user", "content": "hi"}],
    }
    block_form = {
        "model": MODEL,
        "max_tokens": 64,
        "system": [{"type": "text", "text": "be terse"}],
        "messages": [{"role": "user", "content": "hi"}],
    }
    deps = build_real_deps()
    string_body = serialize_request(parse_request(string_form).ok, deps)
    block_body = serialize_request(parse_request(block_form).ok, deps)
    assert string_body.is_ok() and block_body.is_ok()
    assert _canonical(string_body.ok) == _canonical(block_body.ok)


# Every shape the chat IR cannot carry must FAIL CLOSED: parse returns a typed
# error so dispatch falls back to v1 (researcher-6 §1.4/§1.5). The substring
# pins what each error names (an extra_forbidden field reads "not yet
# supported by translation v2: <field>"; the semantic rejections name the
# v1 path). A mutation that silently serves any of these breaks the assertion.
_FALLBACK: dict[str, tuple[dict, str]] = {
    "missing_max_tokens": (
        {"model": MODEL, "messages": [{"role": "user", "content": "x"}]},
        "max_tokens",
    ),
    "document_block": (
        {
            "model": MODEL,
            "max_tokens": 64,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "JVBER",
                            },
                        }
                    ],
                }
            ],
        },
        "not yet supported",
    ),
    "non_text_system_block": (
        {
            "model": MODEL,
            "max_tokens": 64,
            "system": [{"type": "image", "text": None}],
            "messages": [{"role": "user", "content": "x"}],
        },
        "non-text system block",
    ),
    "non_text_tool_result_part": (
        {
            "model": MODEL,
            "max_tokens": 64,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "AAAA",
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
        },
        "non-text tool_result",
    ),
    "image_file_source": (
        {
            "model": MODEL,
            "max_tokens": 64,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "file", "file_id": "f1"}}
                    ],
                }
            ],
        },
        "image source other than base64/url",
    ),
    "thinking_summary": (
        {
            "model": MODEL,
            "max_tokens": 64,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 2000,
                "summary": "detailed",
            },
            "messages": [{"role": "user", "content": "x"}],
        },
        "thinking.summary",
    ),
    "output_format": (
        {
            "model": MODEL,
            "max_tokens": 64,
            "output_format": {"type": "json_schema", "schema": {"type": "object"}},
            "messages": [{"role": "user", "content": "x"}],
        },
        "output_format",
    ),
    "output_config": (
        {
            "model": MODEL,
            "max_tokens": 64,
            "output_config": {"effort": "high"},
            "messages": [{"role": "user", "content": "x"}],
        },
        "output_config",
    ),
    "mcp_servers": (
        {
            "model": MODEL,
            "max_tokens": 64,
            "mcp_servers": [{"type": "url", "url": "https://m", "name": "m"}],
            "messages": [{"role": "user", "content": "x"}],
        },
        "mcp_servers",
    ),
    "container": (
        {
            "model": MODEL,
            "max_tokens": 64,
            "container": {"id": "c1"},
            "messages": [{"role": "user", "content": "x"}],
        },
        "container",
    ),
    "context_management": (
        {
            "model": MODEL,
            "max_tokens": 64,
            "context_management": {"edits": []},
            "messages": [{"role": "user", "content": "x"}],
        },
        "context-management",
    ),
    "hosted_web_search_tool": (
        {
            "model": MODEL,
            "max_tokens": 64,
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "messages": [{"role": "user", "content": "x"}],
        },
        "tools.0.type",
    ),
    "metadata_extra_key": (
        {
            "model": MODEL,
            "max_tokens": 64,
            "metadata": {"user_id": "u1", "trace_id": "t1"},
            "messages": [{"role": "user", "content": "x"}],
        },
        "not yet supported",
    ),
    "tool_use_caller": (
        {
            "model": MODEL,
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get",
                            "input": {},
                            "caller": {
                                "type": "code_execution_20250825",
                                "tool_id": "t",
                            },
                        }
                    ],
                },
            ],
        },
        "caller",
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
