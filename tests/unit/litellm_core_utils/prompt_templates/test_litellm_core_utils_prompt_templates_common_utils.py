import base64
import copy
import json
import os
from collections.abc import Mapping
from typing import Final, cast

import pytest

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    ENCRYPTED_REASONING_SIGNATURE_PREFIX,
    TOOL_RESULT_IMAGE_BOUNDARY,
    TOOL_RESULT_IMAGE_PLACEHOLDER,
    TOOL_RESULT_MEDIA_BOUNDARY,
    TOOL_RESULT_MEDIA_PLACEHOLDER,
    add_system_prompt_to_messages,
    cache_control_as_prompt_cache_breakpoint,
    converse_redacted_content,
    encrypted_content_from_signature,
    encrypted_reasoning_signature,
    get_file_ids_from_messages,
    get_format_from_file_id,
    get_semantic_cache_prompt_from_messages,
    get_str_from_messages,
    handle_any_messages_to_chat_completion_str_messages_conversion,
    handle_messages_with_content_list_to_str_conversion,
    hoist_images_from_tool_messages,
    is_encrypted_reasoning_block,
    merge_consecutive_system_messages,
    native_redacted_thinking_data,
    parse_tool_call_arguments,
    prompt_cache_breakpoint_for_wire,
    responses_reasoning_items_from_thinking_blocks,
    salvage_concatenated_tool_arguments,
    split_concatenated_json_objects,
    split_embedding_media_for_tpm,
    split_media_for_tpm,
    strip_encrypted_reasoning_from_messages,
    strip_litellm_response_only_fields,
    system_messages_first,
    update_messages_with_model_file_ids,
    with_prompt_cache_breakpoint,
)
from litellm.types.llms.openai import ChatCompletionAudioObject, ChatCompletionFileObject, ChatCompletionImageObject

_ARTIFACT_FIELD_PATTERN: Final = r'^(?!__.*__$)[^\p{Cc}\p{Cf}\p{Zl}\p{Zp}"\\./[\]]{1,200}$'
_ARTIFACT_DATA_ID_PATTERN: Final = r"^(?!\.\.?(?:\/|$))[A-Za-z0-9_\-.~:@+]{1,200}$"


def test_get_format_from_file_id():
    unified_file_id = "litellm_proxy:application/pdf;unified_id,cbbe3534-8bf8-4386-af00-f5f6b7e370bf"

    format = get_format_from_file_id(unified_file_id)

    assert format == "application/pdf"


def test_update_messages_with_model_file_ids():
    file_id = "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9wZGY7dW5pZmllZF9pZCxmYzdmMmVhNS0wZjUwLTQ5ZjYtODljMS03ZTZhNTRiMTIxMzg"
    model_id = "my_model_id"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this recording?"},
                {
                    "type": "file",
                    "file": {
                        "file_id": file_id,
                    },
                },
            ],
        },
    ]

    model_file_id_mapping = {file_id: {"my_model_id": "provider_file_id"}}

    updated_messages = update_messages_with_model_file_ids(messages, model_id, model_file_id_mapping)

    assert updated_messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this recording?"},
                {
                    "type": "file",
                    "file": {
                        "file_id": "provider_file_id",
                        "format": "application/pdf",
                    },
                },
            ],
        }
    ]


def test_handle_any_messages_to_chat_completion_str_messages_conversion_list():
    # Test with list of messages
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    result = handle_any_messages_to_chat_completion_str_messages_conversion(messages)
    assert len(result) == 2
    assert result[0] == messages[0]
    assert result[1] == messages[1]


def test_handle_any_messages_to_chat_completion_str_messages_conversion_dict():
    # Test with single dictionary message
    message = {"role": "user", "content": "Hello"}
    result = handle_any_messages_to_chat_completion_str_messages_conversion(message)
    assert len(result) == 1
    assert result[0]["input"] == json.dumps(message)


def test_handle_any_messages_to_chat_completion_str_messages_conversion_str():
    # Test with string message
    message = "Hello"
    result = handle_any_messages_to_chat_completion_str_messages_conversion(message)
    assert len(result) == 1
    assert result[0]["input"] == message


def test_handle_any_messages_to_chat_completion_str_messages_conversion_other():
    # Test with non-string/dict/list type
    message = 123
    result = handle_any_messages_to_chat_completion_str_messages_conversion(message)
    assert len(result) == 1
    assert result[0]["input"] == "123"


def test_handle_any_messages_to_chat_completion_str_messages_conversion_complex():
    # Test with complex nested structure
    message = {
        "role": "user",
        "content": {"text": "Hello", "metadata": {"timestamp": "2024-01-01"}},
    }
    result = handle_any_messages_to_chat_completion_str_messages_conversion(message)
    assert len(result) == 1
    assert result[0]["input"] == json.dumps(message)


def test_add_system_prompt_to_messages_prepend():
    """Adds system prompt at beginning when no system message exists."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    result = add_system_prompt_to_messages(messages, "You are a helpful assistant.")
    assert result == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]


def test_add_system_prompt_to_messages_empty_prompt_unchanged():
    """Returns messages unchanged when system_prompt is empty."""
    messages = [{"role": "user", "content": "Hello"}]
    assert add_system_prompt_to_messages(messages, "") == messages
    assert add_system_prompt_to_messages(messages, None) == messages


def test_add_system_prompt_to_messages_merge_with_first_system():
    """Merges new prompt into first system message when merge_with_first_system=True."""
    messages = [
        {"role": "system", "content": "Existing system prompt."},
        {"role": "user", "content": "Hello"},
    ]
    result = add_system_prompt_to_messages(messages, "You are helpful.", merge_with_first_system=True)
    assert result == [
        {"role": "system", "content": "You are helpful.\n\nExisting system prompt."},
        {"role": "user", "content": "Hello"},
    ]


def test_add_system_prompt_to_messages_merge_with_first_system_adds_new_when_no_system():
    """When merge_with_first_system=True but no system message, adds new one at start."""
    messages = [{"role": "user", "content": "Hello"}]
    result = add_system_prompt_to_messages(messages, "You are helpful.", merge_with_first_system=True)
    assert result == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]


def test_add_system_prompt_to_messages_empty_list():
    """Adds system prompt to empty messages list."""
    result = add_system_prompt_to_messages([], "You are helpful.")
    assert result == [{"role": "system", "content": "You are helpful."}]


def test_convert_prefix_message_to_non_prefix_messages():
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        convert_prefix_message_to_non_prefix_messages,
    )

    messages = [
        {"role": "assistant", "content": "value", "prefix": True},
    ]
    result = convert_prefix_message_to_non_prefix_messages(messages)
    assert result == [
        {
            "role": "system",
            "content": "You are a helpful assistant. You are given a message and you need to respond to it. You are also given a generated content. You need to respond to the message in continuation of the generated content. Do not repeat the same content. Your response should be in continuation of this text: ",
        },
        {"role": "assistant", "content": "value"},
    ]


# ── split_concatenated_json_objects tests ──


def test_split_concatenated_json_single_object():
    """A single valid JSON object is returned as a one-element list."""
    result = split_concatenated_json_objects('{"location": "Boston"}')
    assert result == [{"location": "Boston"}]


def test_split_concatenated_json_multiple_objects():
    """
    Multiple JSON objects concatenated without separators are split correctly.
    This is the exact pattern from issue #20543 where Bedrock Claude Sonnet 4.5
    returns concatenated JSON in a single tool call arguments string.
    """
    raw = (
        '{"command": ["curl", "-i", "http://localhost:9009"]}'
        '{"command": ["curl", "-i", "http://localhost:9009/robots.txt"]}'
        '{"command": ["curl", "-i", "http://localhost:9009/sitemap.xml"]}'
    )
    result = split_concatenated_json_objects(raw)
    assert len(result) == 3
    assert result[0] == {"command": ["curl", "-i", "http://localhost:9009"]}
    assert result[1] == {"command": ["curl", "-i", "http://localhost:9009/robots.txt"]}
    assert result[2] == {"command": ["curl", "-i", "http://localhost:9009/sitemap.xml"]}


def test_split_concatenated_json_with_whitespace():
    """Objects separated by whitespace are handled correctly."""
    raw = '{"a": 1}  {"b": 2}\n{"c": 3}'
    result = split_concatenated_json_objects(raw)
    assert len(result) == 3
    assert result[0] == {"a": 1}
    assert result[1] == {"b": 2}
    assert result[2] == {"c": 3}


def test_split_concatenated_json_empty_string():
    """Empty or whitespace-only strings return an empty list."""
    assert split_concatenated_json_objects("") == []
    assert split_concatenated_json_objects("   ") == []


def test_split_concatenated_json_non_dict_value():
    """Non-dict JSON values (e.g. arrays, strings) are replaced with {}."""
    result = split_concatenated_json_objects("[1, 2, 3]")
    assert result == [{}]


def test_split_concatenated_json_wholly_invalid_returns_empty():
    """
    Wholly unparseable JSON degrades to an empty list instead of raising.

    Regression for https://github.com/BerriAI/litellm/issues/18667: a raise
    here propagated out of `_convert_to_bedrock_tool_call_invoke` and turned
    every replayed conversation into a 500.
    """
    assert split_concatenated_json_objects("not json at all") == []


def test_split_concatenated_json_malformed_object_returns_empty():
    """
    A single malformed object (missing comma between keys) degrades to an
    empty list rather than raising `Expecting ',' delimiter`.
    """
    assert split_concatenated_json_objects('{"location": "Boston" "unit": "celsius"}') == []


def test_split_concatenated_json_salvages_prefix_before_truncated_tail():
    """
    Complete objects parsed before an unparseable/truncated tail are kept;
    only the bad tail is discarded.
    """
    result = split_concatenated_json_objects('{"a": 1}{"b": 2}{"c":')
    assert result == [{"a": 1}, {"b": 2}]


def test_parse_tool_call_arguments_rejects_concatenated_json() -> None:
    with pytest.raises(ValueError, match="Failed to parse tool call arguments"):
        parse_tool_call_arguments('{"a":1}{"b":2}')


def _distinct_json_objects(count: int) -> str:
    return "".join(json.dumps({"n": index}, separators=(",", ":")) for index in range(count))


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ('{"a":1}{"b":2}', ({"a": 1}, {"b": 2})),
        ('{"a":1}{"a":1}{"a":1}', ({"a": 1},)),
        ('{"a":1}{"a":1}{"b":2}', ({"a": 1}, {"a": 1}, {"b": 2})),
        (_distinct_json_objects(8), tuple({"n": index} for index in range(8))),
        (_distinct_json_objects(9), ()),
        (_distinct_json_objects(9) + " junk", ()),
        ('{"a":1}' * 7 + '{"b":2}', tuple({"a": 1} for _ in range(7)) + ({"b": 2},)),
        ('{"a":1}' * 8 + '{"b":2}', ()),
        ('{"a":1}' * 5000, ({"a": 1},)),
        ('{"a":1}' * 20, ({"a": 1},)),
        ('{"a":1}{"b":', ()),
        ('0{"x":1}', ()),
        ('{"x":1}0', ()),
        ('[1]{"x":1}', ()),
        ('{"a":1}{"b":2}}', ()),
        ('{"a":1} junk', ()),
    ),
)
def test_salvage_concatenated_tool_arguments(raw: str, expected: tuple[dict[str, object], ...]) -> None:
    assert salvage_concatenated_tool_arguments(raw) == expected


# ---------------------------------------------------------------------------
# Regression tests for non-OpenAI file content blocks.
#
# `type: "file"` is a public content-block discriminator. Several producers
# (LangChain v1, provider-native shapes, custom user code) emit blocks with
# `type: "file"` but without the OpenAI Chat Completions `file` sub-dict.
# The discovery helpers below are used unconditionally inside
# `AnthropicConfig.validate_environment`, so any crash there surfaces as a
# `500 InternalServerError` before the request is even dispatched.
# ---------------------------------------------------------------------------


def test_get_file_ids_from_messages_skips_langchain_v1_file_block():
    """A LangChain v1 standardized file block must not crash file-id discovery."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "summarise this PDF"},
                # LangChain v1 shape produced by `_normalize_messages`.
                # No `file` sub-dict: the discriminator is `type: "file"` but
                # the payload lives on `base64`/`mime_type` siblings.
                {
                    "type": "file",
                    "id": "lc_1",
                    "base64": "JVBERi0xLjQK",
                    "mime_type": "application/pdf",
                    "extras": {"file_format": "application/pdf"},
                },
            ],
        }
    ]

    assert get_file_ids_from_messages(messages) == []


def test_get_file_ids_from_messages_still_extracts_from_openai_shape():
    """Well-formed OpenAI file blocks still yield their file_id."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "file", "file": {"file_id": "file-abc"}},
            ],
        }
    ]

    assert get_file_ids_from_messages(messages) == ["file-abc"]


def test_get_file_ids_from_messages_mixed_shapes():
    """Mixed OpenAI and non-OpenAI file blocks: extract from the former,
    ignore the latter."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "file", "file": {"file_id": "file-keep"}},
                {
                    "type": "file",
                    "id": "lc_2",
                    "base64": "AAA",
                    "mime_type": "application/pdf",
                },
            ],
        }
    ]

    assert get_file_ids_from_messages(messages) == ["file-keep"]


def test_get_file_ids_from_messages_file_field_not_dict():
    """`file` set to a non-dict value (e.g. stringified payload) must not crash."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "file", "file": "unexpectedly-a-string"},
            ],
        }
    ]

    assert get_file_ids_from_messages(messages) == []


def test_get_file_ids_from_messages_skips_bare_string_content_items():
    messages = [
        {
            "role": "user",
            "content": [
                "what type of file is this?",
                {"type": "file", "file": {"file_id": "file-abc"}},
            ],
        }
    ]

    assert get_file_ids_from_messages(messages) == ["file-abc"]


def test_update_messages_with_model_file_ids_skips_non_openai_file_blocks():
    """`update_messages_with_model_file_ids` is also called on user content
    before provider dispatch. It must tolerate non-OpenAI file blocks the same
    way."""
    langchain_v1_block = {
        "type": "file",
        "id": "lc_3",
        "base64": "AAA",
        "mime_type": "application/pdf",
    }
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                langchain_v1_block,
            ],
        }
    ]

    updated = update_messages_with_model_file_ids(messages, "model-1", {})

    # Messages pass through unchanged when there is no `file` sub-dict to remap.
    assert updated == messages


# Reusable fixture (decodes to: litellm_proxy:application/pdf;unified_id,...;
# target_model_names,gpt-4o;llm_output_file_id,file-ECBPW7ML9g7XHdwGgUPZaM;
# llm_output_file_model_id,...)
UNIFIED_FILE_ID_B64 = (
    "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9wZGY7dW5pZmllZF9pZCw2YzBiNTg5MC04OTE0"
    "LTQ4ZTAtYjhmNC0wYWU1ZWQzYzE0YTU7dGFyZ2V0X21vZGVsX25hbWVzLGdwdC00bztsbG1f"
    "b3V0cHV0X2ZpbGVfaWQsZmlsZS1FQ0JQVzdNTDlnN1hIZHdHZ1VQWmFNO2xsbV9vdXRwdXRf"
    "ZmlsZV9tb2RlbF9pZCxlMjY0NTNmOWU3NmU3OTkzNjgwZDAwNjhkOThjMWY0Y2MyMDViYmFk"
    "MDk2N2EzM2M2NjQ4OTM1NjhjYTc0M2My"
)


def test_update_messages_with_model_file_ids_decodes_unified_id_when_mapping_empty():
    """When the mapping is empty (e.g. multi-replica cache miss), the function
    must decode the base64-encoded unified file id and substitute the embedded
    llm_output_file_id — mirroring the Responses-API sibling."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this recording?"},
                {
                    "type": "file",
                    "file": {
                        "file_id": UNIFIED_FILE_ID_B64,
                        "format": "audio/wav",
                    },
                },
            ],
        }
    ]

    updated = update_messages_with_model_file_ids(messages, "any-model-id", {})

    assert updated[0]["content"][1]["file"]["file_id"] == "file-ECBPW7ML9g7XHdwGgUPZaM"
    # Customer-supplied format is preserved (this is the field whose absence
    # the misleading error message used to complain about).
    assert updated[0]["content"][1]["file"]["format"] == "audio/wav"


def test_update_messages_with_model_file_ids_mapping_takes_precedence_over_decode():
    """When both mapping and decode would resolve, the mapping must win
    (preserves per-deployment routing precision)."""
    mapping = {UNIFIED_FILE_ID_B64: {"model-A": "mapped-provider-file-id"}}
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "file_id": UNIFIED_FILE_ID_B64,
                        "format": "application/pdf",
                    },
                },
            ],
        }
    ]

    updated = update_messages_with_model_file_ids(messages, "model-A", mapping)

    assert updated[0]["content"][0]["file"]["file_id"] == "mapped-provider-file-id"


def test_update_messages_with_model_file_ids_non_unified_passes_through():
    """A raw provider id (e.g. gs:// URI or a random string) must be left
    untouched when the mapping doesn't resolve it. The decode fallback must
    not corrupt non-unified ids."""
    raw_id = "gs://my-bucket/uploads/abc-123.wav"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "file", "file": {"file_id": raw_id, "format": "audio/wav"}},
            ],
        }
    ]

    updated = update_messages_with_model_file_ids(messages, "model-A", {})

    assert updated[0]["content"][0]["file"]["file_id"] == raw_id


def test_update_messages_with_model_file_ids_mapping_miss_falls_back_to_decode():
    """A mapping that exists but doesn't contain this file_id should still
    trigger the decode fallback — covers the case where the hook resolved
    *some* ids but not this one."""
    other_id = "some-other-file-id"
    mapping = {other_id: {"model-A": "other-provider-id"}}
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {"file_id": UNIFIED_FILE_ID_B64, "format": "audio/wav"},
                },
            ],
        }
    ]

    updated = update_messages_with_model_file_ids(messages, "model-A", mapping)

    assert updated[0]["content"][0]["file"]["file_id"] == "file-ECBPW7ML9g7XHdwGgUPZaM"


def test_update_messages_with_model_file_ids_tolerates_non_dict_content_items():
    """Content list items aren't always dicts. text_completion forwards
    token-ids (list of ints, or list of list of ints for batch) through
    this path. The function must skip non-dict items instead of indexing
    into them."""
    messages_token_ids = [{"role": "user", "content": [15496, 995]}]
    messages_token_ids_batch = [{"role": "user", "content": [[15496, 995], [9906, 0]]}]

    # Both should pass through unchanged without raising.
    assert update_messages_with_model_file_ids(messages_token_ids, "model-A", {}) == messages_token_ids
    assert update_messages_with_model_file_ids(messages_token_ids_batch, "model-A", {}) == messages_token_ids_batch


class TestExtractFileDataBareStr:
    """``extract_file_data`` used to accept bare ``str`` values and ``open()``
    them server-side. When the helper runs inside a proxy request handler the
    value is attacker-controlled, so the open() call was a textbook arbitrary
    local file read. Lock the new contract: bare ``str`` is rejected with a
    clear migration message; ``pathlib.Path`` is still accepted for SDK
    ergonomics because it's a Python-level type that HTTP form values can't
    fabricate."""

    def test_rejects_bare_str(self):
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            extract_file_data,
        )

        with pytest.raises(ValueError, match="does not accept bare str inputs"):
            extract_file_data("/etc/passwd")

    def test_accepts_pathlib_path(self):
        import tempfile
        from pathlib import Path

        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            extract_file_data,
        )

        content = b"hello"
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(content)
            tmp_path = Path(f.name)

        try:
            extracted = extract_file_data(tmp_path)
            assert extracted.get("content") == content
        finally:
            os.unlink(str(tmp_path))

    def test_accepts_bytes(self):
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            extract_file_data,
        )

        extracted = extract_file_data(b"raw bytes content")
        assert extracted.get("content") == b"raw bytes content"

    def test_accepts_tuple(self):
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            extract_file_data,
        )

        extracted = extract_file_data(("foo.txt", b"raw bytes content"))
        assert extracted.get("filename") == "foo.txt"
        assert extracted.get("content") == b"raw bytes content"


class TestUnpackLegacyDefs:
    """Cover the public ``unpack_legacy_defs`` helper directly so the no-op
    branches (non-dict input, schema with no legacy/OpenAPI defs) are exercised
    without needing a provider-specific entry point.
    """

    @pytest.mark.parametrize(
        "value",
        [None, [], "string-not-a-dict", 42, 1.5, True, set(), tuple()],
    )
    def test_non_dict_returns_unchanged_no_op(self, value):
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            unpack_legacy_defs,
        )

        # Should never raise; returns the input unchanged.
        assert unpack_legacy_defs(value) is value
        assert unpack_legacy_defs(value, copy=True) is value

    def test_dict_without_legacy_defs_is_no_op(self):
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            unpack_legacy_defs,
        )

        schema = {
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/A"}},
            "$defs": {"A": {"type": "string"}},
        }
        snapshot = json.loads(json.dumps(schema))

        # No `definitions` and no `components.schemas` -> early return, no work.
        out = unpack_legacy_defs(schema)
        assert out is schema
        assert schema == snapshot, "schema mutated despite no legacy defs"

    def test_components_with_no_schemas_block_is_no_op(self):
        """``components`` without a ``schemas`` sub-key must not be popped."""
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            unpack_legacy_defs,
        )

        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "components": {"securitySchemes": {"foo": "bar"}},
        }
        snapshot = json.loads(json.dumps(schema))

        unpack_legacy_defs(schema)
        assert schema == snapshot, "components without schemas was incorrectly popped"

    def test_legitimate_schema_within_budget_succeeds(self):
        """A flat schema with many distinct ``$ref``s into small targets must
        inline cleanly under the default budget -- the budget rejects bombs,
        not legitimately-shaped schemas.
        """
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            unpack_legacy_defs,
        )

        n = 200
        schema = {
            "type": "object",
            "properties": {f"f{i}": {"$ref": f"#/definitions/T{i}"} for i in range(n)},
            "definitions": {f"T{i}": {"type": "string"} for i in range(n)},
        }

        out = unpack_legacy_defs(schema)
        assert "definitions" not in out
        for i in range(n):
            assert out["properties"][f"f{i}"] == {"type": "string"}

    # Schema-bomb amplification vectors. ``max_inlined_bytes`` is the universal
    # measure of expansion: every other dimension (ref count, node count,
    # scalar size) reduces to bytes-on-the-wire, so a single byte budget
    # closes all three vectors at once.

    def test_rejects_fan_out_bomb(self):
        """Each level multiplies refs (cycle detection only stops re-entry
        along the *same* path). Must trip the byte budget."""
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            unpack_legacy_defs,
        )

        depth, fanout = 12, 2  # 2**12 = 4096 leaves
        definitions = {
            f"L{i}": {
                "type": "object",
                "properties": {f"x{j}": {"$ref": f"#/definitions/L{i + 1}"} for j in range(fanout)},
            }
            for i in range(depth)
        }
        definitions[f"L{depth}"] = {"type": "string"}
        schema = {
            "type": "object",
            "properties": {"root": {"$ref": "#/definitions/L0"}},
            "definitions": definitions,
        }

        with pytest.raises(ValueError, match="byte budget"):
            unpack_legacy_defs(schema, max_inlined_bytes=100_000)

    def test_rejects_target_amplification_bomb(self):
        """Few refs each deep-copying one large target -- bounded total
        expanded bytes catches it even though ref count is small."""
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            unpack_legacy_defs,
        )

        big = {
            "type": "object",
            "properties": {f"p{i}": {"type": "string"} for i in range(100)},
        }
        schema = {
            "type": "object",
            "properties": {f"r{i}": {"$ref": "#/definitions/Big"} for i in range(50)},
            "definitions": {"Big": big},
        }

        with pytest.raises(ValueError, match="byte budget"):
            unpack_legacy_defs(schema, max_inlined_bytes=10_000)

    def test_rejects_scalar_byte_amplification_bomb(self):
        """Many ``$ref``s to a target containing one large scalar (e.g. a
        long ``description``, ``const`` value, or ``enum`` entry). A
        node-counter would treat this as 1 node per resolution and miss it;
        a byte budget catches the actual wire-size amplification.
        """
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            unpack_legacy_defs,
        )

        big_description = "x" * 100_000  # 100KB string
        schema = {
            "type": "object",
            "properties": {f"r{i}": {"$ref": "#/definitions/Big"} for i in range(50)},
            "definitions": {
                "Big": {"type": "string", "description": big_description},
            },
        }
        # 50 refs * ~100KB string == ~5MB cumulative; 1MB budget trips.
        with pytest.raises(ValueError, match="byte budget"):
            unpack_legacy_defs(schema, max_inlined_bytes=1_000_000)

    def test_budget_does_not_trip_for_legitimate_large_schema(self):
        """An OpenAPI-derived tool with ~50 small targets must inline cleanly
        under the default ``max_inlined_bytes`` budget."""
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            unpack_legacy_defs,
        )

        schema = {
            "type": "object",
            "properties": {f"r{i}": {"$ref": f"#/components/schemas/T{i}"} for i in range(50)},
            "components": {
                "schemas": {
                    f"T{i}": {
                        "type": "object",
                        "properties": {f"p{j}": {"type": "string"} for j in range(5)},
                    }
                    for i in range(50)
                }
            },
        }

        out = unpack_legacy_defs(schema)
        assert "components" not in out
        assert out["properties"]["r0"]["properties"]["p0"] == {"type": "string"}


class TestTextCompletionPromptToMessages:
    """`/v1/completions` prompt wrapping, shared by the real-time and batch paths."""

    def test_string_prompt_becomes_single_user_message(self):
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            text_completion_prompt_to_messages,
        )

        assert text_completion_prompt_to_messages("summarize this") == ({"role": "user", "content": "summarize this"},)

    def test_list_of_strings_becomes_one_message_each(self):
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            text_completion_prompt_to_messages,
        )

        assert text_completion_prompt_to_messages(["first", "second"]) == (
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        )

    @pytest.mark.parametrize(
        "prompt",
        [
            [1, 2, 3],
            [[1, 2], [3, 4]],
            ["ok", 7],
            [],
            "",
            None,
            {"role": "user"},
        ],
    )
    def test_unsupported_prompt_shapes_raise(self, prompt):
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            text_completion_prompt_to_messages,
        )

        with pytest.raises(ValueError, match="non-empty string or a non-empty list of strings"):
            text_completion_prompt_to_messages(prompt)


DATA_URI_PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
BOUNDARY_PART = {"type": "text", "text": TOOL_RESULT_IMAGE_BOUNDARY}


def _tool_msg(content, tool_call_id="call_1"):
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _assistant_tool_call_msg(*tool_call_ids):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": tid, "type": "function", "function": {"name": "read_image", "arguments": "{}"}}
            for tid in tool_call_ids
        ],
    }


def test_hoist_images_from_tool_messages_bare_data_uri_string_passes_through():
    messages = [
        {"role": "user", "content": "read the image"},
        _assistant_tool_call_msg("call_1"),
        _tool_msg(DATA_URI_PNG),
    ]

    result = hoist_images_from_tool_messages(messages)

    assert result is messages


@pytest.mark.parametrize(
    "part,placeholder,boundary",
    [
        (
            {"type": "image_url", "image_url": {"url": DATA_URI_PNG}},
            TOOL_RESULT_IMAGE_PLACEHOLDER, TOOL_RESULT_IMAGE_BOUNDARY,
        ),
        (
            {"type": "file", "file": {"file_data": "data:application/pdf;base64,cGRm", "filename": "report.pdf"}},
            TOOL_RESULT_MEDIA_PLACEHOLDER, TOOL_RESULT_MEDIA_BOUNDARY,
        ),
        (
            {"type": "input_audio", "input_audio": {"data": "c291bmQ=", "format": "wav"}},
            TOOL_RESULT_MEDIA_PLACEHOLDER, TOOL_RESULT_MEDIA_BOUNDARY,
        ),
    ],
)
def test_hoist_tool_media_preserves_payload_and_call(
    part: ChatCompletionImageObject | ChatCompletionFileObject | ChatCompletionAudioObject,
    placeholder: str, boundary: str,
) -> None:
    messages: Final = [
        _assistant_tool_call_msg("call_1"),
        {**_tool_msg([part]), "cache_control": {"type": "ephemeral"}},
    ]
    original: Final = copy.deepcopy(messages)

    result: Final = hoist_images_from_tool_messages(messages)

    assert result == [
        messages[0], {**messages[1], "content": placeholder},
        {"role": "user", "content": [{"type": "text", "text": boundary}, part]},
    ]
    assert messages == original


def test_hoist_images_from_tool_messages_keeps_text_parts_in_tool_message():
    messages = [
        _assistant_tool_call_msg("call_1"),
        _tool_msg(
            [
                {"type": "text", "text": "screenshot follows"},
                {"type": "image_url", "image_url": {"url": DATA_URI_PNG}},
            ]
        ),
    ]

    result = hoist_images_from_tool_messages(messages)

    assert result[1]["content"] == [{"type": "text", "text": "screenshot follows"}]
    assert result[2]["content"] == [BOUNDARY_PART, {"type": "image_url", "image_url": {"url": DATA_URI_PNG}}]


def test_hoist_images_from_tool_messages_parallel_tool_calls_insert_after_run() -> None:
    image: Final = {"type": "image_url", "image_url": {"url": DATA_URI_PNG}}
    file: Final = {"type": "file", "file": {"file_id": "file_report"}}
    audio: Final = {"type": "input_audio", "input_audio": {"data": "c291bmQ=", "format": "wav"}}
    text: Final = {"type": "text", "text": "Report and recording follow"}
    messages: Final = [
        _assistant_tool_call_msg("call_1", "call_2"),
        _tool_msg([image], tool_call_id="call_1"),
        _tool_msg([text, file, audio], tool_call_id="call_2"),
        {"role": "assistant", "content": "looking"},
    ]

    result: Final = hoist_images_from_tool_messages(messages)

    assert result == [
        messages[0],
        {**messages[1], "content": TOOL_RESULT_IMAGE_PLACEHOLDER},
        {**messages[2], "content": [text]},
        {"role": "user", "content": [{"type": "text", "text": TOOL_RESULT_MEDIA_BOUNDARY}, image, file, audio]},
        messages[3],
    ]


def test_hoist_images_from_tool_messages_no_tool_messages_returns_input_unchanged():
    messages = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": DATA_URI_PNG}}]},
        {"role": "assistant", "content": "a cat"},
    ]

    result = hoist_images_from_tool_messages(messages)

    assert result is messages


def test_hoist_images_from_tool_messages_text_only_tool_message_unchanged():
    messages = [
        _assistant_tool_call_msg("call_1"),
        _tool_msg("plain text result"),
        _tool_msg([{"type": "text", "text": "another"}], tool_call_id="call_2"),
    ]

    result = hoist_images_from_tool_messages(messages)

    assert result is messages


def test_hoist_images_from_tool_messages_does_not_mutate_input():
    tool_message = _tool_msg([{"type": "image_url", "image_url": {"url": DATA_URI_PNG}}])
    messages = [_assistant_tool_call_msg("call_1"), tool_message]

    hoist_images_from_tool_messages(messages)

    assert tool_message["content"] == [{"type": "image_url", "image_url": {"url": DATA_URI_PNG}}]
    assert len(messages) == 2


@pytest.mark.parametrize(
    "sibling_content",
    [None, [{"type": "text", "text": "42 files"}], '{"artifact":"ready"}'],
    ids=["none_content", "text_only_list", "opaque_json"],
)
def test_hoist_images_from_tool_messages_imageless_sibling_in_image_run_unchanged(sibling_content):
    imageless_tool_msg = _tool_msg(sibling_content, tool_call_id="call_2")
    messages = [
        _assistant_tool_call_msg("call_1", "call_2"),
        _tool_msg([{"type": "image_url", "image_url": {"url": DATA_URI_PNG}}]),
        imageless_tool_msg,
    ]

    result = hoist_images_from_tool_messages(messages)

    assert [m["role"] for m in result] == ["assistant", "tool", "tool", "user"]
    assert result[1]["content"] == TOOL_RESULT_IMAGE_PLACEHOLDER
    assert result[2] is imageless_tool_msg
    assert result[3]["content"] == [BOUNDARY_PART, {"type": "image_url", "image_url": {"url": DATA_URI_PNG}}]


def test_hoist_images_from_tool_messages_earlier_tool_run_without_images_unchanged():
    messages = [
        _assistant_tool_call_msg("call_1"),
        _tool_msg("plain text result"),
        _assistant_tool_call_msg("call_2"),
        _tool_msg([{"type": "image_url", "image_url": {"url": DATA_URI_PNG}}], tool_call_id="call_2"),
    ]

    result = hoist_images_from_tool_messages(messages)

    assert [m["role"] for m in result] == ["assistant", "tool", "assistant", "tool", "user"]
    assert result[1]["content"] == "plain text result"
    assert result[3]["content"] == TOOL_RESULT_IMAGE_PLACEHOLDER
    assert result[4]["content"] == [BOUNDARY_PART, {"type": "image_url", "image_url": {"url": DATA_URI_PNG}}]


class TestCustomToolFormatShapeConversion:
    def test_flat_grammar_to_chat_shape(self):
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            convert_custom_tool_format_to_chat_shape,
        )

        assert convert_custom_tool_format_to_chat_shape(
            {"type": "grammar", "definition": "start: patch", "syntax": "lark"}
        ) == {"type": "grammar", "grammar": {"definition": "start: patch", "syntax": "lark"}}

    def test_nested_grammar_to_responses_shape(self):
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            convert_custom_tool_format_to_responses_shape,
        )

        assert convert_custom_tool_format_to_responses_shape(
            {"type": "grammar", "grammar": {"definition": "start: patch", "syntax": "regex"}}
        ) == {"type": "grammar", "definition": "start: patch", "syntax": "regex"}

    def test_both_directions_are_idempotent_and_pass_text_through(self):
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            convert_custom_tool_format_to_chat_shape,
            convert_custom_tool_format_to_responses_shape,
        )

        flat = {"type": "grammar", "definition": "d", "syntax": "lark"}
        nested = {"type": "grammar", "grammar": {"definition": "d", "syntax": "lark"}}
        text = {"type": "text"}
        assert convert_custom_tool_format_to_chat_shape(nested) == nested
        assert convert_custom_tool_format_to_responses_shape(flat) == flat
        assert convert_custom_tool_format_to_chat_shape(text) == text
        assert convert_custom_tool_format_to_responses_shape(text) == text
        assert convert_custom_tool_format_to_chat_shape(convert_custom_tool_format_to_responses_shape(nested)) == nested

    def test_unrecognized_formats_pass_through(self):
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            convert_custom_tool_format_to_chat_shape,
            convert_custom_tool_format_to_responses_shape,
        )

        for weird in ({}, {"type": "grammar"}, {"type": "future_format", "x": 1}):
            assert convert_custom_tool_format_to_chat_shape(dict(weird)) in (weird, {"type": "grammar", "grammar": {}})
            assert convert_custom_tool_format_to_responses_shape(dict(weird)) == weird


# --- x-litellm-model upload-path decoding (litellm #29830) -------------------


def _xlitellm_encoded(raw_id: str, model: str) -> str:
    from litellm.proxy.openai_files_endpoints.common_utils import (
        encode_file_id_with_model,
    )

    return encode_file_id_with_model(raw_id, model)


def test_update_messages_with_model_file_ids_decodes_xlitellm_encoded_id():
    """x-litellm-model upload returns `file-<b64(litellm:<raw>;model,<m>)>`.
    Without decoding, the encoded id leaks to upstream OpenAI and errors as
    'Files [...] were not found'. Decode it back to raw provider id."""
    raw_id = "file-ExTuCawUqxEMjVFK6xwR9B"
    encoded_id = _xlitellm_encoded(raw_id, "gpt-5.1")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize this."},
                {"type": "file", "file": {"file_id": encoded_id}},
            ],
        }
    ]

    updated = update_messages_with_model_file_ids(messages, "model-A", {})

    assert updated[0]["content"][1]["file"]["file_id"] == raw_id


def test_update_responses_input_with_model_file_ids_decodes_xlitellm_encoded_id():
    """Same bug on /v1/responses path. Without decoding the encoded id (>64
    chars), OpenAI rejects with 'string too long. Expected ... maximum length
    64'."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        update_responses_input_with_model_file_ids,
    )

    raw_id = "file-ExTuCawUqxEMjVFK6xwR9B"
    encoded_id = _xlitellm_encoded(raw_id, "gpt-5.1")
    input_items = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Summarize."},
                {"type": "input_file", "file_id": encoded_id},
            ],
        }
    ]

    updated = update_responses_input_with_model_file_ids(input_items)

    assert updated[0]["content"][1]["file_id"] == raw_id



@pytest.mark.parametrize("unusable_file_id", [[1], {"a": 1}])
def test_update_responses_input_with_model_file_ids_keeps_a_non_string_file_id_as_sent(unusable_file_id):
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        update_responses_input_with_model_file_ids,
    )

    encoded_id = _xlitellm_encoded("file-ExTuCawUqxEMjVFK6xwR9B", "gpt-5.1")
    input_items = [
        {
            "role": "user",
            "content": [
                {"type": "input_file", "file_id": encoded_id},
                {"type": "input_file", "file_id": unusable_file_id},
            ],
        }
    ]

    updated = update_responses_input_with_model_file_ids(
        input_items, "model-A", {encoded_id: {"model-A": "file-on-a"}}
    )

    assert [item["file_id"] for item in updated[0]["content"]] == ["file-on-a", unusable_file_id]

def test_update_messages_xlitellm_decode_does_not_override_mapping():
    """If the call-site already resolved a provider id via the mapping, that
    wins. The new decode fallback runs only when no mapping match."""
    raw_id = "file-ExTuCawUqxEMjVFK6xwR9B"
    encoded_id = _xlitellm_encoded(raw_id, "gpt-5.1")
    mapping = {encoded_id: {"model-A": "provider-explicit-id"}}
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "file", "file": {"file_id": encoded_id}},
            ],
        }
    ]

    updated = update_messages_with_model_file_ids(messages, "model-A", mapping)

    assert updated[0]["content"][0]["file"]["file_id"] == "provider-explicit-id"


def test_drop_tool_reference_parts_keeps_text_parts():
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        drop_tool_reference_parts_from_tool_messages,
    )

    messages = [
        _assistant_tool_call_msg("call_1"),
        _tool_msg(
            [
                {"type": "text", "text": "WebFetch tool loaded successfully."},
                {"type": "tool_reference", "tool_name": "WebFetch"},
            ]
        ),
    ]

    result = drop_tool_reference_parts_from_tool_messages(messages)

    assert result[1]["content"] == [{"type": "text", "text": "WebFetch tool loaded successfully."}]
    assert result[1]["tool_call_id"] == "call_1"


def test_drop_tool_reference_parts_reference_only_becomes_empty_text():
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        drop_tool_reference_parts_from_tool_messages,
    )

    messages = [
        _assistant_tool_call_msg("call_1"),
        _tool_msg([{"type": "tool_reference", "tool_name": "WebFetch"}]),
    ]

    result = drop_tool_reference_parts_from_tool_messages(messages)

    assert result[1] == {"role": "tool", "tool_call_id": "call_1", "content": ""}


def test_drop_tool_reference_parts_without_references_passes_through():
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        drop_tool_reference_parts_from_tool_messages,
    )

    messages = [
        _assistant_tool_call_msg("call_1"),
        _tool_msg([{"type": "text", "text": "plain result"}]),
    ]

    assert drop_tool_reference_parts_from_tool_messages(messages) is messages


def test_drop_tool_reference_parts_leaves_non_tool_messages_alone():
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        drop_tool_reference_parts_from_tool_messages,
    )

    user_message = {"role": "user", "content": [{"type": "tool_reference", "tool_name": "WebFetch"}]}
    messages = [
        user_message,
        _assistant_tool_call_msg("call_1"),
        _tool_msg([{"type": "tool_reference", "tool_name": "WebFetch"}]),
    ]

    result = drop_tool_reference_parts_from_tool_messages(messages)

    assert result[0] == user_message
    assert result[2]["content"] == ""


class TestSystemMessagesFirst:
    def test_stable_partition_keeps_order_within_each_group(self):
        messages = [
            {"role": "user", "content": "u1"},
            {"role": "system", "content": "s1"},
            {"role": "assistant", "content": "a1"},
            {"role": "developer", "content": "d1"},
            {"role": "tool", "tool_call_id": "c1", "content": "t1"},
            {"role": "system", "content": "s2"},
        ]

        result = system_messages_first(messages)

        assert [m["content"] for m in result] == ["s1", "d1", "s2", "u1", "a1", "t1"]
        assert [m["content"] for m in messages] == ["u1", "s1", "a1", "d1", "t1", "s2"]
        assert all(
            result_message is original for result_message, original in zip(result[3:], messages[::2], strict=True)
        )

    @pytest.mark.parametrize(
        "messages",
        [
            [],
            [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}],
            [{"role": "system", "content": "s1"}, {"role": "user", "content": "u1"}],
            [{"role": "system", "content": "s1"}, {"role": "system", "content": "s2"}],
        ],
    )
    def test_already_ordered_messages_come_back_unchanged(self, messages):
        assert system_messages_first(messages) == messages


class TestRequestContainsImageContent:
    """One detector for every dialect that reaches pre-routing hooks untranslated."""

    @pytest.mark.parametrize(
        "part",
        [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
            {"type": "input_image", "image_url": "data:image/png;base64,aGk="},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}},
            {
                "type": "tool_result",
                "tool_use_id": "tu_1",
                "content": [{"type": "image", "source": {"type": "base64", "data": "aGk="}}],
            },
        ],
    )
    def test_detects_every_image_dialect_including_tool_results(self, part):
        from litellm.litellm_core_utils.prompt_templates.common_utils import request_contains_image_content

        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}, part]}]
        assert request_contains_image_content(messages) is True

    @pytest.mark.parametrize(
        "messages",
        [
            [{"role": "user", "content": "plain string"}],
            [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "x"}}]}],
            [{"role": "user", "content": [{"type": "tool_result", "content": [{"type": "text", "text": "ok"}]}]}],
            [{"role": "user", "content": None}],
            [{"role": "user", "content": [None, "text", {"type": []}]}],
            ["malformed"],
            [None, 5],
            "malformed",
            5,
            None,
            [],
        ],
    )
    def test_ignores_text_audio_and_degenerate_shapes(self, messages):
        from litellm.litellm_core_utils.prompt_templates.common_utils import request_contains_image_content

        assert request_contains_image_content(messages) is False

    def test_hostile_nesting_is_depth_bounded(self):
        from litellm.litellm_core_utils.prompt_templates.common_utils import request_contains_image_content

        nested: dict = {"type": "image", "source": {"type": "base64", "data": "aGk="}}
        for _ in range(50):
            nested = {"type": "tool_result", "content": [nested]}
        assert request_contains_image_content([{"role": "user", "content": [nested]}]) is False


class TestEncryptedReasoningReplay:
    """Regression for https://github.com/BerriAI/litellm/issues/40288."""

    def test_signature_round_trips_the_encrypted_content(self):
        assert encrypted_content_from_signature(encrypted_reasoning_signature("gAAAA_bytes")) == "gAAAA_bytes"

    @pytest.mark.parametrize(
        "signature", [None, "", "ErcBCkgIValidAnthropicSignature", "litellm_encrypted_reasoning:", 7]
    )
    def test_anything_else_is_not_encrypted_content(self, signature):
        assert encrypted_content_from_signature(signature) is None

    def test_encrypted_thinking_block_replays_its_own_item(self):
        items = responses_reasoning_items_from_thinking_blocks(
            [{"type": "thinking", "thinking": "Plan.", "signature": encrypted_reasoning_signature("gAAAA_1")}]
        )
        assert items == (
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Plan."}],
                "encrypted_content": "gAAAA_1",
            },
        )

    def test_encrypted_redacted_block_replays_with_an_empty_summary(self):
        items = responses_reasoning_items_from_thinking_blocks(
            [{"type": "redacted_thinking", "data": encrypted_reasoning_signature("gAAAA_1")}]
        )
        assert items == ({"type": "reasoning", "summary": [], "encrypted_content": "gAAAA_1"},)

    def test_plain_blocks_collapse_into_one_summary_item_around_encrypted_ones(self):
        items = responses_reasoning_items_from_thinking_blocks(
            [
                {"type": "thinking", "thinking": "A.", "signature": None},
                {"type": "thinking", "thinking": "B.", "signature": ""},
                {"type": "thinking", "thinking": "C.", "signature": encrypted_reasoning_signature("gAAAA_c")},
                {"type": "redacted_thinking", "data": "anthropic-minted-opaque-data"},
                {"type": "thinking", "thinking": "D."},
            ]
        )
        assert items == (
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "A."}, {"type": "summary_text", "text": "B."}],
            },
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "C."}], "encrypted_content": "gAAAA_c"},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "D."}]},
        )
        assert all("id" not in item for item in items)

    def test_blocks_without_text_or_encrypted_content_produce_nothing(self):
        assert responses_reasoning_items_from_thinking_blocks([{"type": "thinking", "thinking": ""}]) == ()
        assert responses_reasoning_items_from_thinking_blocks([]) == ()

    @pytest.mark.parametrize(
        ("block", "expected"),
        [
            ({"type": "thinking", "thinking": "x", "signature": encrypted_reasoning_signature("g")}, True),
            ({"type": "redacted_thinking", "data": encrypted_reasoning_signature("g")}, True),
            ({"type": "thinking", "thinking": "x", "signature": ENCRYPTED_REASONING_SIGNATURE_PREFIX}, True),
            ({"type": "redacted_thinking", "data": ENCRYPTED_REASONING_SIGNATURE_PREFIX}, True),
            ({"type": "thinking", "thinking": "x", "signature": "ErcBCkgIValid"}, False),
            ({"type": "redacted_thinking", "data": "EmwKAhgBEgy"}, False),
            ({"type": "text", "text": encrypted_reasoning_signature("g")}, False),
            ("not a block", False),
        ],
    )
    def test_is_encrypted_reasoning_block(self, block, expected):
        assert is_encrypted_reasoning_block(block) is expected

    def test_strip_drops_every_bridge_tagged_block_and_leaves_no_unsigned_thinking_behind(self):
        assistant_content = [
            {"type": "thinking", "thinking": "minted by Anthropic", "signature": "ErcBCkgIValid"},
            {"type": "thinking", "thinking": "packed by the bridge", "signature": encrypted_reasoning_signature("g1")},
            {"type": "redacted_thinking", "data": encrypted_reasoning_signature("g2")},
            {"type": "thinking", "thinking": "", "signature": encrypted_reasoning_signature("g3")},
            {"type": "text", "text": "answer"},
        ]
        messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": [{"type": "text", "text": "follow-up"}]},
        ]

        strip_encrypted_reasoning_from_messages(messages)

        assert messages[1]["content"] is assistant_content
        assert assistant_content == [
            {"type": "thinking", "thinking": "minted by Anthropic", "signature": "ErcBCkgIValid"},
            {"type": "text", "text": "answer"},
        ]
        assert all(block["signature"] for block in assistant_content if block["type"] == "thinking")
        assert messages[0] == {"role": "user", "content": "question"}
        assert messages[2] == {"role": "user", "content": [{"type": "text", "text": "follow-up"}]}

    def test_strip_uses_predicate_to_keep_selected_encrypted_blocks(self):
        kept_signature = encrypted_reasoning_signature("keep")
        stripped_signature = encrypted_reasoning_signature("strip")
        content = [
            {"type": "thinking", "thinking": "keep", "signature": kept_signature},
            {"type": "thinking", "thinking": "strip", "signature": stripped_signature},
        ]
        messages = [{"role": "assistant", "content": content}]

        strip_encrypted_reasoning_from_messages(messages, should_strip=lambda block: block.get("thinking") == "strip")

        assert messages[0]["content"] is content
        assert content == [{"type": "thinking", "thinking": "keep", "signature": kept_signature}]

    @pytest.mark.parametrize(
        "messages",
        [
            "not a list",
            None,
            [{"role": "user", "content": None}],
            [{"role": "user", "content": "plain string"}],
            ["not a message"],
            [{"role": "assistant", "content": [{"type": "thinking", "thinking": "x", "signature": "ErcBCkgIValid"}]}],
        ],
    )
    def test_strip_leaves_history_without_bridge_reasoning_untouched(self, messages):
        before = copy.deepcopy(messages)

        strip_encrypted_reasoning_from_messages(messages)

        assert messages == before


class TestMergeConsecutiveSystemMessages:
    def test_merges_each_run_of_string_system_messages_with_a_blank_line(self):
        messages = [
            {"role": "system", "content": "You are terse.", "cache_control": {"type": "ephemeral"}},
            {"role": "system", "content": "Skills: none."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "system", "content": "Reminder A"},
            {"role": "system", "content": "Reminder B"},
            {"role": "user", "content": "Bye"},
        ]

        merged = merge_consecutive_system_messages(messages)

        assert merged == [
            {"role": "system", "content": "You are terse.\n\nSkills: none.", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "system", "content": "Reminder A\n\nReminder B"},
            {"role": "user", "content": "Bye"},
        ]

    def test_merges_into_text_parts_when_any_system_content_is_a_list(self):
        cached_part = {"type": "text", "text": "Skills: none.", "cache_control": {"type": "ephemeral"}}
        messages = [
            {"role": "system", "content": "You are terse."},
            {"role": "system", "content": [cached_part, {"type": "text", "text": "Be brief."}]},
            {"role": "system", "content": "Answer in English."},
            {"role": "user", "content": "Hello"},
        ]

        merged = merge_consecutive_system_messages(messages)

        assert merged == [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are terse."},
                    cached_part,
                    {"type": "text", "text": "Be brief."},
                    {"type": "text", "text": "Answer in English."},
                ],
            },
            {"role": "user", "content": "Hello"},
        ]
        assert merged[0]["content"][1] is cached_part

    @pytest.mark.parametrize(
        "messages",
        [
            [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
            [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "Hello"},
                {"role": "system", "content": "Reminder"},
            ],
            [],
        ],
        ids=["single-system", "no-system", "separated-systems", "empty"],
    )
    def test_leaves_messages_without_consecutive_system_messages_untouched(self, messages):
        before = copy.deepcopy(messages)

        merged = merge_consecutive_system_messages(messages)

        assert merged == before
        assert [message is original for message, original in zip(merged, messages)] == [True] * len(messages)

    @pytest.mark.parametrize(
        ("messages", "expected_content"),
        [
            ([{"role": "system"}, {"role": "system", "content": "Skills: none."}], "Skills: none."),
            ([{"role": "system", "content": "You are terse."}, {"role": "system"}], "You are terse."),
            (
                [{"role": "system"}, {"role": "system", "content": [{"type": "text", "text": "Be brief."}]}],
                [{"type": "text", "text": "Be brief."}],
            ),
        ],
        ids=["missing-then-str", "str-then-missing", "missing-then-list"],
    )
    def test_skips_system_messages_without_content_when_merging(self, messages, expected_content):
        merged = merge_consecutive_system_messages([*messages, {"role": "user", "content": "Hello"}])

        assert merged == [{"role": "system", "content": expected_content}, {"role": "user", "content": "Hello"}]

    def test_keeps_the_first_message_when_no_system_message_in_the_run_has_content(self):
        merged = merge_consecutive_system_messages(
            [{"role": "system"}, {"role": "system"}, {"role": "user", "content": "Hi"}]
        )

        assert merged == [{"role": "system"}, {"role": "user", "content": "Hi"}]


_CLAUDE_CODE_TOOL_TURN: Final = [
    {"role": "user", "content": "list the files"},
    {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}],
    },
    {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "calc.py test_calc.py"}],
    },
]


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        pytest.param(_CLAUDE_CODE_TOOL_TURN, "list the filescalc.py test_calc.py", id="tool-result-string"),
        pytest.param(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [
                                {"type": "text", "text": "x = 1"},
                                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ""}},
                            ],
                        }
                    ],
                }
            ],
            "x = 1",
            id="tool-result-blocks",
        ),
        pytest.param(
            [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1"}]}],
            "",
            id="tool-result-without-content",
        ),
    ],
)
def test_get_semantic_cache_prompt_from_messages_keeps_tool_result_text(
    messages: list[dict[str, object]], expected: str
) -> None:
    assert get_semantic_cache_prompt_from_messages(messages) == expected


def test_get_semantic_cache_prompt_from_messages_differs_from_the_turn_before_it() -> None:
    assert get_str_from_messages(_CLAUDE_CODE_TOOL_TURN) == get_str_from_messages(_CLAUDE_CODE_TOOL_TURN[:1])
    assert get_semantic_cache_prompt_from_messages(_CLAUDE_CODE_TOOL_TURN) != get_semantic_cache_prompt_from_messages(
        _CLAUDE_CODE_TOOL_TURN[:1]
    )


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param([{"role": "system", "content": "be brief. "}, {"role": "user", "content": "hello"}], id="strings"),
        pytest.param(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is "},
                        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                        {"type": "text", "text": "this?"},
                    ],
                }
            ],
            id="text-parts",
        ),
        pytest.param(
            [
                {"role": "assistant"},
                {"role": "assistant", "content": None},
                {"role": "user", "content": ""},
                {"role": "tool", "content": "small", "search_results": [{"source": "s", "title": "t", "content": []}]},
            ],
            id="empty-content-and-search-results",
        ),
    ],
)
def test_get_semantic_cache_prompt_from_messages_matches_get_str_from_messages_without_tool_results(
    messages: list[dict[str, object]],
) -> None:
    assert get_semantic_cache_prompt_from_messages(messages) == get_str_from_messages(messages)


_MALFORMED_PROMPT_CACHE_BREAKPOINTS: Final = (
    "yes",
    1,
    ["explicit"],
    "",
    {},
    {"mode": "bogus"},
    {"mode": "bogus", "ttl": "30m"},
    {1: "explicit"},
)
_MALFORMED_PROMPT_CACHE_BREAKPOINT_IDS: Final = (
    "string",
    "int",
    "list",
    "empty-string",
    "empty-object",
    "bogus-mode",
    "bogus-mode-with-ttl",
    "non-string-key",
)
_VALID_PROMPT_CACHE_BREAKPOINTS: Final = ({"mode": "explicit"}, {"mode": "explicit", "ttl": "30m"})


@pytest.mark.parametrize("drop_params", (True, False), ids=("drop", "keep"))
def test_prompt_cache_breakpoint_for_wire_passes_a_missing_marker_through(drop_params: bool) -> None:
    assert prompt_cache_breakpoint_for_wire(None, drop_params) is None


@pytest.mark.parametrize("marker", _MALFORMED_PROMPT_CACHE_BREAKPOINTS, ids=_MALFORMED_PROMPT_CACHE_BREAKPOINT_IDS)
def test_prompt_cache_breakpoint_for_wire_drops_a_malformed_marker_under_drop_params(marker: object) -> None:
    assert prompt_cache_breakpoint_for_wire(marker, True) is None


@pytest.mark.parametrize("marker", _MALFORMED_PROMPT_CACHE_BREAKPOINTS, ids=_MALFORMED_PROMPT_CACHE_BREAKPOINT_IDS)
def test_prompt_cache_breakpoint_for_wire_forwards_a_malformed_marker_without_drop_params(marker: object) -> None:
    assert prompt_cache_breakpoint_for_wire(marker, False) is marker


@pytest.mark.parametrize("drop_params", (True, False), ids=("drop", "keep"))
@pytest.mark.parametrize("marker", _VALID_PROMPT_CACHE_BREAKPOINTS, ids=("explicit", "explicit-30m"))
def test_prompt_cache_breakpoint_for_wire_keeps_a_valid_marker(marker: dict[str, str], drop_params: bool) -> None:
    assert prompt_cache_breakpoint_for_wire(marker, drop_params) == marker


def test_prompt_cache_breakpoint_for_wire_strips_an_unknown_key_under_drop_params() -> None:
    assert prompt_cache_breakpoint_for_wire({"mode": "explicit", "note": "kept"}, True) == {"mode": "explicit"}


@pytest.mark.parametrize(
    "marker",
    ({"mode": "explicit", "ttl": "1h"}, {"mode": "explicit", "ttl": "1h", "note": "kept"}),
    ids=("bad-ttl", "bad-ttl-and-unknown-key"),
)
def test_prompt_cache_breakpoint_for_wire_drops_an_unsupported_ttl_and_keeps_the_mode_under_drop_params(
    marker: dict[str, str],
) -> None:
    assert prompt_cache_breakpoint_for_wire(marker, True) == {"mode": "explicit"}
    assert prompt_cache_breakpoint_for_wire(marker, False) is marker


def test_cache_control_is_replaced_by_an_explicit_prompt_cache_breakpoint_on_the_same_block() -> None:
    marked: Final = {"type": "text", "text": "rules", "cache_control": {"type": "ephemeral", "ttl": "1h"}}

    assert cache_control_as_prompt_cache_breakpoint(marked) == {
        "type": "text",
        "text": "rules",
        "prompt_cache_breakpoint": {"mode": "explicit"},
    }


def test_a_block_without_cache_control_is_returned_untouched() -> None:
    unmarked: Final = {"type": "text", "text": "rules"}

    assert cache_control_as_prompt_cache_breakpoint(unmarked) is unmarked


def test_a_breakpoint_the_client_wrote_survives_a_message_level_mark_ttl_included() -> None:
    block: Final = {
        "type": "input_text",
        "text": "rules",
        "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "30m"},
    }

    assert with_prompt_cache_breakpoint(block, {"mode": "explicit"}) is block


def test_a_breakpoint_the_client_wrote_survives_a_per_part_cache_control() -> None:
    both: Final = {
        "type": "input_text",
        "text": "rules",
        "cache_control": {"type": "ephemeral"},
        "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "30m"},
    }

    assert cache_control_as_prompt_cache_breakpoint(both) == {
        "type": "input_text",
        "text": "rules",
        "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "30m"},
    }


_NATIVE_REDACTED_DATA: Final = "EmwKAhgBEgy3va3pzix/LafPsn4aDFIT2Xlxh0L5L8rLVyIwxtE3rAFBa8cr3qpP"
_CONVERSE_REDACTED_DATA: Final = base64.b64encode(_NATIVE_REDACTED_DATA.encode()).decode()


@pytest.mark.parametrize(
    "data, native",
    [
        pytest.param(_NATIVE_REDACTED_DATA, _NATIVE_REDACTED_DATA, id="claude-native"),
        pytest.param(_CONVERSE_REDACTED_DATA, _NATIVE_REDACTED_DATA, id="converse-wrapped-legacy-history"),
        pytest.param(
            encrypted_reasoning_signature("gAAAA_1"), encrypted_reasoning_signature("gAAAA_1"), id="openai-encrypted"
        ),
        pytest.param("", "", id="blank"),
        pytest.param(base64.b64encode(b"not base64!").decode(), base64.b64encode(b"not base64!").decode(), id="b64-once"),
    ],
)
def test_native_redacted_thinking_data_only_unwraps_the_converse_base64_layer(data: str, native: str) -> None:
    assert native_redacted_thinking_data(data) == native


@pytest.mark.parametrize("data", [_NATIVE_REDACTED_DATA, _CONVERSE_REDACTED_DATA], ids=["native", "legacy-wrapped"])
def test_converse_redacted_content_wraps_claude_native_data_exactly_once(data: str) -> None:
    assert converse_redacted_content(data) == _CONVERSE_REDACTED_DATA


def test_strip_litellm_response_only_fields_drops_nested_tool_call_provider_fields() -> None:
    message: Final = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "provider_specific_fields": {"namespace": "ns"},
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Paris"}',
                    "provider_specific_fields": {"namespace": "ns"},
                },
            }
        ],
    }

    stripped: Final = strip_litellm_response_only_fields(message)

    tool_call: Final = cast("dict[str, object]", stripped["tool_calls"][0])
    assert "provider_specific_fields" not in tool_call
    function: Final = cast("dict[str, object]", tool_call["function"])
    assert "provider_specific_fields" not in function
    assert function["name"] == "get_weather"
    assert function["arguments"] == '{"city": "Paris"}'


@pytest.mark.parametrize("part_type", ["image_url", "input_image"])
@pytest.mark.parametrize(
    "fields",
    [
        {"image_url": "data:image/png;base64,IMAGE", "detail": "low"},
        {"image_url": {"url": "data:image/png;base64,IMAGE", "detail": "high"}},
        {"url": "data:image/png;base64,IMAGE", "detail": "low"},
    ],
)
def test_shared_image_url_agrees_with_responses_normalization(
    part_type: str, fields: Mapping[str, object],
) -> None:
    from litellm.litellm_core_utils.prompt_templates.common_utils import as_openai_image_part
    from litellm.responses.litellm_completion_transformation.transformation import LiteLLMCompletionResponsesConfig

    part: Final = {"type": part_type, **fields}
    normalized: Final = LiteLLMCompletionResponsesConfig._transform_input_image_item_to_image_item(part)
    expected: Final = {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE"}}
    assert as_openai_image_part(part) == expected
    assert as_openai_image_part(normalized) == expected


@pytest.mark.parametrize(
    "part,projected,media",
    [
        (
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE", "detail": "high"}},
            {"type": "image_url", "image_url": {"url": "", "detail": "high"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE", "detail": "high"}},
        ),
        (
            {"type": "input_image", "image_url": "https://example.com/image.png", "detail": "low"},
            {"type": "input_image", "image_url": "", "detail": "low"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png", "detail": "low"}},
        ),
        (
            {"type": "computer_screenshot", "image_url": "data:image/png;base64,IMAGE"},
            {"type": "computer_screenshot", "image_url": ""},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE", "detail": "auto"}},
        ),
        (
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "IMAGE"}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ""}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE", "detail": "auto"}},
        ),
        (
            {"type": "input_audio", "input_audio": {"data": "AUDIO", "format": "wav"}},
            {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}},
            {"type": "input_audio", "input_audio": {"data": "AUDIO", "format": "wav"}},
        ),
        (
            {"type": "file", "file": {"file_data": "IMAGE", "format": "image/png", "filename": "image.png"}},
            {"type": "file", "file": {"file_data": "", "format": "image/png", "filename": "image.png"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE", "detail": "auto"}},
        ),
        (
            {"type": "input_file", "file_data": "data:audio/wav;base64,AUDIO", "filename": "audio.wav"},
            {"type": "input_file", "file_data": "", "filename": "audio.wav"},
            {"type": "input_audio", "input_audio": {"data": "AUDIO", "format": "wav"}},
        ),
    ],
)
def test_tpm_projection_blanks_only_declared_media_transport(
    part: Mapping[str, object], projected: Mapping[str, object], media: Mapping[str, object],
) -> None:
    text: Final = {"type": "text", "text": "data:image/png;base64,IMAGE"}
    request: Final = [{"role": "user", "content": [text, {**part, "cache_control": {"type": "ephemeral"}}]}]
    original: Final = copy.deepcopy(request)
    assert split_media_for_tpm(request) == (
        [{"role": "user", "content": [text, {**projected, "cache_control": {"type": "ephemeral"}}]}],
        (media,),
    )
    assert request == original


def test_tpm_projection_retains_nested_and_native_history_around_media() -> None:
    image: Final = {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE", "detail": "high"}}
    empty_image: Final = {"type": "image_url", "image_url": {"url": "", "detail": "high"}}
    document: Final = {"type": "document", "title": "keep", "source": {"type": "content", "content": [image]}}
    empty_document: Final = {**document, "source": {"type": "content", "content": [empty_image]}}
    tool: Final = {"type": "tool_result", "tool_use_id": "call", "content": [document, {"type": "text", "text": "keep"}]}
    empty_tool: Final = {**tool, "content": [empty_document, {"type": "text", "text": "keep"}]}
    request: Final = [
        {"role": "user", "content": [tool]},
        {"type": "function_call_output", "call_id": "call", "output": [image]},
        {"type": "custom_tool_call_output", "call_id": "custom", "output": [image]},
        {"type": "computer_call_output", "call_id": "computer", "output": {
            "type": "computer_screenshot", "image_url": "data:image/png;base64,IMAGE",
        }},
        {"type": "image_generation_call", "id": "ig", "result": "IMAGE", "status": "completed"},
        {"type": "code_interpreter_call", "id": "ci", "code": "keep()", "outputs": [
            {"type": "logs", "logs": "keep"}, {"type": "image", "url": "https://example.com/image.png"},
        ]},
    ]
    original: Final = copy.deepcopy(request)
    projected, media = split_media_for_tpm(request)
    assert projected == [
        {"role": "user", "content": [empty_tool]},
        {"type": "function_call_output", "call_id": "call", "output": [empty_image]},
        {"type": "custom_tool_call_output", "call_id": "custom", "output": [empty_image]},
        {"type": "computer_call_output", "call_id": "computer", "output": {
            "type": "computer_screenshot", "image_url": "",
        }},
        {"type": "image_generation_call", "id": "ig", "result": "", "status": "completed"},
        {"type": "code_interpreter_call", "id": "ci", "code": "keep()", "outputs": [
            {"type": "logs", "logs": "keep"}, {"type": "image", "url": ""},
        ]},
    ]
    assert media == (
        image, image, image,
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE", "detail": "auto"}},
        {"type": "image_url", "image_url": {"url": "data:application/octet-stream;base64,IMAGE", "detail": "auto"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png", "detail": "auto"}},
    )
    assert request == original


def test_tpm_projection_preserves_non_media_identity_and_opaque_values() -> None:
    image: Final = {"type": "input_image", "image_url": "data:image/png;base64,OPAQUE"}
    request: Final = [
        {"role": "user", "content": json.dumps(image)},
        {"type": "custom_tool_call_output", "call_id": "call", "output": json.dumps(image)},
        {"type": "function_call", "call_id": "call", "name": "keep", "arguments": {"content": [image]}},
        {"type": "file_search_call", "results": [{"text": "keep", "content": [image]}]},
        {"type": "image_generation_call", "id": "ig", "status": "in_progress", "result": None},
        {"role": "user", "content": [
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "PDF"}},
            {"type": "document", "source": {"type": "text", "data": json.dumps(image)}},
            {"type": "file", "file": {"format": "application/pdf", "file_data": "PDF"}},
            {"type": "text", "text": "keep", "metadata": {"content": [image]}},
        ]},
    ]
    projected, media = split_media_for_tpm(request)
    assert projected is request
    assert media == ()


def test_content_list_flattening_returns_new_messages_and_leaves_the_callers_untouched() -> None:
    messages: Final = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": " world"}]},
        {"role": "assistant", "content": "Hi"},
    ]
    messages_before: Final = copy.deepcopy(messages)

    flattened: Final = handle_messages_with_content_list_to_str_conversion(messages)

    assert flattened == [{"role": "user", "content": "Hello world"}, {"role": "assistant", "content": "Hi"}]
    assert messages == messages_before


@pytest.mark.parametrize("mime", ["application/pdf", "video/mp4", "audio/wav", "image/png"])
def test_embedding_media_split_holds_at_most_one_copy_of_a_large_payload(mime: str) -> None:
    import gc
    import tracemalloc

    uri: Final = f"data:{mime};base64," + base64.b64encode(bytes(3 * 1024 * 1024)).decode("ascii")
    split_embedding_media_for_tpm(f"data:{mime};base64,AAAA")
    gc.collect()
    tracemalloc.start()
    try:
        _, media = split_embedding_media_for_tpm(uri)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert media
    assert peak < 1.5 * len(uri)
