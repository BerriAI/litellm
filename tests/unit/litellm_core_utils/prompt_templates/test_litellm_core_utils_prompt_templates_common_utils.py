import contextlib
import copy
import importlib
import json
import tracemalloc
import warnings
from collections.abc import Mapping
from typing import Final, cast
from unittest import mock

import pytest

from litellm.litellm_core_utils.prompt_templates import common_utils
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    TOOL_RESULT_IMAGE_BOUNDARY,
    TOOL_RESULT_IMAGE_PLACEHOLDER,
    TOOL_RESULT_MEDIA_BOUNDARY,
    TOOL_RESULT_MEDIA_PLACEHOLDER,
    add_system_prompt_to_messages,
    defs_unpacked,
    get_content_from_model_response,
    get_semantic_cache_prompt_from_messages,
    get_str_from_messages,
    handle_any_messages_to_chat_completion_str_messages_conversion,
    handle_messages_with_content_list_to_str_conversion,
    hoist_images_from_tool_messages,
    merge_consecutive_system_messages,
    strip_litellm_response_only_fields,
    system_messages_first,
)
from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
    flatten_combinators_and_drop_non_python_regex_patterns,
)
from litellm.litellm_core_utils.prompt_templates.tool_call_json_repair import parse_tool_call_arguments
from litellm.types.llms.openai import ChatCompletionAudioObject, ChatCompletionFileObject, ChatCompletionImageObject
from litellm.types.utils import ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse

_ARTIFACT_FIELD_PATTERN: Final = r'^(?!__.*__$)[^\p{Cc}\p{Cf}\p{Zl}\p{Zp}"\\./[\]]{1,200}$'
_ARTIFACT_DATA_ID_PATTERN: Final = r"^(?!\.\.?(?:\/|$))[A-Za-z0-9_\-.~:@+]{1,200}$"


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
            TOOL_RESULT_IMAGE_PLACEHOLDER,
            TOOL_RESULT_IMAGE_BOUNDARY,
        ),
        (
            {"type": "file", "file": {"file_data": "data:application/pdf;base64,cGRm", "filename": "report.pdf"}},
            TOOL_RESULT_MEDIA_PLACEHOLDER,
            TOOL_RESULT_MEDIA_BOUNDARY,
        ),
        (
            {"type": "input_audio", "input_audio": {"data": "c291bmQ=", "format": "wav"}},
            TOOL_RESULT_MEDIA_PLACEHOLDER,
            TOOL_RESULT_MEDIA_BOUNDARY,
        ),
    ],
)
def test_hoist_tool_media_preserves_payload_and_call(
    part: ChatCompletionImageObject | ChatCompletionFileObject | ChatCompletionAudioObject,
    placeholder: str,
    boundary: str,
) -> None:
    messages: Final = [
        _assistant_tool_call_msg("call_1"),
        {**_tool_msg([part]), "cache_control": {"type": "ephemeral"}},
    ]
    original: Final = copy.deepcopy(messages)

    result: Final = hoist_images_from_tool_messages(messages)

    assert result == [
        messages[0],
        {**messages[1], "content": placeholder},
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
    part_type: str,
    fields: Mapping[str, object],
) -> None:
    from litellm.litellm_core_utils.prompt_templates.common_utils import as_openai_image_part
    from litellm.responses.litellm_completion_transformation.transformation import LiteLLMCompletionResponsesConfig

    part: Final = {"type": part_type, **fields}
    normalized: Final = LiteLLMCompletionResponsesConfig._transform_input_image_item_to_image_item(part)
    expected: Final = {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE"}}
    assert as_openai_image_part(part) == expected
    assert as_openai_image_part(normalized) == expected


def test_content_list_flattening_returns_new_messages_and_leaves_the_callers_untouched() -> None:
    messages: Final = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": " world"}]},
        {"role": "assistant", "content": "Hi"},
    ]
    messages_before: Final = copy.deepcopy(messages)

    flattened: Final = handle_messages_with_content_list_to_str_conversion(messages)

    assert flattened == [{"role": "user", "content": "Hello world"}, {"role": "assistant", "content": "Hi"}]
    assert messages == messages_before


_MOVED_PRIVATE_NAMES_BY_MODULE: Final = (
    (
        "file_id_utils",
        ("_decode_vector_store_ids_in_tools",),
    ),
    (
        "json_schema_utils",
        (
            "_EMPTY_SCHEMA",
            "_LOCAL_SCHEMA_REF_PREFIXES",
            "_MAX_SCHEMA_FLATTEN_DEPTH",
            "_MAX_SCHEMA_NESTING",
            "_OPENAI_REJECTED_TOP_LEVEL_SCHEMA_KEYS",
            "_REGEX_LOOKAROUND_RE",
            "_SCHEMA_LIST",
            "_SUBSCHEMA_KEYWORDS",
            "_SUBSCHEMA_LIST_KEYWORDS",
            "_SUBSCHEMA_MAP_KEYWORDS",
            "_TOP_LEVEL_SCHEMA_COMBINATORS",
            "_any_of",
            "_combinator_required_names",
            "_dropped_pattern_properties",
            "_flatten_schema_against_root",
            "_grammar_branches",
            "_is_not_python_regex",
            "_is_object_schema",
            "_is_open_object",
            "_is_union",
            "_keyword_value_rebuilt",
            "_list_at",
            "_may_be_string",
            "_mergeable_branch",
            "_named_schemas",
            "_node_without_rejected_regex",
            "_optional_property_names",
            "_property_schemas",
            "_refs_in",
            "_resolve_local_schema_ref",
            "_root_definitions",
            "_schema_at",
            "_schema_branches",
            "_schema_levels",
            "_schema_nodes",
            "_schema_properties",
            "_schema_required_names",
            "_schema_without_rejected_regex",
            "_subschemas",
            "_uses_regex_lookaround",
        ),
    ),
    (
        "prompt_cache_breakpoint_utils",
        (
            "_BREAKPOINT_BLOCKS",
            "_BREAKPOINT_BLOCK_LIST_KEYS",
            "_BREAKPOINT_FIELDS",
            "_MarkedT",
            "_PROMPT_CACHE_BREAKPOINT",
            "_breakpoints_stripped_from",
            "_without_prompt_cache_breakpoint",
        ),
    ),
    (
        "reasoning_utils",
        (
            "_carries_encrypted_reasoning",
            "_converse_wrapped_payload",
            "_encrypted_reasoning_field",
            "_provider_text_or_none",
            "_readable_thinking_text",
            "_reasoning_item_from_block_group",
            "_reasoning_replay_group_key",
            "_strip_encrypted_reasoning_from_blocks",
        ),
    ),
    (
        "tool_call_json_repair",
        (
            "_DECODED_JSON",
            "_attempt_json_repair",
        ),
    ),
    (
        "tpm_media_utils",
        (
            "_TPMMediaProjection",
            "_tpm_blank_fields",
            "_tpm_code_output_media",
            "_tpm_document_media",
            "_tpm_file_media",
            "_tpm_image_part",
            "_tpm_input_media",
            "_tpm_media_leaf",
            "_tpm_media_list",
            "_tpm_replace_field",
            "_tpm_tool_result_media",
        ),
    ),
)


@pytest.mark.parametrize(("module_name", "names"), _MOVED_PRIVATE_NAMES_BY_MODULE)
def test_moved_private_helpers_still_import_from_common_utils(module_name: str, names: tuple[str, ...]) -> None:
    new_home: Final = importlib.import_module(f"litellm.litellm_core_utils.prompt_templates.{module_name}")
    unreachable: Final = [name for name in names if getattr(common_utils, name, None) is not getattr(new_home, name)]
    assert unreachable == []


def test_unknown_private_name_still_raises_attribute_error_on_common_utils() -> None:
    missing_name: Final = "_not_a_moved_helper"
    with pytest.raises(AttributeError, match=missing_name):
        getattr(common_utils, missing_name)


def test_old_path_access_warns_from_the_callers_line_with_the_new_home() -> None:
    moved_name: Final = "_any_of"
    with pytest.warns(DeprecationWarning, match=r"common_utils\._any_of moved to .*\.json_schema_utils") as caught:
        getattr(common_utils, moved_name)
    assert [warning.filename for warning in caught] == [__file__]


def test_patching_a_moved_helper_at_its_old_path_reaches_the_callers_in_its_new_home() -> None:
    patched_result: Final = {"patched": True}
    with mock.patch(
        "litellm.litellm_core_utils.prompt_templates.common_utils._attempt_json_repair", return_value=patched_result
    ):
        assert parse_tool_call_arguments('{"a": 1') == patched_result
    assert parse_tool_call_arguments('{"a": 1') == {"a": 1}


def test_patching_a_re_exported_helper_at_its_old_path_reaches_the_callers_in_its_new_home() -> None:
    schema: Final = {"anyOf": [{"type": "object", "properties": {"a": {"type": "string"}}}]}
    unpatched: Final = flatten_combinators_and_drop_non_python_regex_patterns(schema)
    patched_result: Final = {"patched": True}
    with mock.patch(
        "litellm.litellm_core_utils.prompt_templates.common_utils.flatten_top_level_schema_combinators",
        return_value=patched_result,
    ):
        assert flatten_combinators_and_drop_non_python_regex_patterns(schema) == patched_result
    assert flatten_combinators_and_drop_non_python_regex_patterns(schema) == unpatched


def test_old_path_patch_restores_the_moved_helper_when_warnings_become_errors_inside_it() -> None:
    with warnings.catch_warnings(), contextlib.suppress(DeprecationWarning):
        with mock.patch(
            "litellm.litellm_core_utils.prompt_templates.common_utils._attempt_json_repair", return_value=None
        ):
            warnings.simplefilter("error", DeprecationWarning)
    assert parse_tool_call_arguments('{"a": 1') == {"a": 1}


def test_old_path_patch_with_create_leaves_the_moved_helper_in_place() -> None:
    with mock.patch(
        "litellm.litellm_core_utils.prompt_templates.common_utils._attempt_json_repair", return_value=None, create=True
    ):
        assert parse_tool_call_arguments("{}") == {}
    assert parse_tool_call_arguments('{"a": 1') == {"a": 1}


def test_overriding_a_moved_helper_through_the_old_path_warns_and_deleting_it_restores_the_released_one() -> None:
    moved_name: Final = "_attempt_json_repair"
    with pytest.warns(DeprecationWarning, match=r"common_utils\._attempt_json_repair moved to .*\.tool_call_json_repair"):
        setattr(common_utils, moved_name, mock.Mock(return_value={"overridden": True}))
    assert parse_tool_call_arguments('{"a": 1') == {"overridden": True}
    delattr(common_utils, moved_name)
    assert parse_tool_call_arguments('{"a": 1') == {"a": 1}


def test_patching_a_name_common_utils_still_defines_stays_on_common_utils() -> None:
    with mock.patch.object(common_utils, "invalid_type_message", return_value="patched"):
        assert common_utils.invalid_type_message("field", 1) == "patched"
    assert common_utils.invalid_type_message("field", 1) != "patched"


def test_get_content_from_model_response_includes_server_tool_calls_before_client_calls() -> None:
    server_call: Final = ChatCompletionMessageToolCall(
        id="srvtoolu_01", type="function", function=Function(name="web_search", arguments='{"query": "persimmon"}')
    )
    client_call: Final = ChatCompletionMessageToolCall(
        id="call_1", type="function", function=Function(name="get_weather", arguments='{"city": "SF"}')
    )
    response: Final = ModelResponse(
        choices=[
            Choices(
                index=0,
                finish_reason="tool_calls",
                message=Message(
                    role="assistant",
                    content="Looking it up.",
                    tool_calls=[client_call.model_dump()],
                    provider_specific_fields={"server_tool_calls": [server_call.model_dump()]},
                ),
            )
        ],
    )

    assert get_content_from_model_response(response) == (
        "Looking it up." + server_call.model_dump_json() + client_call.model_dump_json()
    )


def test_defs_unpacked_keeps_memory_flat_for_a_schema_with_many_unused_defs() -> None:
    schema: Final = {
        "type": "object",
        "properties": {f"p{i}": {"type": "string"} for i in range(2000)},
        "$defs": {f"D{i}": {"type": "string"} for i in range(2000)},
    }
    tracemalloc.start()
    try:
        defs_unpacked(schema)
        peak_bytes: Final = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak_bytes < 10_000_000


def test_defs_unpacked_resolves_each_ref_against_its_nearest_defs() -> None:
    nested: Final = {
        "$defs": {"Leaf": {"type": "integer"}},
        "type": "object",
        "properties": {"leaf": {"$ref": "#/$defs/Leaf"}},
    }
    schema: Final = {
        "type": "object",
        "properties": {"outer": {"$ref": "#/$defs/Leaf"}, "nested": nested},
        "$defs": {"Leaf": {"type": "string"}},
    }
    assert defs_unpacked(schema) == {
        "type": "object",
        "properties": {
            "outer": {"type": "string"},
            "nested": {**nested, "properties": {"leaf": {"type": "integer"}}},
        },
    }


def test_defs_unpacked_resolves_refs_into_both_defs_and_definitions() -> None:
    schema: Final = {
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/A"}, "b": {"$ref": "#/definitions/B"}},
        "$defs": {"A": {"type": "string"}},
        "definitions": {"B": {"type": "integer"}},
    }
    assert defs_unpacked(schema) == {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
    }


def test_defs_unpacked_resolves_a_name_both_blocks_define_to_its_defs_schema() -> None:
    schema: Final = {
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/T"}},
        "$defs": {"T": {"type": "string"}},
        "definitions": {"T": {"type": "integer"}},
    }
    assert defs_unpacked(schema) == {"type": "object", "properties": {"a": {"type": "string"}}}


def test_defs_unpacked_inlines_a_ref_target_alone_unless_asked_to_keep_its_siblings() -> None:
    city: Final = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    schema: Final = {
        "type": "object",
        "properties": {"capital": {"$ref": "#/$defs/City", "description": "The capital", "required": ["country"]}},
        "$defs": {"City": city},
    }
    assert defs_unpacked(schema) == {"type": "object", "properties": {"capital": city}}


def test_defs_unpacked_keeps_the_annotations_next_to_a_ref() -> None:
    city: Final = {"type": "object", "properties": {"name": {"type": "string"}}, "description": "A city"}
    schema: Final = {
        "type": "object",
        "properties": {"capital": {"$ref": "#/$defs/City", "description": "The capital city"}},
        "$defs": {"City": city},
    }
    assert defs_unpacked(schema, keep_ref_siblings=True) == {
        "type": "object",
        "properties": {"capital": {**city, "description": "The capital city"}},
    }


def test_defs_unpacked_adds_a_required_next_to_a_ref_to_the_target_constraints() -> None:
    city: Final = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "country": {"type": "string"}},
        "required": ["name"],
    }
    town: Final = {"type": "object", "properties": {"name": {"type": "string"}}}
    schema: Final = {
        "type": "object",
        "properties": {
            "capital": {"$ref": "#/$defs/City", "required": ["country", "name", "country"], "type": "string"},
            "town": {"$ref": "#/$defs/Town", "required": ["name"]},
        },
        "$defs": {"City": city, "Town": town},
    }
    assert defs_unpacked(schema, keep_ref_siblings=True) == {
        "type": "object",
        "properties": {
            "capital": {**city, "required": ["name", "country"]},
            "town": {**town, "required": ["name"]},
        },
    }


def test_defs_unpacked_drops_a_required_next_to_a_ref_when_either_side_is_not_a_list_of_names() -> None:
    city: Final = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "country": {"type": "string"}},
        "required": ["name"],
    }
    town: Final = {"type": "object", "properties": {"name": {"type": "string"}}, "required": "name"}
    schema: Final = {
        "type": "object",
        "properties": {
            "capital": {"$ref": "#/$defs/City", "required": ["country", {"name": True}]},
            "town": {"$ref": "#/$defs/Town", "required": ["name"]},
        },
        "$defs": {"City": city, "Town": town},
    }
    assert defs_unpacked(schema, keep_ref_siblings=True) == {"type": "object", "properties": {"capital": city, "town": town}}


def test_defs_unpacked_counts_the_annotations_next_to_a_ref_against_the_inlining_budget() -> None:
    hops: Final = 120
    chain: Final = {f"D{i}": {"$ref": f"#/$defs/D{i + 1}"} for i in range(hops)}
    schema: Final = {
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/D0", "description": "x" * 100_000}},
        "$defs": {**chain, f"D{hops}": {"type": "string"}},
    }
    with pytest.raises(ValueError, match="byte budget"):
        defs_unpacked(schema, keep_ref_siblings=True)
