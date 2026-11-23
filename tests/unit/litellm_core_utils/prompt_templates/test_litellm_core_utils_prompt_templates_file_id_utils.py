import os

import pytest

from litellm.litellm_core_utils.prompt_templates.file_id_utils import (
    get_file_ids_from_messages,
    get_format_from_file_id,
    update_messages_with_model_file_ids,
)


UNIFIED_FILE_ID_B64 = (
    "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9wZGY7dW5pZmllZF9pZCw2YzBiNTg5MC04OTE0"
    "LTQ4ZTAtYjhmNC0wYWU1ZWQzYzE0YTU7dGFyZ2V0X21vZGVsX25hbWVzLGdwdC00bztsbG1f"
    "b3V0cHV0X2ZpbGVfaWQsZmlsZS1FQ0JQVzdNTDlnN1hIZHdHZ1VQWmFNO2xsbV9vdXRwdXRf"
    "ZmlsZV9tb2RlbF9pZCxlMjY0NTNmOWU3NmU3OTkzNjgwZDAwNjhkOThjMWY0Y2MyMDViYmFk"
    "MDk2N2EzM2M2NjQ4OTM1NjhjYTc0M2My"
)


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
        from litellm.litellm_core_utils.prompt_templates.file_id_utils import (
            extract_file_data,
        )

        with pytest.raises(ValueError, match="does not accept bare str inputs"):
            extract_file_data("/etc/passwd")

    def test_accepts_pathlib_path(self):
        import tempfile
        from pathlib import Path

        from litellm.litellm_core_utils.prompt_templates.file_id_utils import (
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
        from litellm.litellm_core_utils.prompt_templates.file_id_utils import (
            extract_file_data,
        )

        extracted = extract_file_data(b"raw bytes content")
        assert extracted.get("content") == b"raw bytes content"

    def test_accepts_tuple(self):
        from litellm.litellm_core_utils.prompt_templates.file_id_utils import (
            extract_file_data,
        )

        extracted = extract_file_data(("foo.txt", b"raw bytes content"))
        assert extracted.get("filename") == "foo.txt"
        assert extracted.get("content") == b"raw bytes content"


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
    from litellm.litellm_core_utils.prompt_templates.file_id_utils import (
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
    from litellm.litellm_core_utils.prompt_templates.file_id_utils import (
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

    updated = update_responses_input_with_model_file_ids(input_items, "model-A", {encoded_id: {"model-A": "file-on-a"}})

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
