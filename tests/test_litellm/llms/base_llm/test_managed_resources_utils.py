"""
Tests for `litellm.llms.base_llm.managed_resources.utils.extract_model_id_from_unified_id`.

The regex inside this helper is shared by both the vector-store unified-ID
format (`...;model_id,<value>;...`) and the file-ID format (`...;llm_output_file_model_id,<uuid>`).
A naive regex (`r"model_id,([^;]+)"`) substring-matches the latter and
returns the deployment UUID, which then gets fed as a model candidate
into the team-access check and 403s every team-BYOK file attach
(LIT-3244 patch/1.86.0 second-order finding). These tests pin the
field-boundary anchor that prevents that.
"""

import pytest

from litellm.llms.base_llm.managed_resources.utils import (
    encode_unified_id,
    extract_model_id_from_unified_id,
)

# ---------------------------------------------------------------------------
# Vector-store unified-ID shape — has a top-level `model_id,<value>` field.
# Existing behavior must be preserved: returns the value.
# ---------------------------------------------------------------------------


def test_extract_model_id_returns_value_for_vector_store_unified_id():
    unified_id = (
        "litellm_proxy:vector_store"
        ";unified_id,abc-123"
        ";target_model_names,gpt-4,gemini"
        ";resource_id,vs_xyz"
        ";model_id,deployment-uuid-456"
    )
    assert extract_model_id_from_unified_id(unified_id) == "deployment-uuid-456"


def test_extract_model_id_returns_value_when_field_is_first():
    """`model_id` is the very first field after the prefix (anchor must accept start-of-string)."""
    unified_id = "litellm_proxy:vector_store;model_id,first-field-value;unified_id,abc"
    # First field after the prefix is preceded by `;`, so it matches via the
    # `;model_id,` branch. Pin that the anchor isn't accidentally too strict.
    assert extract_model_id_from_unified_id(unified_id) == "first-field-value"


# ---------------------------------------------------------------------------
# File-ID shape — has `llm_output_file_model_id,<uuid>` but no top-level
# `model_id,` field. Must return None (the previous regex would have
# substring-matched and returned the deployment UUID).
# ---------------------------------------------------------------------------


def test_extract_model_id_returns_none_for_file_id_without_model_id_field():
    """Regression pin for LIT-3244 patch/1.86.0.

    File-IDs constructed via `LITELLM_MANAGED_FILE_COMPLETE_STR` have
    `llm_output_file_model_id,<deployment_uuid>` but no top-level
    `model_id,` field. The previous regex matched the substring and
    returned the UUID, which then 403'd team-BYOK file attaches with
    `Tried to access <uuid>`.
    """
    file_id = (
        "litellm_proxy:text/plain"
        ";unified_id,file-uuid-123"
        ";target_model_names,openai/gpt-4o"
        ";llm_output_file_id,file-OpenAIReturnedId"
        ";llm_output_file_model_id,813bf25f-e5a7-4658-8253-a6f677be8eb5"
    )
    assert extract_model_id_from_unified_id(file_id) is None, (
        "File-ID has no top-level `model_id,` field — the deployment UUID "
        "in `llm_output_file_model_id,` must NOT be returned. Returning it "
        "feeds the UUID as a model candidate into the team-access check "
        "and 403s every team-BYOK file attach (LIT-3244 patch/1.86.0)."
    )


def test_extract_model_id_returns_none_for_file_id_with_model_id_value_null():
    """The current file-ID builder writes `llm_output_file_model_id,None`
    (the Python `None` stringified) when the upstream model_id isn't known.
    Still no top-level `model_id,` field → must return None.
    """
    file_id = (
        "litellm_proxy:text/plain"
        ";unified_id,uuid"
        ";target_model_names,openai/gpt-4o"
        ";llm_output_file_id,file-Y"
        ";llm_output_file_model_id,None"
    )
    assert extract_model_id_from_unified_id(file_id) is None


# ---------------------------------------------------------------------------
# Base64-encoded inputs must decode and apply the same anchor.
# ---------------------------------------------------------------------------


def test_extract_model_id_decodes_base64_then_anchors():
    file_id_plain = (
        "litellm_proxy:text/plain"
        ";unified_id,uuid"
        ";target_model_names,openai/gpt-4o"
        ";llm_output_file_id,file-Y"
        ";llm_output_file_model_id,813bf25f-e5a7-4658-8253-a6f677be8eb5"
    )
    encoded = encode_unified_id(file_id_plain)
    assert extract_model_id_from_unified_id(encoded) is None

    vector_store_plain = (
        "litellm_proxy:vector_store"
        ";unified_id,abc"
        ";target_model_names,gpt-4"
        ";resource_id,vs_xyz"
        ";model_id,real-model-id"
    )
    encoded_vs = encode_unified_id(vector_store_plain)
    assert extract_model_id_from_unified_id(encoded_vs) == "real-model-id"


# ---------------------------------------------------------------------------
# Defensive: malformed / non-string inputs must not raise.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_input", [None, 42, b"bytes-not-str", []])
def test_extract_model_id_returns_none_for_non_string_input(bad_input):
    assert extract_model_id_from_unified_id(bad_input) is None  # type: ignore[arg-type]


def test_extract_model_id_returns_none_when_field_absent():
    assert (
        extract_model_id_from_unified_id(
            "litellm_proxy:other;unified_id,abc;some_field,whatever"
        )
        is None
    )
