"""
Unit tests for encode_file_id_with_model, decode_model_from_file_id,
and get_original_file_id in common_utils.py.

Tests the model-based routing ID encoding/decoding used by the batch
and file proxy endpoints.
"""

from litellm.proxy.openai_files_endpoints.common_utils import (
    decode_model_from_file_id,
    encode_file_id_with_model,
    get_original_file_id,
)


class TestEncodeFileIdWithModel:
    """Tests for encode_file_id_with_model."""

    def test_openai_file_id_gets_file_prefix(self):
        """OpenAI file IDs (file-xxx) should produce file- prefix."""
        result = encode_file_id_with_model("file-abc123", "gpt-4o")
        assert result.startswith("file-")

    def test_openai_batch_id_gets_batch_prefix(self):
        """OpenAI batch IDs (batch_xxx) should produce batch_ prefix."""
        result = encode_file_id_with_model("batch_abc123", "gpt-4o")
        assert result.startswith("batch_")

    def test_vertex_numeric_batch_id_gets_batch_prefix_with_id_type(self):
        """Vertex AI numeric batch IDs should produce batch_ prefix when id_type='batch'."""
        result = encode_file_id_with_model(
            "3814889423749775360", "gemini-2.5-pro", id_type="batch"
        )
        assert result.startswith("batch_"), (
            f"Expected batch_ prefix for Vertex numeric batch ID, got: {result[:10]}"
        )

    def test_vertex_numeric_id_defaults_to_file_prefix(self):
        """Vertex AI numeric IDs should default to file- prefix when id_type is not specified."""
        result = encode_file_id_with_model("3814889423749775360", "gemini-2.5-pro")
        assert result.startswith("file-"), (
            "Default id_type should produce file- prefix for backward compatibility"
        )

    def test_gcs_uri_gets_file_prefix(self):
        """GCS URIs (output_file_id) should produce file- prefix."""
        result = encode_file_id_with_model(
            "gs://bucket/path/to/file.jsonl", "gemini-2.5-pro"
        )
        assert result.startswith("file-")


class TestRoundTrip:
    """Tests for encode -> decode round-trip integrity."""

    def test_roundtrip_openai_file_id(self):
        """Encode then decode an OpenAI file ID — model and original ID should be recovered."""
        original = "file-abc123"
        model = "gpt-4o-litellm"
        encoded = encode_file_id_with_model(original, model)

        assert decode_model_from_file_id(encoded) == model
        assert get_original_file_id(encoded) == original

    def test_roundtrip_openai_batch_id(self):
        """Encode then decode an OpenAI batch ID — model and original ID should be recovered."""
        original = "batch_abc123"
        model = "gpt-4o-test"
        encoded = encode_file_id_with_model(original, model)

        assert decode_model_from_file_id(encoded) == model
        assert get_original_file_id(encoded) == original

    def test_roundtrip_vertex_numeric_batch_id(self):
        """Encode then decode a Vertex AI numeric batch ID with id_type='batch'."""
        original = "3814889423749775360"
        model = "gemini-2.5-pro"
        encoded = encode_file_id_with_model(original, model, id_type="batch")

        assert encoded.startswith("batch_")
        assert decode_model_from_file_id(encoded) == model
        assert get_original_file_id(encoded) == original

    def test_roundtrip_vertex_gcs_uri_file_id(self):
        """Encode then decode a Vertex AI GCS URI (output file)."""
        original = "gs://vertex-bucket/litellm-files/output.jsonl"
        model = "gemini-2.5-pro"
        encoded = encode_file_id_with_model(original, model)

        assert encoded.startswith("file-")
        assert decode_model_from_file_id(encoded) == model
        assert get_original_file_id(encoded) == original


class TestDecodeEdgeCases:
    """Tests for decode functions with non-encoded inputs."""

    def test_decode_model_returns_none_for_plain_id(self):
        """Plain (non-encoded) IDs should return None from decode_model_from_file_id."""
        assert decode_model_from_file_id("batch_abc123") is None
        assert decode_model_from_file_id("file-abc123") is None
        assert decode_model_from_file_id("3814889423749775360") is None

    def test_get_original_file_id_returns_input_for_plain_id(self):
        """Plain (non-encoded) IDs should be returned as-is from get_original_file_id."""
        assert get_original_file_id("batch_abc123") == "batch_abc123"
        assert get_original_file_id("file-abc123") == "file-abc123"

    def test_decode_model_handles_non_string(self):
        """Non-string inputs should return None without raising."""
        assert decode_model_from_file_id(None) is None  # type: ignore[arg-type]
        assert decode_model_from_file_id(12345) is None  # type: ignore[arg-type]
