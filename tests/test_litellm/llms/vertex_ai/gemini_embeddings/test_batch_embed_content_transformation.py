"""
Tests for Gemini batchEmbedContents transformation logic.

Covers:
- Text-only inputs (single and batch)
- Multimodal inputs (data URIs, GCS URLs, file references)
- Mixed text + multimodal inputs
- Response processing with correct indices
"""

import pytest

from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.llms.vertex_ai.gemini_embeddings.batch_embed_content_transformation import (
    _build_part_for_input,
    _is_multimodal_input,
    process_embed_content_response,
    process_response,
    transform_openai_input_gemini_content,
    transform_openai_input_gemini_embed_content,
)
from litellm.types.llms.vertex_ai import VertexAIBatchEmbeddingsResponseObject
from litellm.types.utils import EmbeddingResponse


IMAGE_DATA_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAIAQMAAAD+wSzIAAAABlBMVEX///+/v7+jQ3Y5AAAADklEQVQI12P4AIX8EAgALgAD/aNpbtEAAAAASUVORK5CYII"
GCS_URL = "gs://my-bucket/image.png"


class TestIsMultimodalInput:
    def test_text_only_string(self):
        assert _is_multimodal_input("hello world") is False

    def test_text_only_list(self):
        assert _is_multimodal_input(["hello", "world"]) is False

    def test_data_uri(self):
        assert _is_multimodal_input([IMAGE_DATA_URI]) is True

    def test_gcs_url(self):
        assert _is_multimodal_input([GCS_URL]) is True

    def test_file_reference(self):
        assert _is_multimodal_input(["files/abc123"]) is True

    def test_mixed_text_and_image(self):
        assert _is_multimodal_input(["hello", IMAGE_DATA_URI]) is True

    def test_nested_text_is_not_multimodal(self):
        """Nested list with text is not multimodal."""
        assert _is_multimodal_input([["text_a", "text_b"]]) is False

    def test_nested_list_with_image_is_multimodal(self):
        assert _is_multimodal_input([["a red shoe", IMAGE_DATA_URI]]) is True


class TestBuildPartForInput:
    def test_text_input(self):
        part = _build_part_for_input("hello")
        assert part["text"] == "hello"
        assert part.get("inline_data") is None

    def test_data_uri_input(self):
        part = _build_part_for_input(IMAGE_DATA_URI)
        assert part.get("text") is None
        assert part["inline_data"] is not None
        assert part["inline_data"]["mime_type"] == "image/png"

    def test_gcs_url_input(self):
        part = _build_part_for_input(GCS_URL)
        assert part.get("text") is None
        assert part["file_data"] is not None
        assert part["file_data"]["mime_type"] == "image/png"
        assert part["file_data"]["file_uri"] == GCS_URL

    def test_file_reference_resolved(self):
        resolved = {
            "files/abc": {"mime_type": "image/jpeg", "uri": "https://example.com/abc"}
        }
        part = _build_part_for_input("files/abc", resolved_files=resolved)
        assert part["file_data"] is not None
        assert part["file_data"]["mime_type"] == "image/jpeg"

    def test_file_reference_unresolved_raises(self):
        with pytest.raises(ValueError, match="not resolved"):
            _build_part_for_input("files/abc")


class TestTransformOpenaiInputGeminiContent:
    """Test that transform_openai_input_gemini_content creates separate requests per input."""

    def test_single_text(self):
        result = transform_openai_input_gemini_content(
            input="hello", model="gemini-embedding-2-preview", optional_params={}
        )
        assert len(result["requests"]) == 1
        assert result["requests"][0]["content"]["parts"][0]["text"] == "hello"

    def test_multiple_texts(self):
        result = transform_openai_input_gemini_content(
            input=["hello", "world"],
            model="gemini-embedding-2-preview",
            optional_params={},
        )
        assert len(result["requests"]) == 2
        assert result["requests"][0]["content"]["parts"][0]["text"] == "hello"
        assert result["requests"][1]["content"]["parts"][0]["text"] == "world"

    def test_multimodal_inputs_are_separate_requests(self):
        """Key regression test for #24209: each input becomes its own request."""
        result = transform_openai_input_gemini_content(
            input=["The food was delicious", IMAGE_DATA_URI],
            model="gemini-embedding-2-preview",
            optional_params={},
        )
        assert len(result["requests"]) == 2
        # First request is text
        assert (
            result["requests"][0]["content"]["parts"][0]["text"]
            == "The food was delicious"
        )
        # Second request is image
        assert result["requests"][1]["content"]["parts"][0]["inline_data"] is not None

    def test_dimensions_mapped_to_output_dimensionality(self):
        result = transform_openai_input_gemini_content(
            input="hello",
            model="gemini-embedding-2-preview",
            optional_params={"dimensions": 256},
        )
        assert result["requests"][0]["outputDimensionality"] == 256

    def test_model_name_prefixed(self):
        result = transform_openai_input_gemini_content(
            input="hello", model="gemini-embedding-2-preview", optional_params={}
        )
        assert result["requests"][0]["model"] == "models/gemini-embedding-2-preview"

    def test_gcs_url_input(self):
        result = transform_openai_input_gemini_content(
            input=[GCS_URL], model="gemini-embedding-2-preview", optional_params={}
        )
        assert len(result["requests"]) == 1
        assert result["requests"][0]["content"]["parts"][0]["file_data"] is not None

    def test_mixed_text_image_gcs(self):
        result = transform_openai_input_gemini_content(
            input=["hello", IMAGE_DATA_URI, GCS_URL],
            model="gemini-embedding-2-preview",
            optional_params={},
        )
        assert len(result["requests"]) == 3

    def test_nested_input_combined_embedding(self):
        """Nested list produces one request with multiple parts (combined embedding)."""
        result = transform_openai_input_gemini_content(
            input=[["a red shoe", IMAGE_DATA_URI]],
            model="gemini-embedding-2-preview",
            optional_params={},
        )
        assert len(result["requests"]) == 1
        parts = result["requests"][0]["content"]["parts"]
        assert len(parts) == 2
        assert parts[0]["text"] == "a red shoe"
        assert parts[1]["inline_data"] is not None

    def test_mixed_nested_and_flat(self):
        """Mixed nested + flat produces correct number of requests."""
        result = transform_openai_input_gemini_content(
            input=[["text", IMAGE_DATA_URI], "standalone"],
            model="gemini-embedding-2-preview",
            optional_params={},
        )
        assert len(result["requests"]) == 2
        # First: combined (2 parts)
        assert len(result["requests"][0]["content"]["parts"]) == 2
        # Second: standalone (1 part)
        assert len(result["requests"][1]["content"]["parts"]) == 1
        assert result["requests"][1]["content"]["parts"][0]["text"] == "standalone"


class TestTransformOpenaiInputGeminiEmbedContent:
    """Test transform_openai_input_gemini_embed_content (vertex_ai / embedContent path)."""

    def test_text_and_image_combined(self):
        result = transform_openai_input_gemini_embed_content(
            input=["hello", IMAGE_DATA_URI],
            model="gemini-embedding-2-preview",
            optional_params={},
        )
        assert "content" in result
        parts = result["content"]["parts"]
        assert len(parts) == 2
        assert parts[0]["text"] == "hello"
        assert parts[1]["inline_data"] is not None

    def test_gcs_url(self):
        result = transform_openai_input_gemini_embed_content(
            input=[GCS_URL],
            model="gemini-embedding-2-preview",
            optional_params={},
        )
        parts = result["content"]["parts"]
        assert len(parts) == 1
        assert parts[0]["file_data"]["file_uri"] == GCS_URL

    def test_dimensions_mapped(self):
        result = transform_openai_input_gemini_embed_content(
            input="hello",
            model="gemini-embedding-2-preview",
            optional_params={"dimensions": 256},
        )
        assert result["outputDimensionality"] == 256


class TestProcessResponse:
    """Test that process_response sets correct indices."""

    def test_single_embedding_index(self):
        predictions: VertexAIBatchEmbeddingsResponseObject = {
            "embeddings": [{"values": [0.1, 0.2]}]
        }
        model_response = EmbeddingResponse()
        result = process_response(
            input="hello",
            model_response=model_response,
            model="gemini-embedding-2-preview",
            _predictions=predictions,
        )
        assert len(result.data) == 1
        assert result.data[0]["index"] == 0

    def test_multiple_embeddings_have_correct_indices(self):
        """Regression test: indices should be 0, 1, 2... not all 0."""
        predictions: VertexAIBatchEmbeddingsResponseObject = {
            "embeddings": [
                {"values": [0.1, 0.2]},
                {"values": [0.3, 0.4]},
                {"values": [0.5, 0.6]},
            ]
        }
        model_response = EmbeddingResponse()
        result = process_response(
            input=["a", "b", "c"],
            model_response=model_response,
            model="gemini-embedding-2-preview",
            _predictions=predictions,
        )
        assert len(result.data) == 3
        assert result.data[0]["index"] == 0
        assert result.data[1]["index"] == 1
        assert result.data[2]["index"] == 2

    def test_multimodal_mixed_input(self):
        """process_response works with mixed text + multimodal inputs."""
        predictions: VertexAIBatchEmbeddingsResponseObject = {
            "embeddings": [{"values": [0.1, 0.2]}, {"values": [0.3, 0.4]}]
        }
        result = process_response(
            input=["hello", IMAGE_DATA_URI],
            model_response=EmbeddingResponse(),
            model="gemini-embedding-2-preview",
            _predictions=predictions,
        )
        assert len(result.data) == 2
        assert result.data[0]["index"] == 0
        assert result.data[1]["index"] == 1
        # Should count tokens only for the text element, not the image
        assert result.usage.prompt_tokens > 0

    def test_nested_input_token_counting(self):
        """Nested list: only plain-text sub-elements should be counted."""
        predictions: VertexAIBatchEmbeddingsResponseObject = {
            "embeddings": [{"values": [0.1, 0.2]}]
        }
        result = process_response(
            input=[["a red shoe", IMAGE_DATA_URI]],
            model_response=EmbeddingResponse(),
            model="gemini-embedding-2-preview",
            _predictions=predictions,
        )
        assert len(result.data) == 1
        assert result.usage.prompt_tokens > 0

    def test_nested_empty_list_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            transform_openai_input_gemini_content(
                input=[[]],
                model="gemini-embedding-2-preview",
                optional_params={},
            )

    def test_nested_non_string_element_raises(self):
        with pytest.raises(ValueError, match="must be strings"):
            transform_openai_input_gemini_content(
                input=[[["doubly", "nested"]]],
                model="gemini-embedding-2-preview",
                optional_params={},
            )


class TestProcessEmbedContentResponseUsage:
    """Gemini Embedding 2 embedContent usageMetadata must drive spend.

    Regression for multimodal calls recording prompt_tokens=0 / spend=$0.
    """

    MODEL = "gemini-embedding-2"

    def test_multimodal_image_preserves_usage_metadata(self):
        response_json = {
            "embedding": {"values": [0.1, 0.2, 0.3]},
            "usageMetadata": {
                "promptTokenCount": 258,
                "totalTokenCount": 258,
                "promptTokensDetails": [{"modality": "IMAGE", "tokenCount": 258}],
            },
        }
        result = process_embed_content_response(
            input=[IMAGE_DATA_URI],
            model_response=EmbeddingResponse(),
            model=self.MODEL,
            response_json=response_json,
        )
        assert result.usage.prompt_tokens == 258
        assert result.usage.total_tokens == 258
        assert result.usage.prompt_tokens_details.image_count == 1

        prompt_cost, _ = generic_cost_per_token(
            model=self.MODEL,
            usage=result.usage,
            custom_llm_provider="vertex_ai",
        )
        assert prompt_cost > 0

    def test_text_modality_detail_populated(self):
        response_json = {
            "embedding": {"values": [0.1, 0.2]},
            "usageMetadata": {
                "promptTokenCount": 12,
                "totalTokenCount": 12,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 12}],
            },
        }
        result = process_embed_content_response(
            input="a short caption",
            model_response=EmbeddingResponse(),
            model=self.MODEL,
            response_json=response_json,
        )
        assert result.usage.prompt_tokens == 12
        assert result.usage.prompt_tokens_details.text_tokens == 12

        prompt_cost, _ = generic_cost_per_token(
            model=self.MODEL,
            usage=result.usage,
            custom_llm_provider="vertex_ai",
        )
        assert prompt_cost > 0

    def test_video_modality_derives_seconds_and_text_floor(self):
        response_json = {
            "embedding": {"values": [0.1]},
            "usageMetadata": {
                "promptTokenCount": 516,
                "totalTokenCount": 516,
                "promptTokensDetails": [{"modality": "VIDEO", "tokenCount": 516}],
            },
        }
        result = process_embed_content_response(
            input=["gs://bucket/clip.mp4"],
            model_response=EmbeddingResponse(),
            model=self.MODEL,
            response_json=response_json,
        )
        assert result.usage.prompt_tokens == 516
        assert result.usage.prompt_tokens_details.video_length_seconds == pytest.approx(
            2.0
        )
        assert result.usage.prompt_tokens_details.text_tokens == 1

    def test_missing_usage_metadata_does_not_estimate_from_base64(self):
        response_json = {"embedding": {"values": [0.1, 0.2]}}
        result = process_embed_content_response(
            input=[IMAGE_DATA_URI],
            model_response=EmbeddingResponse(),
            model=self.MODEL,
            response_json=response_json,
        )
        assert result.usage.prompt_tokens == 0
        assert result.usage.total_tokens == 0

    def test_missing_usage_metadata_text_falls_back_to_token_counter(self):
        response_json = {"embedding": {"values": [0.1, 0.2]}}
        result = process_embed_content_response(
            input="hello world this is plain text",
            model_response=EmbeddingResponse(),
            model=self.MODEL,
            response_json=response_json,
        )
        assert result.usage.prompt_tokens > 0

    def test_file_reference_image_billed_per_image_not_text(self):
        """files/... image refs must bill per-image, not at the text token rate."""
        response_json = {
            "embedding": {"values": [0.1, 0.2, 0.3]},
            "usageMetadata": {
                "promptTokenCount": 258,
                "totalTokenCount": 258,
                "promptTokensDetails": [{"modality": "IMAGE", "tokenCount": 258}],
            },
        }
        result = process_embed_content_response(
            input=["files/img123"],
            model_response=EmbeddingResponse(),
            model=self.MODEL,
            response_json=response_json,
            resolved_files={
                "files/img123": {
                    "mime_type": "image/png",
                    "uri": "https://example.com/img123",
                }
            },
        )
        assert result.usage.prompt_tokens_details.image_count == 1
        assert result.usage.prompt_tokens_details.text_tokens == 0

        prompt_cost, _ = generic_cost_per_token(
            model=self.MODEL,
            usage=result.usage,
            custom_llm_provider="vertex_ai",
        )
        assert prompt_cost == pytest.approx(0.00012)

    def test_file_reference_non_image_not_counted_as_image(self):
        """A files/... ref resolving to a non-image mime must not be image-counted."""
        response_json = {
            "embedding": {"values": [0.1, 0.2]},
            "usageMetadata": {
                "promptTokenCount": 64,
                "totalTokenCount": 64,
                "promptTokensDetails": [{"modality": "AUDIO", "tokenCount": 64}],
            },
        }
        result = process_embed_content_response(
            input=["files/clip1"],
            model_response=EmbeddingResponse(),
            model=self.MODEL,
            response_json=response_json,
            resolved_files={
                "files/clip1": {
                    "mime_type": "audio/mpeg",
                    "uri": "https://example.com/clip1",
                }
            },
        )
        assert result.usage.prompt_tokens_details.image_count == 0
        assert result.usage.prompt_tokens_details.audio_tokens == 64
        assert result.usage.prompt_tokens_details.audio_length_seconds == pytest.approx(
            2.0
        )

        prompt_cost, _ = generic_cost_per_token(
            model=self.MODEL,
            usage=result.usage,
            custom_llm_provider="vertex_ai",
        )
        assert prompt_cost == pytest.approx(2.0 * 0.00016)

    def test_video_plus_audio_does_not_double_bill_text(self):
        """Video+audio responses must not get video tokens reassigned to text."""
        response_json = {
            "embedding": {"values": [0.1]},
            "usageMetadata": {
                "promptTokenCount": 580,
                "totalTokenCount": 580,
                "promptTokensDetails": [
                    {"modality": "VIDEO", "tokenCount": 516},
                    {"modality": "AUDIO", "tokenCount": 64},
                ],
            },
        }
        result = process_embed_content_response(
            input=["gs://bucket/clip.mp4"],
            model_response=EmbeddingResponse(),
            model=self.MODEL,
            response_json=response_json,
        )
        assert result.usage.prompt_tokens_details.text_tokens == 1
        assert result.usage.prompt_tokens_details.video_length_seconds == pytest.approx(
            2.0
        )
        assert result.usage.prompt_tokens_details.audio_length_seconds == pytest.approx(
            2.0
        )

        prompt_cost, _ = generic_cost_per_token(
            model=self.MODEL,
            usage=result.usage,
            custom_llm_provider="vertex_ai",
        )
        # 1 floor text token at 2e-7 + 2s of video at 7.9e-4 + 2s of audio at 1.6e-4
        assert prompt_cost == pytest.approx(1 * 2e-7 + 2 * 0.00079 + 2 * 0.00016)
