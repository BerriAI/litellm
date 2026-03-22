"""
Tests for Gemini batchEmbedContents transformation logic.

Covers:
- Text-only inputs (single and batch)
- Multimodal inputs (data URIs, GCS URLs, file references)
- Mixed text + multimodal inputs
- Response processing with correct indices
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.vertex_ai.gemini_embeddings.batch_embed_content_transformation import (
    _build_part_for_input,
    _is_multimodal_input,
    process_response,
    transform_openai_input_gemini_content,
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
        resolved = {"files/abc": {"mime_type": "image/jpeg", "uri": "https://example.com/abc"}}
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
            input=["hello", "world"], model="gemini-embedding-2-preview", optional_params={}
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
        assert result["requests"][0]["content"]["parts"][0]["text"] == "The food was delicious"
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
        assert result.usage.prompt_tokens >= 0
