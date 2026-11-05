"""
Tests for Gemini batchEmbedContents transformation logic.

Covers:
- Text-only inputs (single and batch)
- Multimodal inputs (data URIs, GCS URLs, file references)
- Mixed text + multimodal inputs
- Response processing with correct indices
"""

import json
from collections.abc import Callable
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.llms.vertex_ai.gemini_embeddings.batch_embed_content_transformation import (
    _build_part_for_input,
    file_reference_name,
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
VIDEO_DATA_URI = "data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAA"
FILES_URI = "https://generativelanguage.googleapis.com/v1beta/files/clip123"


def _file_block(**file: object) -> dict[str, object]:
    return {"type": "file", "file": file}


@pytest.fixture(autouse=True)
def _local_model_cost_map(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


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
        assert _is_multimodal_input([FILES_URI]) is True

    def test_mixed_text_and_image(self):
        assert _is_multimodal_input(["hello", IMAGE_DATA_URI]) is True

    def test_nested_text_is_not_multimodal(self):
        """Nested list with text is not multimodal."""
        assert _is_multimodal_input([["text_a", "text_b"]]) is False

    def test_file_content_block_is_multimodal(self) -> None:
        assert _is_multimodal_input([_file_block(file_data=VIDEO_DATA_URI)]) is True

    def test_nested_list_with_image_is_multimodal(self):
        assert _is_multimodal_input([["a red shoe", IMAGE_DATA_URI]]) is True

    def test_single_object_input_answers_400(self) -> None:
        with pytest.raises(BadRequestError, match="string or a list"):
            _is_multimodal_input(_file_block(file_data=VIDEO_DATA_URI))


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

    def test_file_reference_unresolved_answers_400_naming_the_gemini_provider(self) -> None:
        with pytest.raises(BadRequestError, match="gemini/ provider"):
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
        predictions: VertexAIBatchEmbeddingsResponseObject = {"embeddings": [{"values": [0.1, 0.2]}]}
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
        predictions: VertexAIBatchEmbeddingsResponseObject = {"embeddings": [{"values": [0.1, 0.2]}]}
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
        with pytest.raises(BadRequestError, match="must be strings or file content blocks, got list"):
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
        assert result.usage.prompt_tokens_details.image_tokens == 258

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

    def test_video_modality_preserves_token_count(self):
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
        assert result.usage.prompt_tokens_details.video_tokens == 516
        assert result.usage.prompt_tokens_details.text_tokens == 0

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


class TestFileContentBlocks:
    MODEL: Final = "gemini-embedding-2-preview"
    CLIP_METADATA: Final = {"fps": 2, "start_offset": "3s", "end_offset": "6s"}
    CLIP_PART: Final = {"fps": 2.0, "startOffset": "3s", "endOffset": "6s"}

    def test_data_uri_block_forwards_format_and_video_metadata(self) -> None:
        part: Final = _build_part_for_input(
            _file_block(
                file_data="data:application/octet-stream;base64,QUJD",
                format="video/mp4",
                video_metadata=self.CLIP_METADATA,
            )
        )
        assert part == {
            "inline_data": {"mime_type": "video/mp4", "data": "QUJD"},
            "video_metadata": self.CLIP_PART,
        }

    def test_gcs_block_infers_mime_type_and_forwards_video_metadata(self) -> None:
        part: Final = _build_part_for_input(
            _file_block(file_id="gs://my-bucket/clip.mp4", video_metadata={"start_offset": "0s", "end_offset": "3s"})
        )
        assert part == {
            "file_data": {"mime_type": "video/mp4", "file_uri": "gs://my-bucket/clip.mp4"},
            "video_metadata": {"startOffset": "0s", "endOffset": "3s"},
        }

    def test_file_reference_block_uses_the_resolved_file(self) -> None:
        resolved_files: Final = {"files/clip123": {"mime_type": "video/mp4", "uri": FILES_URI}}
        part: Final = _build_part_for_input(
            _file_block(file_id="files/clip123", video_metadata={"fps": 1}),
            resolved_files=resolved_files,
        )
        assert part == {
            "file_data": {"mime_type": "video/mp4", "file_uri": FILES_URI},
            "video_metadata": {"fps": 1.0},
        }

    def test_files_api_uri_block_uses_the_resolved_file(self) -> None:
        resolved_files: Final = {FILES_URI: {"mime_type": "video/mp4", "uri": FILES_URI}}
        part: Final = _build_part_for_input(
            _file_block(file_id=FILES_URI, video_metadata={"start_offset": "0s", "end_offset": "3s"}),
            resolved_files=resolved_files,
        )
        assert part == {
            "file_data": {"mime_type": "video/mp4", "file_uri": FILES_URI},
            "video_metadata": {"startOffset": "0s", "endOffset": "3s"},
        }

    @pytest.mark.parametrize("reference", ["files/clip123", FILES_URI])
    def test_file_reference_name_is_the_files_name_for_both_forms(self, reference: str) -> None:
        assert file_reference_name(reference) == "files/clip123"

    def test_block_without_video_metadata_sends_no_video_metadata_key(self) -> None:
        part: Final = _build_part_for_input(_file_block(file_data=IMAGE_DATA_URI, filename="dot.png"))
        assert part == {"inline_data": {"mime_type": "image/png", "data": IMAGE_DATA_URI.split(",", 1)[1]}}

    def test_block_with_empty_video_metadata_sends_no_video_metadata_key(self) -> None:
        part: Final = _build_part_for_input(_file_block(file_data=VIDEO_DATA_URI, video_metadata={}))
        assert part == {"inline_data": {"mime_type": "video/mp4", "data": VIDEO_DATA_URI.split(",", 1)[1]}}

    def test_batch_path_nested_block_and_text_share_one_request(self) -> None:
        result: Final = transform_openai_input_gemini_content(
            input=[[_file_block(file_data=VIDEO_DATA_URI, video_metadata=self.CLIP_METADATA), "a solid color clip"]],
            model=self.MODEL,
            optional_params={"dimensions": 768},
        )
        [request] = result["requests"]
        assert request["outputDimensionality"] == 768
        assert request["content"]["parts"] == [
            {
                "inline_data": {"mime_type": "video/mp4", "data": VIDEO_DATA_URI.split(",", 1)[1]},
                "video_metadata": self.CLIP_PART,
            },
            {"text": "a solid color clip"},
        ]

    def test_batch_path_flat_block_and_text_are_separate_requests(self) -> None:
        result: Final = transform_openai_input_gemini_content(
            input=[_file_block(file_data=VIDEO_DATA_URI, video_metadata=self.CLIP_METADATA), "a solid color clip"],
            model=self.MODEL,
            optional_params={},
        )
        assert len(result["requests"]) == 2
        assert result["requests"][0]["content"]["parts"][0]["video_metadata"] == self.CLIP_PART
        assert result["requests"][1]["content"]["parts"] == [{"text": "a solid color clip"}]

    def test_embed_content_path_accepts_flat_block_and_text(self) -> None:
        result: Final = transform_openai_input_gemini_embed_content(
            input=[_file_block(file_data=VIDEO_DATA_URI, video_metadata=self.CLIP_METADATA), "a solid color clip"],
            model=self.MODEL,
            optional_params={},
        )
        parts: Final = result["content"]["parts"]
        assert parts[0]["video_metadata"] == self.CLIP_PART
        assert parts[1] == {"text": "a solid color clip"}

    def test_embed_content_path_still_rejects_nested_lists(self) -> None:
        with pytest.raises(ValueError, match="Nested"):
            transform_openai_input_gemini_embed_content(
                input=[[_file_block(file_data=VIDEO_DATA_URI), "a solid color clip"]],
                model=self.MODEL,
                optional_params={},
            )

    @pytest.mark.parametrize(
        "block, named_in_error",
        [
            (
                _file_block(file_data=VIDEO_DATA_URI, video_metadata={"fps": 1, "startOffset": "1s"}),
                "video_metadata.startOffset",
            ),
            (_file_block(file_data=VIDEO_DATA_URI, video_metadata={"fps": "fast"}), "video_metadata.fps"),
            (_file_block(file_data=VIDEO_DATA_URI, video_metadata={"fps": "1"}), "video_metadata.fps"),
            (_file_block(file_data=VIDEO_DATA_URI, video_metadata={"fps": True}), "video_metadata.fps"),
            (_file_block(file_data=VIDEO_DATA_URI, video_metadata={"start_offset": 5}), "video_metadata.start_offset"),
            (_file_block(file_data=VIDEO_DATA_URI, detail="high"), "file.detail"),
            (_file_block(file_data=VIDEO_DATA_URI, format=""), "file.format"),
            (_file_block(file_id="gs://my-bucket/clip.mp4", file_data=VIDEO_DATA_URI), "not both"),
            (_file_block(), "needs file.file_id or file.file_data"),
            (
                _file_block(file_id="https://example.com/clip.mp4"),
                "a data: URI, a gs:// URL, a files/ reference, or a Gemini Files API URI",
            ),
            ({"type": "image_url", "image_url": {"url": IMAGE_DATA_URI}}, "Input should be 'file'"),
        ],
    )
    def test_malformed_block_answers_400_naming_the_field(
        self, block: dict[str, object], named_in_error: str
    ) -> None:
        with pytest.raises(BadRequestError, match=named_in_error):
            _build_part_for_input(block)

    def test_drop_params_drops_the_block_keys_this_surface_does_not_take(self) -> None:
        block: Final = _file_block(
            file_data=VIDEO_DATA_URI,
            detail="high",
            video_metadata={"fps": 1, "start_offset": "1s", "resolution": "low"},
        )
        part: Final = _build_part_for_input({**block, "cache_control": {"type": "ephemeral"}}, drop_params=True)
        assert part["inline_data"]["mime_type"] == "video/mp4"
        assert part["video_metadata"] == {"fps": 1.0, "startOffset": "1s"}

    def test_drop_params_still_answers_400_for_a_malformed_value(self) -> None:
        block: Final = _file_block(file_data=VIDEO_DATA_URI, detail="high", video_metadata={"fps": "fast"})
        with pytest.raises(BadRequestError, match=r"video_metadata\.fps"):
            _build_part_for_input(block, drop_params=True)

    @pytest.mark.parametrize(
        "transform", [transform_openai_input_gemini_content, transform_openai_input_gemini_embed_content]
    )
    def test_transforms_forward_drop_params_to_every_block(self, transform: Callable[..., object]) -> None:
        block: Final = _file_block(file_data=VIDEO_DATA_URI, detail="high")
        with pytest.raises(BadRequestError, match=r"file\.detail"):
            transform(input=[block], model="gemini-embedding-2-preview", optional_params={})
        transform(input=[block], model="gemini-embedding-2-preview", optional_params={}, drop_params=True)

    def test_batch_path_forwards_drop_params_into_nested_lists(self) -> None:
        block: Final = _file_block(file_data=VIDEO_DATA_URI, detail="high")
        body: Final = transform_openai_input_gemini_content(
            input=[[block, "a caption"]], model="gemini-embedding-2-preview", optional_params={}, drop_params=True
        )
        assert len(body["requests"][0]["content"]["parts"]) == 2

    @pytest.mark.parametrize(
        "transform", [transform_openai_input_gemini_content, transform_openai_input_gemini_embed_content]
    )
    def test_single_object_input_answers_400(self, transform: Callable[..., object]) -> None:
        with pytest.raises(BadRequestError, match="string or a list"):
            transform(
                input=_file_block(file_data=VIDEO_DATA_URI), model="gemini-embedding-2-preview", optional_params={}
            )

    def test_process_response_counts_only_the_text_tokens_next_to_a_block(self) -> None:
        text: Final = "a solid color clip"
        with_block: Final = process_response(
            input=[_file_block(file_data=VIDEO_DATA_URI, video_metadata=self.CLIP_METADATA), text],
            model_response=EmbeddingResponse(),
            model=self.MODEL,
            _predictions={"embeddings": [{"values": [0.1]}, {"values": [0.2]}]},
        )
        text_only: Final = process_response(
            input=[text],
            model_response=EmbeddingResponse(),
            model=self.MODEL,
            _predictions={"embeddings": [{"values": [0.2]}]},
        )
        assert with_block.usage.prompt_tokens == text_only.usage.prompt_tokens > 0

    def test_embed_content_usage_fallback_with_a_block_does_not_estimate(self) -> None:
        result: Final = process_embed_content_response(
            input=[_file_block(file_data=VIDEO_DATA_URI, video_metadata=self.CLIP_METADATA)],
            model_response=EmbeddingResponse(),
            model=self.MODEL,
            response_json={"embedding": {"values": [0.1, 0.2]}},
        )
        assert result.usage.prompt_tokens == 0

    def test_image_block_counts_as_image_only_input(self) -> None:
        result: Final = process_embed_content_response(
            input=[_file_block(file_data=IMAGE_DATA_URI)],
            model_response=EmbeddingResponse(),
            model=self.MODEL,
            response_json={
                "embedding": {"values": [0.1, 0.2]},
                "usageMetadata": {"promptTokenCount": 258, "totalTokenCount": 258},
            },
        )
        assert result.usage.prompt_tokens_details.image_tokens == 258


@pytest.mark.respx(assert_all_called=True)
def test_gemini_embedding(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents"
    ).mock(return_value=httpx.Response(200, json={"embeddings": [{"values": [0.0123, -0.0456, 0.0789]}]}))

    response: Final = litellm.embedding(
        model="gemini/gemini-embedding-001",
        input="Hello, world!",
        api_key="gemini-test-key",
    )

    assert json.loads(route.calls.last.request.content) == {
        "requests": [{"model": "models/gemini-embedding-001", "content": {"parts": [{"text": "Hello, world!"}]}}]
    }
    assert response.model == "gemini-embedding-001"
    assert [(item.index, item.embedding) for item in response.data] == [(0, [0.0123, -0.0456, 0.0789])]
    expected_prompt_tokens: Final = litellm.token_counter(model="gemini/gemini-embedding-001", text="Hello, world!")
    assert response.usage.prompt_tokens == expected_prompt_tokens
    assert response.usage.total_tokens == expected_prompt_tokens
