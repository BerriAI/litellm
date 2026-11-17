"""
Tests for Gemini batchEmbedContents transformation logic.

Covers:
- Text-only inputs (single and batch)
- Multimodal inputs (data URIs, GCS URLs, file references)
- Mixed text + multimodal inputs
- Response processing with correct indices
"""

import asyncio
import json
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.caching.caching import Cache
from litellm.caching.caching_handler import _PENDING_CACHE_WRITES
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
from litellm.types.caching import LiteLLMCacheType
from litellm.types.llms.vertex_ai import GeminiEmbeddingInput, VertexAIBatchEmbeddingsResponseObject
from litellm.types.utils import EmbeddingResponse, Usage

IMAGE_DATA_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAIAQMAAAD+wSzIAAAABlBMVEX///+/v7+jQ3Y5AAAADklEQVQI12P4AIX8EAgALgAD/aNpbtEAAAAASUVORK5CYII"
GCS_URL = "gs://my-bucket/image.png"
VIDEO_DATA_URI = "data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAA"
AUDIO_DATA_URI = "data:audio/mpeg;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA"
FILES_URI = "https://generativelanguage.googleapis.com/v1beta/files/clip123"


def _prompt_cost(model: str, usage: Usage) -> float:
    cost, _ = generic_cost_per_token(model=model, usage=usage, custom_llm_provider="vertex_ai")
    return cost


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

    @pytest.mark.parametrize("details_key", ["promptTokensDetails", "promptTokenDetails"])
    def test_multimodal_image_preserves_usage_metadata(self, details_key: str):
        response_json = {
            "embedding": {"values": [0.1, 0.2, 0.3]},
            "usageMetadata": {
                "promptTokenCount": 258,
                "totalTokenCount": 258,
                details_key: [{"modality": "IMAGE", "tokenCount": 258}],
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

    def test_document_modality_is_billed_as_image_tokens(self):
        """PDF usage arrives as DOCUMENT tokens, which Google bills at the image token rate.

        https://ai.google.dev/gemini-api/docs/pricing "Document token billing", read 2026-11-13.
        """

        def usage_for(details: tuple[dict[str, object], ...], prompt_tokens: int) -> Usage:
            return process_embed_content_response(
                input=[{"type": "file", "file": {"file_data": "gs://bucket/invoice.pdf", "format": "application/pdf"}}],
                model_response=EmbeddingResponse(),
                model=self.MODEL,
                response_json={
                    "embedding": {"values": [0.1, 0.2]},
                    "usageMetadata": {
                        "promptTokenCount": prompt_tokens,
                        "totalTokenCount": prompt_tokens,
                        "promptTokensDetails": list(details),
                    },
                },
            ).usage

        document_usage: Final = usage_for(
            ({"modality": "TEXT", "tokenCount": 12}, {"modality": "DOCUMENT", "tokenCount": 258}), 270
        )
        image_usage: Final = usage_for(
            ({"modality": "TEXT", "tokenCount": 12}, {"modality": "IMAGE", "tokenCount": 258}), 270
        )

        assert document_usage.prompt_tokens == 270
        assert document_usage.prompt_tokens_details.text_tokens == 12
        assert document_usage.prompt_tokens_details.image_tokens == 258
        assert _prompt_cost(self.MODEL, document_usage) == _prompt_cost(self.MODEL, image_usage) > 0

    def test_unrecognized_modality_keeps_usage_and_bills_as_text(self):
        def usage_for(modality: str) -> Usage:
            return process_embed_content_response(
                input=["gs://bucket/clip.mp4"],
                model_response=EmbeddingResponse(),
                model=self.MODEL,
                response_json={
                    "embedding": {"values": [0.1]},
                    "usageMetadata": {
                        "promptTokenCount": 112,
                        "totalTokenCount": 112,
                        "promptTokensDetails": [
                            {"modality": "TEXT", "tokenCount": 12},
                            {"modality": modality, "tokenCount": 100},
                        ],
                    },
                },
            ).usage

        unknown_usage: Final = usage_for("MODALITY_UNSPECIFIED")

        assert unknown_usage.prompt_tokens == 112
        assert unknown_usage.prompt_tokens_details.text_tokens == 112
        assert _prompt_cost(self.MODEL, unknown_usage) == _prompt_cost(self.MODEL, usage_for("TEXT")) > 0

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


@pytest.mark.parametrize(
    ("media_input", "modality", "media_price_key", "media_details_field"),
    [
        (IMAGE_DATA_URI, "IMAGE", "input_cost_per_image_token", "image_tokens"),
        (
            {"type": "file", "file": {"file_data": VIDEO_DATA_URI, "video_metadata": {"end_offset": "5s"}}},
            "VIDEO",
            "input_cost_per_video_token",
            "video_tokens",
        ),
    ],
)
@pytest.mark.respx(assert_all_called=True)
def test_gemini_batch_embedding_usage_comes_from_usage_metadata(
    respx_mock: respx.MockRouter,
    media_input: str | dict[str, object],
    modality: str,
    media_price_key: str,
    media_details_field: str,
) -> None:
    respx_mock.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "embeddings": [{"values": [0.1, 0.2]}, {"values": [0.3, 0.4]}],
                "usageMetadata": {
                    "promptTokenCount": 334,
                    "promptTokenDetails": [
                        {"modality": "TEXT", "tokenCount": 4},
                        {"modality": modality, "tokenCount": 330},
                    ],
                },
            },
        )
    )

    response: Final = litellm.embedding(
        model="gemini/gemini-embedding-2",
        input=["a red apple", media_input],
        api_key="gemini-test-key",
    )

    assert response.usage.prompt_tokens == 334
    assert response.usage.total_tokens == 334
    assert response.usage.prompt_tokens_details.text_tokens == 4
    assert getattr(response.usage.prompt_tokens_details, media_details_field) == 330
    prices: Final = litellm.model_cost["gemini/gemini-embedding-2"]
    assert litellm.completion_cost(completion_response=response) == pytest.approx(
        4 * prices["input_cost_per_token"] + 330 * prices[media_price_key]
    )


@pytest.mark.parametrize(
    "usage_metadata",
    [
        {},
        {"promptTokenDetails": [{"modality": "TEXT", "tokenCount": 3}, {"modality": "IMAGE", "tokenCount": 258}]},
    ],
)
def test_gemini_batch_usage_metadata_without_a_prompt_count_bills_like_none_was_sent(
    usage_metadata: Mapping[str, object],
) -> None:
    def usage_for(predictions: VertexAIBatchEmbeddingsResponseObject) -> Usage:
        return process_response(
            input=["a red apple", IMAGE_DATA_URI],
            model_response=EmbeddingResponse(),
            model="gemini-embedding-2",
            _predictions=predictions,
        ).usage

    embeddings: Final = [{"values": [0.1]}, {"values": [0.2]}]
    with_metadata: Final = usage_for({"embeddings": embeddings, "usageMetadata": usage_metadata})
    without_metadata: Final = usage_for({"embeddings": embeddings})

    assert with_metadata.prompt_tokens == without_metadata.prompt_tokens > 0
    assert _prompt_cost("gemini-embedding-2", with_metadata) == _prompt_cost("gemini-embedding-2", without_metadata)


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.respx(assert_all_called=True)
async def test_gemini_batch_embedding_bills_a_files_image_at_the_image_rate_without_modality_details(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, use_async: bool
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.get("https://generativelanguage.googleapis.com/v1beta/files/abc123").mock(
        return_value=httpx.Response(
            200,
            json={"mimeType": "image/png", "uri": "https://generativelanguage.googleapis.com/v1beta/files/abc123"},
        )
    )
    respx_mock.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents"
    ).mock(
        return_value=httpx.Response(
            200, json={"embeddings": [{"values": [0.1, 0.2]}], "usageMetadata": {"promptTokenCount": 258}}
        )
    )

    response: Final = (
        await litellm.aembedding(model="gemini/gemini-embedding-2", input=["files/abc123"], api_key="gemini-test-key")
        if use_async
        else litellm.embedding(model="gemini/gemini-embedding-2", input=["files/abc123"], api_key="gemini-test-key")
    )

    assert response.usage.prompt_tokens_details.image_tokens == 258
    assert litellm.completion_cost(completion_response=response) == pytest.approx(
        258 * litellm.model_cost["gemini/gemini-embedding-2"]["input_cost_per_image_token"]
    )


def test_gemini_batch_embedding_adds_nothing_to_the_logged_hidden_params() -> None:
    response: Final = process_response(
        input=["a red apple", IMAGE_DATA_URI],
        model_response=EmbeddingResponse(hidden_params={"custom_llm_provider": "gemini"}),
        model="gemini-embedding-2",
        _predictions={
            "embeddings": [{"values": [0.1]}, {"values": [0.2]}],
            "usageMetadata": {
                "promptTokenCount": 261,
                "promptTokenDetails": [{"modality": "TEXT", "tokenCount": 3}, {"modality": "IMAGE", "tokenCount": 258}],
            },
        },
    )

    assert response.hidden_params == {"custom_llm_provider": "gemini"}
    assert len(response.item_usage) == 2


def test_gemini_batch_embedding_gives_a_read_only_file_block_its_own_image_tokens() -> None:
    response: Final = process_response(
        input=["a red apple", MappingProxyType(_file_block(file_data=IMAGE_DATA_URI))],
        model_response=EmbeddingResponse(),
        model="gemini-embedding-2",
        _predictions={
            "embeddings": [{"values": [0.1]}, {"values": [0.2]}],
            "usageMetadata": {
                "promptTokenCount": 261,
                "promptTokenDetails": [{"modality": "TEXT", "tokenCount": 3}, {"modality": "IMAGE", "tokenCount": 258}],
            },
        },
    )

    assert tuple(
        (usage.prompt_tokens_details.text_tokens, usage.prompt_tokens_details.image_tokens)
        for usage in response.item_usage
    ) == ((3, 0), (0, 258))


@pytest.mark.parametrize(
    ("input", "usage_metadata", "expected"),
    [
        pytest.param(
            [IMAGE_DATA_URI, _file_block(file_data=IMAGE_DATA_URI)],
            {
                "promptTokenCount": 520,
                "promptTokenDetails": [{"modality": "TEXT", "tokenCount": 4}, {"modality": "IMAGE", "tokenCount": 516}],
            },
            ((260, 2, 258), (260, 2, 258)),
            id="images-only",
        ),
        pytest.param(
            ["hi", "a red apple"],
            {"promptTokenCount": 3, "promptTokenDetails": [{"modality": "TEXT", "tokenCount": 13}]},
            ((2, 2, 0), (11, 11, 0)),
            id="details-above-prompt-count",
        ),
        pytest.param(
            ["a red apple", VIDEO_DATA_URI],
            {
                "promptTokenCount": 11,
                "promptTokenDetails": [{"modality": "TEXT", "tokenCount": 1}, {"modality": "VIDEO", "tokenCount": 10}],
            },
            ((1, 1, 0), (10, 0, 0)),
            id="text-count-below-the-local-count",
        ),
        pytest.param(
            [VIDEO_DATA_URI, VIDEO_DATA_URI],
            {
                "promptTokenCount": 24,
                "promptTokenDetails": [{"modality": "TEXT", "tokenCount": 4}, {"modality": "VIDEO", "tokenCount": 20}],
            },
            ((12, 2, 0), (12, 2, 0)),
            id="videos-only",
        ),
        pytest.param(
            ["a red apple\ud83d", "a green pear"],
            {"promptTokenCount": 7, "promptTokenDetails": [{"modality": "TEXT", "tokenCount": 7}]},
            ((4, 4, 0), (3, 3, 0)),
            id="lone-surrogate",
        ),
    ],
)
def test_gemini_batch_embedding_gives_each_input_a_non_negative_share(
    input: GeminiEmbeddingInput,
    usage_metadata: Mapping[str, object],
    expected: tuple[tuple[int, int, int], ...],
) -> None:
    response: Final = process_response(
        input=input,
        model_response=EmbeddingResponse(),
        model="gemini-embedding-2",
        _predictions={"embeddings": [{"values": [0.1]}, {"values": [0.2]}], "usageMetadata": dict(usage_metadata)},
    )

    assert tuple(
        (usage.prompt_tokens, usage.prompt_tokens_details.text_tokens, usage.prompt_tokens_details.image_tokens)
        for usage in response.item_usage
    ) == expected


@pytest.mark.parametrize(
    ("texts", "media", "audio", "video"),
    [
        pytest.param(("a red apple",), VIDEO_DATA_URI, 125, 330, id="text-and-video"),
        pytest.param(("a red apple",), AUDIO_DATA_URI, 125, 0, id="text-and-audio"),
        pytest.param(("a red apple", "一个红苹果从树上掉下来了"), VIDEO_DATA_URI, 125, 330, id="two-texts-and-video"),
    ],
)
def test_gemini_batch_embedding_keeps_each_texts_own_tokens_and_gives_the_rest_to_the_media(
    texts: tuple[str, ...], media: str, audio: int, video: int
) -> None:
    text: Final = 53
    details: Final = (("TEXT", text), ("AUDIO", audio), ("VIDEO", video))
    response: Final = process_response(
        input=[*texts, media],
        model_response=EmbeddingResponse(),
        model="gemini-embedding-2",
        _predictions={
            "embeddings": [{"values": [0.1]} for _ in (*texts, media)],
            "usageMetadata": {
                "promptTokenCount": text + audio + video,
                "promptTokenDetails": [
                    {"modality": modality, "tokenCount": count} for modality, count in details if count
                ],
            },
        },
    )
    local: Final = tuple(litellm.token_counter(model="gemini-embedding-2", text=item) for item in texts)

    assert tuple(
        (
            usage.prompt_tokens,
            usage.prompt_tokens_details.text_tokens,
            usage.prompt_tokens_details.audio_tokens,
            usage.prompt_tokens_details.video_tokens,
        )
        for usage in response.item_usage
    ) == (
        *((count, count, 0, 0) for count in local),
        (text + audio + video - sum(local), text - sum(local), audio, video),
    )


def test_gemini_batch_embedding_without_usage_metadata_splits_the_local_count_by_text_length() -> None:
    long_text: Final = " ".join(("a red apple",) * 200)
    response: Final = process_response(
        input=["hello", long_text],
        model_response=EmbeddingResponse(),
        model="gemini-embedding-001",
        _predictions={"embeddings": [{"values": [0.1]}, {"values": [0.2]}]},
    )
    short, long = response.item_usage

    assert short.prompt_tokens + long.prompt_tokens == response.usage.prompt_tokens
    assert short.prompt_tokens * 100 < long.prompt_tokens


async def _cache_hits(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    batch: tuple[str, ...],
    usage_metadata: Mapping[str, object],
    hit_inputs: tuple[str, ...],
    embed_params: Mapping[str, object] = MappingProxyType({}),
) -> tuple[int, tuple[EmbeddingResponse, ...]]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))
    route: Final = respx_mock.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents"
    ).mock(
        return_value=httpx.Response(
            200,
            json={"embeddings": [{"values": [0.1, 0.2]} for _ in batch], "usageMetadata": dict(usage_metadata)},
        )
    )

    async def embed(input: list[str]) -> EmbeddingResponse:
        return await litellm.aembedding(
            model="gemini/gemini-embedding-2", input=input, api_key="gemini-test-key", **embed_params
        )

    await embed(list(batch))
    await asyncio.gather(*_PENDING_CACHE_WRITES)
    hits: Final = tuple([await embed([item]) for item in hit_inputs])
    return route.call_count, hits


@pytest.mark.asyncio
@pytest.mark.respx(assert_all_called=True)
@pytest.mark.parametrize(
    ("batch", "usage_metadata", "expected_hits"),
    [
        pytest.param(
            ("a red apple", IMAGE_DATA_URI, "a green pear"),
            {
                "promptTokenCount": 267,
                "promptTokenDetails": [
                    {"modality": "TEXT", "tokenCount": 9},
                    {"modality": "IMAGE", "tokenCount": 258},
                ],
            },
            (("a red apple", (4, 4, 0)), (IMAGE_DATA_URI, (258, 0, 258))),
            id="modality-details",
        ),
        pytest.param(
            ("a red apple", "a green pear"),
            {"promptTokenCount": 7},
            (("a red apple", (3, 0, 0)), ("a green pear", (4, 0, 0))),
            id="prompt-count-only",
        ),
        pytest.param(
            ("a red apple", IMAGE_DATA_URI),
            {"promptTokenCount": 300},
            (("a red apple", (150, 0, 0)), (IMAGE_DATA_URI, (150, 0, 0))),
            id="media-without-modality-details",
        ),
    ],
)
async def test_gemini_batch_embedding_cache_hits_report_each_inputs_own_modality_tokens(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    batch: tuple[str, ...],
    usage_metadata: Mapping[str, object],
    expected_hits: tuple[tuple[str, tuple[int, int, int]], ...],
) -> None:
    calls, hits = await _cache_hits(
        respx_mock, monkeypatch, batch, usage_metadata, tuple(item for item, _ in expected_hits)
    )

    assert calls == 1
    assert tuple(
        (
            item,
            (
                hit.usage.prompt_tokens,
                hit.usage.prompt_tokens_details.text_tokens,
                hit.usage.prompt_tokens_details.image_tokens,
            ),
        )
        for (item, _), hit in zip(expected_hits, hits, strict=True)
    ) == expected_hits


@pytest.mark.asyncio
@pytest.mark.respx(assert_all_called=True)
@pytest.mark.parametrize(
    "usage_metadata",
    [
        pytest.param(
            {"promptTokenCount": 601, "promptTokenDetails": [{"modality": "TEXT", "tokenCount": 601}]},
            id="modality-details",
        ),
        pytest.param({"promptTokenCount": 601}, id="prompt-count-only"),
    ],
)
async def test_gemini_batch_embedding_cache_hits_split_text_tokens_by_each_inputs_length(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, usage_metadata: Mapping[str, object]
) -> None:
    long_text: Final = " ".join(("a red apple",) * 200)
    calls, (short_hit, long_hit) = await _cache_hits(
        respx_mock, monkeypatch, ("hello", long_text), usage_metadata, ("hello", long_text)
    )

    assert calls == 1
    assert short_hit.usage.prompt_tokens + long_hit.usage.prompt_tokens == 601
    assert short_hit.usage.prompt_tokens * 100 < long_hit.usage.prompt_tokens


@pytest.mark.asyncio
@pytest.mark.respx(assert_all_called=True)
async def test_gemini_batch_embedding_cache_hits_give_more_text_tokens_to_the_script_that_takes_more_tokens(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    english: Final = "a red apple"
    chinese: Final = "一个红苹果一个红苹果一"
    calls, (english_hit, chinese_hit) = await _cache_hits(
        respx_mock,
        monkeypatch,
        (english, chinese),
        {"promptTokenCount": 12, "promptTokenDetails": [{"modality": "TEXT", "tokenCount": 12}]},
        (english, chinese),
    )

    assert calls == 1
    assert english_hit.usage.prompt_tokens + chinese_hit.usage.prompt_tokens == 12
    assert english_hit.usage.prompt_tokens < chinese_hit.usage.prompt_tokens


@pytest.mark.asyncio
@pytest.mark.respx(assert_all_called=True)
@pytest.mark.parametrize(
    ("audio_track_extraction", "expected_audio_tokens"),
    [
        pytest.param(True, (100, 100), id="video-audio-track-extracted"),
        pytest.param(False, (200, 0), id="video-audio-track-ignored"),
    ],
)
async def test_gemini_batch_embedding_cache_hits_give_a_videos_extracted_audio_tokens_to_the_video(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    audio_track_extraction: bool,
    expected_audio_tokens: tuple[int, int],
) -> None:
    calls, hits = await _cache_hits(
        respx_mock,
        monkeypatch,
        (AUDIO_DATA_URI, VIDEO_DATA_URI),
        {
            "promptTokenCount": 530,
            "promptTokenDetails": [{"modality": "AUDIO", "tokenCount": 200}, {"modality": "VIDEO", "tokenCount": 330}],
        },
        (AUDIO_DATA_URI, VIDEO_DATA_URI),
        {"audio_track_extraction": audio_track_extraction},
    )

    assert calls == 1
    assert tuple(hit.usage.prompt_tokens_details.audio_tokens for hit in hits) == expected_audio_tokens


@pytest.mark.parametrize(
    ("embed_params", "expected_audio_tokens"),
    [
        pytest.param({"audio_track_extraction": True}, (100, 100), id="named-param"),
        pytest.param({"audio_track_extraction": "true"}, (100, 100), id="named-param-as-a-string"),
        pytest.param({"audio_track_extraction": "false"}, (200, 0), id="named-param-false-as-a-string"),
        pytest.param({"audio_track_extraction": "Yes"}, (100, 100), id="named-param-yes"),
        pytest.param({"audio_track_extraction": "y"}, (100, 100), id="named-param-y"),
        pytest.param({"audio_track_extraction": "T"}, (100, 100), id="named-param-t"),
        pytest.param({"audio_track_extraction": "no"}, (200, 0), id="named-param-no"),
        pytest.param(
            {"extra_body": {"embedContentConfig": {"audioTrackExtraction": True}}}, (100, 100), id="extra-body"
        ),
        pytest.param(
            {"extra_body": {"embedContentConfig": {"audioTrackExtraction": 1.0}}},
            (100, 100),
            id="extra-body-float-one",
        ),
        pytest.param(
            {"extra_body": {"embedContentConfig": {"audio_track_extraction": 1}}},
            (100, 100),
            id="extra-body-snake-case-field",
        ),
        pytest.param(
            {"extra_body": {"embed_content_config": {"audioTrackExtraction": "TRUE"}}},
            (100, 100),
            id="extra-body-snake-case-config",
        ),
        pytest.param(
            {"audio_track_extraction": False, "extra_body": {"embedContentConfig": {"audioTrackExtraction": True}}},
            (200, 0),
            id="named-param-off-over-extra-body",
        ),
        pytest.param({}, (200, 0), id="not-set"),
    ],
)
def test_gemini_batch_embedding_gives_a_videos_extracted_audio_tokens_to_the_video_on_sync_calls(
    respx_mock: respx.MockRouter,
    embed_params: Mapping[str, object],
    expected_audio_tokens: tuple[int, int],
) -> None:
    respx_mock.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "embeddings": [{"values": [0.1, 0.2]}, {"values": [0.3, 0.4]}],
                "usageMetadata": {
                    "promptTokenCount": 530,
                    "promptTokenDetails": [
                        {"modality": "AUDIO", "tokenCount": 200},
                        {"modality": "VIDEO", "tokenCount": 330},
                    ],
                },
            },
        )
    )

    response: Final = litellm.embedding(
        model="gemini/gemini-embedding-2",
        input=[AUDIO_DATA_URI, VIDEO_DATA_URI],
        api_key="gemini-test-key",
        **embed_params,
    )

    assert tuple(usage.prompt_tokens_details.audio_tokens for usage in response.item_usage) == expected_audio_tokens


def test_gemini_batch_embedding_sends_an_embed_content_config_with_non_string_keys_to_google(
    respx_mock: respx.MockRouter,
) -> None:
    route: Final = respx_mock.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents"
    ).mock(
        return_value=httpx.Response(
            400, json={"error": {"code": 400, "message": "Invalid JSON payload received.", "status": "INVALID_ARGUMENT"}}
        )
    )

    with pytest.raises(BadRequestError, match="Invalid JSON payload received"):
        litellm.embedding(
            model="gemini/gemini-embedding-2",
            input=["hello"],
            api_key="gemini-test-key",
            extra_body={"embedContentConfig": {1: True}},
        )

    assert route.call_count == 1


@pytest.mark.parametrize("use_embed_content", [False, True])
def test_a_usage_detail_entry_without_a_token_count_keeps_googles_other_counts(use_embed_content: bool) -> None:
    usage_metadata: Final = {
        "promptTokenCount": 261,
        "promptTokenDetails": [
            {"modality": "TEXT", "tokenCount": 3},
            {"modality": "AUDIO"},
            {"modality": "IMAGE", "tokenCount": 258},
        ],
    }
    response: Final = (
        process_embed_content_response(
            input=["a red apple", IMAGE_DATA_URI],
            model_response=EmbeddingResponse(),
            model="gemini-embedding-2",
            response_json={"embedding": {"values": [0.1]}, "usageMetadata": usage_metadata},
        )
        if use_embed_content
        else process_response(
            input=["a red apple", IMAGE_DATA_URI],
            model_response=EmbeddingResponse(),
            model="gemini-embedding-2",
            _predictions={"embeddings": [{"values": [0.1]}, {"values": [0.2]}], "usageMetadata": usage_metadata},
        )
    )

    assert (
        response.usage.prompt_tokens,
        response.usage.prompt_tokens_details.text_tokens,
        response.usage.prompt_tokens_details.image_tokens,
    ) == (261, 3, 258)


class TestEmbedContentConfigParams:
    """`document_ocr`, `audio_track_extraction` and `auto_truncate` only exist inside `embedContentConfig`."""

    @pytest.mark.parametrize(
        "optional_params",
        [
            {"document_ocr": True, "audio_track_extraction": True, "auto_truncate": False},
            {"documentOcr": True, "audioTrackExtraction": True, "autoTruncate": False},
        ],
    )
    def test_embed_content_request_nests_the_named_params(self, optional_params: dict[str, object]) -> None:
        request: Final = transform_openai_input_gemini_embed_content(
            input=["hello"], model="gemini-embedding-2", optional_params={**optional_params, "dimensions": 256}
        )

        assert request == {
            "content": {"parts": [{"text": "hello"}]},
            "outputDimensionality": 256,
            "embedContentConfig": {"documentOcr": True, "audioTrackExtraction": True, "autoTruncate": False},
        }

    def test_batch_embed_contents_nests_the_named_params_in_every_request(self) -> None:
        body: Final = transform_openai_input_gemini_content(
            input=["first", "second"], model="gemini-embedding-2", optional_params={"document_ocr": True}
        )

        assert [request["embedContentConfig"] for request in body["requests"]] == [{"documentOcr": True}] * 2

    def test_named_params_merge_over_an_extra_body_config(self) -> None:
        request: Final = transform_openai_input_gemini_embed_content(
            input=["hello"],
            model="gemini-embedding-2",
            optional_params={
                "document_ocr": True,
                "extra_body": {"embedContentConfig": {"documentOcr": False, "autoTruncate": False}},
            },
        )

        assert request["embedContentConfig"] == {"documentOcr": True, "autoTruncate": False}

    def test_requests_without_the_named_params_carry_no_config(self) -> None:
        request: Final = transform_openai_input_gemini_embed_content(
            input=["hello"], model="gemini-embedding-2", optional_params={"document_ocr": None, "dimensions": 256}
        )

        assert "embedContentConfig" not in request


def test_gemini_embedding_sends_extra_body_fields_on_each_request(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents"
    ).mock(return_value=httpx.Response(200, json={"embeddings": [{"values": [0.1, 0.2]}]}))

    litellm.embedding(
        model="gemini/gemini-embedding-001",
        input="Hello, world!",
        api_key="gemini-test-key",
        extra_body={
            "title": "Greeting",
            "taskType": "RETRIEVAL_DOCUMENT",
            "outputDimensionality": 256,
            "documentOcr": True,
        },
    )

    assert json.loads(route.calls.last.request.content)["requests"] == [
        {
            "model": "models/gemini-embedding-001",
            "content": {"parts": [{"text": "Hello, world!"}]},
            "title": "Greeting",
            "taskType": "RETRIEVAL_DOCUMENT",
            "outputDimensionality": 256,
            "documentOcr": True,
        }
    ]
