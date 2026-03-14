"""
Tests for Gemini streaming grounding metadata → annotations on content-less final chunks.

When Gemini uses web search (grounding), the final streaming chunk often has
groundingMetadata but no "content" key. Previously, _process_candidates skipped
metadata extraction for content-less candidates, so annotations were lost in
streaming mode.

This fix ensures metadata is extracted BEFORE the content check, and annotations
are injected into the synthetic Delta on the final chunk.
"""

from unittest.mock import MagicMock

from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    ModelResponseIterator,
    VertexGeminiConfig,
)


def _make_logging_obj(**kwargs):
    """Create a minimal mock logging object for ModelResponseIterator."""
    logging_obj = MagicMock()
    logging_obj.optional_params = kwargs.get("optional_params", {})
    return logging_obj


def test_streaming_grounding_metadata_on_content_less_final_chunk():
    """
    When the final streaming chunk has groundingMetadata but no "content",
    annotations must still appear on the delta.

    Simulates:
      - Chunk 1: text content, no grounding
      - Chunk 2: finishReason + groundingMetadata, no content
    """
    logging_obj = _make_logging_obj()
    iterator = ModelResponseIterator(
        streaming_response=iter([]),
        sync_stream=True,
        logging_obj=logging_obj,
    )

    # Chunk 1: text content
    chunk_with_text = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "According to web sources, the answer is 42."}],
                    "role": "model",
                },
                "index": 0,
            }
        ],
    }

    # Chunk 2: finishReason + groundingMetadata, NO content
    chunk_with_grounding = {
        "candidates": [
            {
                "finishReason": "STOP",
                "index": 0,
                "groundingMetadata": {
                    "groundingChunks": [
                        {
                            "web": {
                                "uri": "https://example.com/source",
                                "title": "Example Source",
                            }
                        }
                    ],
                    "groundingSupports": [
                        {
                            "segment": {"startIndex": 0, "endIndex": 45},
                            "groundingChunkIndices": [0],
                        }
                    ],
                },
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 20,
            "candidatesTokenCount": 15,
            "totalTokenCount": 35,
        },
    }

    # Process chunk 1 (text)
    response1 = iterator.chunk_parser(chunk_with_text)
    assert response1 is not None
    assert len(response1.choices) == 1

    # Process chunk 2 (final with grounding, no content)
    response2 = iterator.chunk_parser(chunk_with_grounding)
    assert response2 is not None
    assert len(response2.choices) == 1
    assert response2.choices[0].finish_reason == "stop"

    # Annotations must be present on the delta
    delta = response2.choices[0].delta
    assert delta.annotations is not None
    assert len(delta.annotations) == 1
    assert delta.annotations[0]["type"] == "url_citation"
    assert delta.annotations[0]["url_citation"]["url"] == "https://example.com/source"
    assert delta.annotations[0]["url_citation"]["title"] == "Example Source"
    assert delta.annotations[0]["url_citation"]["start_index"] == 0
    assert delta.annotations[0]["url_citation"]["end_index"] == 45

    # Raw grounding metadata must also be set as an attribute
    assert hasattr(response2, "vertex_ai_grounding_metadata")
    assert len(response2.vertex_ai_grounding_metadata) == 1


def test_streaming_grounding_metadata_with_content():
    """
    When grounding metadata arrives on a chunk that ALSO has content,
    annotations should be set on the delta as before (existing behavior).
    """
    logging_obj = _make_logging_obj()
    iterator = ModelResponseIterator(
        streaming_response=iter([]),
        sync_stream=True,
        logging_obj=logging_obj,
    )

    # Single chunk with both content and groundingMetadata
    chunk = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "The capital of France is Paris."}],
                    "role": "model",
                },
                "index": 0,
                "finishReason": "STOP",
                "groundingMetadata": {
                    "groundingChunks": [
                        {
                            "web": {
                                "uri": "https://example.com/paris",
                                "title": "Paris Info",
                            }
                        }
                    ],
                    "groundingSupports": [
                        {
                            "segment": {"startIndex": 0, "endIndex": 30},
                            "groundingChunkIndices": [0],
                        }
                    ],
                },
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 8,
            "totalTokenCount": 18,
        },
    }

    response = iterator.chunk_parser(chunk)
    assert response is not None
    assert len(response.choices) == 1

    delta = response.choices[0].delta
    assert delta.annotations is not None
    assert len(delta.annotations) == 1
    assert delta.annotations[0]["url_citation"]["url"] == "https://example.com/paris"


def test_streaming_no_grounding_metadata_no_annotations():
    """
    When no grounding metadata is present, annotations must remain None
    on the delta (no regression).
    """
    logging_obj = _make_logging_obj()
    iterator = ModelResponseIterator(
        streaming_response=iter([]),
        sync_stream=True,
        logging_obj=logging_obj,
    )

    chunk_text = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello!"}],
                    "role": "model",
                },
                "index": 0,
            }
        ],
    }

    chunk_finish = {
        "candidates": [
            {
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 3,
            "totalTokenCount": 8,
        },
    }

    response1 = iterator.chunk_parser(chunk_text)
    assert response1 is not None

    response2 = iterator.chunk_parser(chunk_finish)
    assert response2 is not None
    assert len(response2.choices) == 1
    delta = response2.choices[0].delta
    # No annotations when no grounding metadata
    assert not hasattr(delta, "annotations") or delta.annotations is None


def test_process_candidates_extracts_metadata_from_content_less_candidates():
    """
    _process_candidates must extract grounding metadata even when a candidate
    has no "content" key. This is the core fix.
    """
    from litellm.types.utils import ModelResponseStream

    candidates = [
        {
            "finishReason": "STOP",
            "index": 0,
            "groundingMetadata": {
                "groundingChunks": [
                    {"web": {"uri": "https://test.com", "title": "Test"}}
                ],
                "groundingSupports": [
                    {
                        "segment": {"startIndex": 0, "endIndex": 10},
                        "groundingChunkIndices": [0],
                    }
                ],
            },
        }
    ]

    model_response = ModelResponseStream(choices=[])
    (
        grounding_metadata,
        url_context_metadata,
        safety_ratings,
        citation_metadata,
        _,
    ) = VertexGeminiConfig._process_candidates(
        candidates,
        model_response,
        standard_optional_params={},
    )

    # Metadata must be extracted even though candidate had no "content"
    assert len(grounding_metadata) == 1
    assert grounding_metadata[0]["groundingChunks"][0]["web"]["uri"] == "https://test.com"
