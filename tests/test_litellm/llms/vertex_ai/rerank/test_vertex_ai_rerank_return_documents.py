"""
Regression tests for Vertex AI rerank return_documents behavior.

Issue: the response transformer read only `id` and `score` from each Vertex
record and silently discarded `content`, so `results[i].document.text` was
always absent even when the caller passed return_documents=True (the default).

The request side already sets `ignoreRecordDetailsInResponse = not
return_documents`, so when return_documents=True Vertex returns `content` on
every record. These tests verify the response transformer now surfaces that
content as `document.text`, and omits the field when Vertex returns IDs only.
"""

import json
from unittest.mock import MagicMock

import httpx

from litellm.llms.vertex_ai.rerank.transformation import VertexAIRerankConfig
from litellm.types.rerank import RerankResponse


def _mock_response(response_data: dict) -> MagicMock:
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json.return_value = response_data
    mock_response.text = json.dumps(response_data)
    return mock_response


def _transform(response_data: dict) -> RerankResponse:
    config = VertexAIRerankConfig()
    return config.transform_rerank_response(
        model="semantic-ranker-default@latest",
        raw_response=_mock_response(response_data),
        model_response=RerankResponse(),
        logging_obj=MagicMock(),
    )


def test_vertex_rerank_return_documents_true_populates_document_text():
    """return_documents=True: Vertex returns content, it must reach document.text."""
    response_data = {
        "records": [
            {
                "id": "1",
                "score": 0.95,
                "title": "doc 1 title",
                "content": "doc 1",
            },
            {
                "id": "0",
                "score": 0.42,
                "title": "doc 0 title",
                "content": "doc 0",
            },
        ]
    }

    result = _transform(response_data)

    assert len(result.results) == 2
    assert result.results[0]["index"] == 1
    assert result.results[0]["relevance_score"] == 0.95
    assert result.results[0]["document"]["text"] == "doc 1"
    assert result.results[1]["document"]["text"] == "doc 0"


def test_vertex_rerank_return_documents_false_omits_document():
    """return_documents=False: Vertex returns IDs only, no document field."""
    response_data = {"records": [{"id": "1"}, {"id": "0"}]}

    result = _transform(response_data)

    assert len(result.results) == 2
    assert result.results[0]["index"] == 1
    assert "document" not in result.results[0]
    assert "document" not in result.results[1]


def test_vertex_rerank_record_without_content_omits_document():
    """A record that lacks content (edge case) must not gain an empty document."""
    response_data = {"records": [{"id": "2", "score": 0.7}]}

    result = _transform(response_data)

    assert len(result.results) == 1
    assert result.results[0]["index"] == 2
    assert result.results[0]["relevance_score"] == 0.7
    assert "document" not in result.results[0]
