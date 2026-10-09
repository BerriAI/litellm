from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.cohere.rerank.transformation import CohereRerankConfig
from litellm.types.rerank import RerankResponse


def _transform(payload: object) -> RerankResponse:
    return CohereRerankConfig().transform_rerank_response(
        model="rerank-english-v3.0",
        raw_response=httpx.Response(200, json=payload),
        model_response=RerankResponse(),
        logging_obj=Mock(),
    )


def test_transform_rerank_response_keeps_the_cohere_payload():
    response = _transform(
        {
            "id": "rerank-1",
            "results": [{"index": 1, "relevance_score": 0.9, "document": {"text": "sensitive-document"}}],
            "meta": {"billed_units": {"search_units": 1}, "tokens": {"input_tokens": 4}},
            "warnings": ["ignored"],
        }
    )

    assert response.model_dump() == {
        "id": "rerank-1",
        "results": [{"index": 1, "relevance_score": 0.9, "document": {"text": "sensitive-document"}}],
        "meta": {"billed_units": {"search_units": 1}, "tokens": {"input_tokens": 4}},
    }


def test_transform_rerank_response_of_an_empty_object_has_no_results():
    assert _transform({}).model_dump() == {"id": None, "results": None, "meta": None}


@pytest.mark.parametrize("payload", [7, "sensitive-document", [{"id": "sensitive-document"}]])
def test_transform_rerank_response_rejects_non_object_bodies(payload: object):
    with pytest.raises(ValidationError) as exc_info:
        _transform(payload)

    assert "sensitive-document" not in str(exc_info.value)


@pytest.mark.parametrize("payload", [{"id": 7}, {"results": "not-a-list"}, {"results": [{"index": 0}]}, {"meta": []}])
def test_transform_rerank_response_rejects_malformed_fields(payload: dict[str, object]):
    with pytest.raises(ValidationError):
        _transform(payload)


def test_map_cohere_rerank_params_returns_every_cohere_param():
    params = CohereRerankConfig().map_cohere_rerank_params(
        non_default_params=None,
        model="rerank-english-v3.0",
        drop_params=False,
        query="capital of france",
        documents=["Paris", {"text": "Berlin"}],
        top_n=1,
    )

    assert params == {
        "query": "capital of france",
        "documents": ["Paris", {"text": "Berlin"}],
        "top_n": 1,
        "rank_fields": None,
        "return_documents": True,
        "max_chunks_per_doc": None,
    }
