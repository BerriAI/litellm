from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.jina_ai.rerank.transformation import JinaAIRerankConfig
from litellm.types.rerank import RerankResponse


def _transform(payload: object, status_code: int = 200) -> RerankResponse:
    return JinaAIRerankConfig().transform_rerank_response(
        model="jina-reranker-v2-base-multilingual",
        raw_response=httpx.Response(status_code, json=payload),
        model_response=RerankResponse(),
        logging_obj=MagicMock(),
    )


def test_transform_rerank_response_maps_results_and_usage():
    response = _transform(
        {
            "id": "rerank-1",
            "results": [
                {"index": 1, "relevance_score": 0.72, "document": "hello"},
                {"index": 0, "relevance_score": 0.25, "document": {"text": "world", "extra": 1}},
                {"index": 2, "relevance_score": 0, "extra": True},
            ],
            "usage": {"total_tokens": 21, "prompt_tokens": 21},
        }
    )

    assert response.id == "rerank-1"
    assert response.results == [
        {"index": 1, "relevance_score": 0.72, "document": {"text": "hello"}},
        {"index": 0, "relevance_score": 0.25, "document": {"text": "world"}},
        {"index": 2, "relevance_score": 0.0},
    ]
    assert response.meta == {"billed_units": {"total_tokens": 21}, "tokens": {}}


@pytest.mark.parametrize(
    ("payload", "expected_meta"),
    [
        ({"results": []}, {"billed_units": {}, "tokens": {}}),
        ({"results": [], "usage": {}}, {"billed_units": {}, "tokens": {}}),
        (
            {"results": [], "usage": {"total_tokens": 12, "input_tokens": 7, "output_tokens": 3, "unknown": 1}},
            {"billed_units": {"total_tokens": 12}, "tokens": {"input_tokens": 7, "output_tokens": 3}},
        ),
    ],
)
def test_transform_rerank_response_keeps_only_known_usage_counters(
    payload: dict[str, object], expected_meta: dict[str, object]
):
    assert _transform(payload).meta == expected_meta


def test_transform_rerank_response_generates_an_id_when_the_provider_sends_none():
    response = _transform({"id": None, "results": [{"index": 0, "relevance_score": 0.5}]})

    assert isinstance(response.id, str)
    assert response.id != ""


def test_transform_rerank_response_empty_results_list_yields_no_results():
    assert _transform({"id": "rerank-1", "results": []}).results == []


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "an", "object"],
        {"results": [], "usage": None},
        {"results": [], "usage": {"total_tokens": 1.5}},
        {"results": [], "usage": {"input_tokens": "many"}},
        {"results": 7},
        {"results": ["not an object"]},
        {"results": [{"index": 0, "relevance_score": 0.5}], "id": 7},
        {"results": [{"index": None, "relevance_score": 0.5}]},
    ],
)
def test_transform_rerank_response_rejects_malformed_payloads(payload: object):
    with pytest.raises(ValidationError):
        _transform(payload)


def test_transform_rerank_response_without_results_raises_value_error_naming_the_body():
    with pytest.raises(ValueError, match="No results found") as exc_info:
        _transform({"id": "rerank-1"})

    assert str(exc_info.value) == "No results found in the response={'id': 'rerank-1'}"


def test_transform_rerank_response_result_without_score_raises_key_error():
    with pytest.raises(KeyError, match="relevance_score"):
        _transform({"results": [{"index": 0}]})


def test_transform_rerank_response_non_200_raises_with_the_response_text():
    with pytest.raises(Exception, match="quota exceeded") as exc_info:
        _transform({"detail": "quota exceeded"}, status_code=429)

    assert type(exc_info.value) is Exception


def test_transform_rerank_response_shape_errors_do_not_echo_the_payload():
    with pytest.raises(ValidationError) as exc_info:
        _transform({"results": [["leaked document text"]]})

    assert "leaked document text" not in str(exc_info.value)
