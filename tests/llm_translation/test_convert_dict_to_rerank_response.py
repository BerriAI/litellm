import json
import os
import sys
from datetime import datetime

sys.path.insert(
    0, os.path.abspath("../../")
)  # Adds the parent directory to the system path


from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_rerank_response import (
    convert_dict_to_rerank_response,
)
import pytest
from litellm.types.rerank import RerankResponse


def test_convert_dict_to_rerank_response_basic():
    """Test basic conversion with all fields present."""
    response_object = {
        "id": "rerank-123",
        "results": [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.7},
        ],
        "meta": {"some_key": "some_value"},
    }
    result = convert_dict_to_rerank_response(
        model_response_object=None,
        response_object=response_object,
    )

    assert isinstance(result, RerankResponse)
    assert result.id == "rerank-123"
    assert result.results == [
        {"index": 0, "relevance_score": 0.9},
        {"index": 1, "relevance_score": 0.7},
    ]
    assert result.meta == {"some_key": "some_value"}


def test_convert_dict_to_rerank_response_with_existing_object():
    """Test conversion with an existing RerankResponse object."""
    existing_response = RerankResponse(id="existing-id", results=[])
    response_object = {
        "id": "new-id",
        "results": [{"index": 0, "relevance_score": 0.8}],
    }
    result = convert_dict_to_rerank_response(
        model_response_object=existing_response,
        response_object=response_object,
    )

    assert result.id == "new-id"
    assert result.results == [{"index": 0, "relevance_score": 0.8}]


def test_convert_dict_to_rerank_response_none_response():
    """Test error handling for None response object."""
    with pytest.raises(Exception, match="Error in response object format"):
        convert_dict_to_rerank_response(
            model_response_object=None,
            response_object=None,
        )
