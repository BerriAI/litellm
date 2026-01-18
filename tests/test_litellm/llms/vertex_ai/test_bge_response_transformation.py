"""
Test BGE response transformation validation.

This test verifies that the BGE response transformer properly validates
and handles different response formats.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../..")
)

import pytest

from litellm.llms.vertex_ai.vertex_embeddings.bge import VertexBGEConfig
from litellm.types.utils import EmbeddingResponse


def test_is_bge_model_detection():
    """
    Test BGE model detection for post-provider-split patterns.
    
    After main.py splits the provider, model strings are passed without the provider prefix.
    Model name transformation (bge/ -> numeric ID) is handled in common_utils._get_vertex_url().
    """
    # Should detect BGE models (after provider split)
    assert VertexBGEConfig.is_bge_model("bge-small-en-v1.5") is True
    assert VertexBGEConfig.is_bge_model("bge/204379420394258432") is True
    assert VertexBGEConfig.is_bge_model("BGE-large-en-v1.5") is True  # case insensitive
    
    # Should not detect non-BGE models
    assert VertexBGEConfig.is_bge_model("textembedding-gecko") is False
    assert VertexBGEConfig.is_bge_model("gemma") is False
    assert VertexBGEConfig.is_bge_model("123456789") is False


def test_bge_response_transformation_success():
    """
    Test successful BGE response transformation.
    
    Verifies that a valid BGE response is properly transformed
    to OpenAI format.
    """
    response = {
        "predictions": [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ],
        "deployedModelId": "123456",
        "model": "projects/test/models/bge-base"
    }
    
    model_response = EmbeddingResponse()
    result = VertexBGEConfig.transform_response(
        response=response,
        model="bge-small-en-v1.5",
        model_response=model_response
    )
    
    assert result.object == "list"
    assert len(result.data) == 2
    assert result.data[0]["embedding"] == [0.1, 0.2, 0.3]
    assert result.data[1]["embedding"] == [0.4, 0.5, 0.6]
    assert result.data[0]["index"] == 0
    assert result.data[1]["index"] == 1
    assert result.model == "bge-small-en-v1.5"


def test_bge_response_missing_predictions():
    """
    Test BGE response transformation with missing predictions field.
    
    Verifies that a KeyError is raised when the response doesn't
    contain the required 'predictions' field.
    """
    response = {
        "deployedModelId": "123456",
        "model": "projects/test/models/bge-base"
    }
    
    model_response = EmbeddingResponse()
    
    with pytest.raises(KeyError, match="Response missing 'predictions' field"):
        VertexBGEConfig.transform_response(
            response=response,
            model="bge-small-en-v1.5",
            model_response=model_response
        )


def test_bge_response_invalid_predictions_type():
    """
    Test BGE response transformation with invalid predictions type.
    
    Verifies that a ValueError is raised when predictions is not a list.
    """
    response = {
        "predictions": "not-a-list"
    }
    
    model_response = EmbeddingResponse()
    
    with pytest.raises(ValueError, match="Expected 'predictions' to be a list"):
        VertexBGEConfig.transform_response(
            response=response,
            model="bge-small-en-v1.5",
            model_response=model_response
        )

