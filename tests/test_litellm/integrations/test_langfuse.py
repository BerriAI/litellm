import json
import os
import sys
from typing import Optional

# Adds the grandparent directory to sys.path to allow importing project modules
sys.path.insert(0, os.path.abspath("../.."))

import asyncio
from unittest.mock import patch

import pytest

import litellm
from litellm.integrations.langfuse import langfuse as langfuse_module
from litellm.integrations.langfuse.langfuse import LangFuseLogger


def test_max_langfuse_clients_limit():
    """
    Test that the max langfuse clients limit is respected when initializing multiple clients
    """
    # Set max clients to 2 for testing
    with patch.object(langfuse_module, "MAX_LANGFUSE_INITIALIZED_CLIENTS", 2):
        # Reset the counter
        litellm.initialized_langfuse_clients = 0

        # First client should succeed
        logger1 = LangFuseLogger(
            langfuse_public_key="test_key_1",
            langfuse_secret="test_secret_1",
            langfuse_host="https://test1.langfuse.com",
        )
        assert litellm.initialized_langfuse_clients == 1

        # Second client should succeed
        logger2 = LangFuseLogger(
            langfuse_public_key="test_key_2",
            langfuse_secret="test_secret_2",
            langfuse_host="https://test2.langfuse.com",
        )
        assert litellm.initialized_langfuse_clients == 2

        # Third client should fail with exception
        with pytest.raises(Exception) as exc_info:
            logger3 = LangFuseLogger(
                langfuse_public_key="test_key_3",
                langfuse_secret="test_secret_3",
                langfuse_host="https://test3.langfuse.com",
            )

        # Verify the error message contains the expected text
        assert "Max langfuse clients reached" in str(exc_info.value)

        # Counter should still be 2 (third client failed to initialize)
        assert litellm.initialized_langfuse_clients == 2


def test_rerank_response_logging():
    """
    Test that rerank responses are properly formatted for Langfuse logging
    """
    from litellm.types.rerank import RerankResponse

    logger = LangFuseLogger(
        langfuse_public_key="test_key",
        langfuse_secret="test_secret",
        langfuse_host="https://test.langfuse.com",
    )

    # Test with results
    response_with_results = RerankResponse(
        id="test-123",
        results=[
            {"index": 0, "relevance_score": 0.95, "document": {"text": "First document"}},
            {"index": 2, "relevance_score": 0.87, "document": {"text": "Third document"}},
            {"index": 1, "relevance_score": 0.72, "document": {"text": "Second document"}},
        ],
    )

    input_val, output_val = logger._get_langfuse_input_output_content(
        kwargs={"call_type": "rerank"},
        response_obj=response_with_results,
        prompt={"query": "test query", "documents": ["doc1", "doc2", "doc3"]},
        level="INFO",
        status_message=None,
    )

    # Check that output is properly formatted
    assert output_val is not None
    assert isinstance(output_val, list)
    assert len(output_val) == 3
    assert output_val[0]["index"] == 0
    assert output_val[0]["relevance_score"] == 0.95
    assert output_val[0]["document"] == "First document"

    # Test with None results
    response_no_results = RerankResponse(id="test-456", results=None)

    input_val, output_val = logger._get_langfuse_input_output_content(
        kwargs={"call_type": "rerank"},
        response_obj=response_no_results,
        prompt={"query": "test query", "documents": ["doc1", "doc2", "doc3"]},
        level="INFO",
        status_message=None,
    )

    # Check that output is None when results are None
    assert output_val is None

    # Test with empty results
    response_empty_results = RerankResponse(id="test-789", results=[])

    input_val, output_val = logger._get_langfuse_input_output_content(
        kwargs={"call_type": "rerank"},
        response_obj=response_empty_results,
        prompt={"query": "test query", "documents": ["doc1", "doc2", "doc3"]},
        level="INFO",
        status_message=None,
    )

    # Check that output is None when results are empty
    assert output_val is None
