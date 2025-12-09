"""
Test passthrough guardrails field-level targeting.

Tests that request_fields and response_fields correctly extract
and send only specified fields to the guardrail.
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy._types import PassThroughGuardrailSettings
from litellm.proxy.pass_through_endpoints.passthrough_guardrails import (
    PassthroughGuardrailHandler,
)


def test_no_fields_set_sends_full_body():
    """
    Test that when no request_fields are set, the entire request body
    is JSON dumped and sent to the guardrail.
    """
    request_data = {
        "model": "rerank-english-v3.0",
        "query": "What is coffee?",
        "documents": [
            {"text": "Paris is the capital of France."},
            {"text": "Coffee is a brewed drink."}
        ]
    }
    
    # No guardrail settings means full body
    result = PassthroughGuardrailHandler.prepare_input(
        request_data=request_data, 
        guardrail_settings=None
    )
    
    # Result should be JSON string of full request
    assert isinstance(result, str)
    result_dict = json.loads(result)
    
    # Should contain all fields
    assert "model" in result_dict
    assert "query" in result_dict
    assert "documents" in result_dict
    assert result_dict["query"] == "What is coffee?"
    assert len(result_dict["documents"]) == 2


def test_request_fields_query_only():
    """
    Test that when request_fields is set to ["query"], only the query field
    is extracted and sent to the guardrail.
    """
    request_data = {
        "model": "rerank-english-v3.0",
        "query": "What is coffee?",
        "documents": [
            {"text": "Paris is the capital of France."},
            {"text": "Coffee is a brewed drink."}
        ]
    }
    
    # Set request_fields to only extract query
    guardrail_settings = PassThroughGuardrailSettings(
        request_fields=["query"]
    )
    
    result = PassthroughGuardrailHandler.prepare_input(
        request_data=request_data, 
        guardrail_settings=guardrail_settings
    )
    
    # Result should only contain query
    assert isinstance(result, str)
    assert "What is coffee?" in result
    
    # Should NOT contain documents
    assert "Paris is the capital" not in result
    assert "Coffee is a brewed drink" not in result


def test_request_fields_documents_wildcard():
    """
    Test that when request_fields is set to ["documents[*]"], only the documents
    array is extracted and sent to the guardrail.
    """
    request_data = {
        "model": "rerank-english-v3.0",
        "query": "What is coffee?",
        "documents": [
            {"text": "Paris is the capital of France."},
            {"text": "Coffee is a brewed drink."}
        ]
    }
    
    # Set request_fields to extract documents array
    guardrail_settings = PassThroughGuardrailSettings(
        request_fields=["documents[*]"]
    )
    
    result = PassthroughGuardrailHandler.prepare_input(
        request_data=request_data, 
        guardrail_settings=guardrail_settings
    )
    
    # Result should contain documents
    assert isinstance(result, str)
    assert "Paris is the capital" in result
    assert "Coffee is a brewed drink" in result
    
    # Should NOT contain query
    assert "What is coffee?" not in result

