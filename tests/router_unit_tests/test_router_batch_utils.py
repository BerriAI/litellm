import sys
import os
import traceback
from dotenv import load_dotenv
from fastapi import Request
from datetime import datetime

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from litellm import Router
import pytest
import litellm
from unittest.mock import patch, MagicMock, AsyncMock

import json
from io import BytesIO
from typing import Dict, List
from litellm.router_utils.batch_utils import (
    replace_model_in_jsonl,
    _get_router_metadata_variable_name,
)


# Fixtures
@pytest.fixture
def sample_jsonl_data() -> List[Dict]:
    """Fixture providing sample JSONL data"""
    return [
        {
            "body": {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        },
        {"body": {"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}},
    ]


@pytest.fixture
def sample_jsonl_bytes(sample_jsonl_data) -> bytes:
    """Fixture providing sample JSONL as bytes"""
    jsonl_str = "\n".join(json.dumps(line) for line in sample_jsonl_data)
    return jsonl_str.encode("utf-8")


@pytest.fixture
def sample_file_like(sample_jsonl_bytes):
    """Fixture providing a file-like object"""
    return BytesIO(sample_jsonl_bytes)


# Test cases
def test_bytes_input(sample_jsonl_bytes):
    """Test with bytes input"""
    new_model = "claude-3"
    result = replace_model_in_jsonl(sample_jsonl_bytes, new_model)

    assert result is not None


def test_tuple_input(sample_jsonl_bytes):
    """Test with tuple input"""
    new_model = "claude-3"
    test_tuple = ("test.jsonl", sample_jsonl_bytes, "application/json")
    result = replace_model_in_jsonl(test_tuple, new_model)

    assert result is not None


def test_file_like_object(sample_file_like):
    """Test with file-like object input"""
    new_model = "claude-3"
    result = replace_model_in_jsonl(sample_file_like, new_model)

    assert result is not None


def test_router_metadata_variable_name():
    """Test that the variable name is correct"""
    assert _get_router_metadata_variable_name(function_name="completion") == "metadata"
    assert (
        _get_router_metadata_variable_name(function_name="batch") == "litellm_metadata"
    )
