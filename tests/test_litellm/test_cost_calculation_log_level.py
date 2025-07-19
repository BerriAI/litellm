"""Test that cost calculation uses appropriate log levels"""
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm import completion_cost


def test_cost_calculation_uses_debug_level(caplog):
    """
    Test that cost calculation logs use DEBUG level instead of INFO.
    This ensures cost calculation details don't appear in production logs.
    Part of fix for issue #9815.
    """
    # Create a mock completion response
    mock_response = {
        "id": "test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Test response"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }
    
    # Test that cost calculation logs are at DEBUG level
    with caplog.at_level(logging.DEBUG):
        try:
            cost = completion_cost(
                completion_response=mock_response,
                model="gpt-3.5-turbo"
            )
        except Exception:
            pass  # Cost calculation may fail, but we're checking log levels
    
    # Find the cost calculation log records
    cost_calc_records = [
        record for record in caplog.records 
        if "selected model name for cost calculation" in record.message
    ]
    
    # Verify that cost calculation logs are at DEBUG level
    assert len(cost_calc_records) > 0, "No cost calculation logs found"
    
    for record in cost_calc_records:
        assert record.levelno == logging.DEBUG, \
            f"Cost calculation log should be DEBUG level, but was {record.levelname}"


def test_batch_cost_calculation_uses_debug_level(caplog):
    """
    Test that batch cost calculation logs also use DEBUG level.
    """
    from litellm.cost_calculator import batch_cost_calculator
    from litellm.types.utils import Usage
    
    # Create a mock usage object
    usage = Usage(prompt_tokens=100, completion_tokens=200, total_tokens=300)
    
    # Test that batch cost calculation logs are at DEBUG level
    with caplog.at_level(logging.DEBUG):
        try:
            batch_cost_calculator(
                usage=usage,
                model="gpt-3.5-turbo",
                custom_llm_provider="openai"
            )
        except Exception:
            pass  # May fail, but we're checking log levels
    
    # Find batch cost calculation log records
    batch_cost_records = [
        record for record in caplog.records 
        if "Calculating batch cost per token" in record.message
    ]
    
    # Verify logs exist and are at DEBUG level
    if batch_cost_records:  # May not always log depending on the code path
        for record in batch_cost_records:
            assert record.levelno == logging.DEBUG, \
                f"Batch cost calculation log should be DEBUG level, but was {record.levelname}"