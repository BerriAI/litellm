"""Test that cost calculation uses appropriate log levels"""
import logging
import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm import completion_cost


def test_cost_calculation_uses_debug_level():
    """
    Test that cost calculation logs use DEBUG level instead of INFO.
    This ensures cost calculation details don't appear in production logs.
    Part of fix for issue #9815.

    Note: This test uses a custom log handler instead of caplog because
    caplog doesn't work reliably with pytest-xdist parallel execution.
    """
    from litellm._logging import verbose_logger

    # Create a custom handler to capture log records
    class LogRecordHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, record):
            self.records.append(record)

    # Set up custom handler
    handler = LogRecordHandler()
    handler.setLevel(logging.DEBUG)
    original_level = verbose_logger.level
    verbose_logger.setLevel(logging.DEBUG)
    verbose_logger.addHandler(handler)

    try:
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

        # Call completion_cost to trigger logs
        try:
            cost = completion_cost(
                completion_response=mock_response,
                model="gpt-3.5-turbo"
            )
        except Exception:
            pass  # Cost calculation may fail, but we're checking log levels

        # Find the cost calculation log records
        cost_calc_records = [
            record for record in handler.records
            if "selected model name for cost calculation" in record.message
        ]

        # Verify that cost calculation logs are at DEBUG level
        assert len(cost_calc_records) > 0, "No cost calculation logs found"

        for record in cost_calc_records:
            assert record.levelno == logging.DEBUG, \
                f"Cost calculation log should be DEBUG level, but was {record.levelname}"
    finally:
        # Clean up: remove handler and restore original logger level
        verbose_logger.removeHandler(handler)
        verbose_logger.setLevel(original_level)


def test_batch_cost_calculation_uses_debug_level():
    """
    Test that batch cost calculation logs also use DEBUG level.

    Note: This test uses a custom log handler instead of caplog because
    caplog doesn't work reliably with pytest-xdist parallel execution.
    """
    from litellm.cost_calculator import batch_cost_calculator
    from litellm.types.utils import Usage
    from litellm._logging import verbose_logger

    # Create a custom handler to capture log records
    class LogRecordHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, record):
            self.records.append(record)

    # Set up custom handler
    handler = LogRecordHandler()
    handler.setLevel(logging.DEBUG)
    original_level = verbose_logger.level
    verbose_logger.setLevel(logging.DEBUG)
    verbose_logger.addHandler(handler)

    try:
        # Create a mock usage object
        usage = Usage(prompt_tokens=100, completion_tokens=200, total_tokens=300)

        # Call batch_cost_calculator to trigger logs
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
            record for record in handler.records
            if "Calculating batch cost per token" in record.message
        ]

        # Verify logs exist and are at DEBUG level
        if batch_cost_records:  # May not always log depending on the code path
            for record in batch_cost_records:
                assert record.levelno == logging.DEBUG, \
                    f"Batch cost calculation log should be DEBUG level, but was {record.levelname}"
    finally:
        # Clean up: remove handler and restore original logger level
        verbose_logger.removeHandler(handler)
        verbose_logger.setLevel(original_level)
