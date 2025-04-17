import logging

import pytest

from litellm._logging import SensitiveDataFilter


def test_sensitive_data_filter():
    # Create a test logger
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)

    # Create a filter
    sensitive_filter = SensitiveDataFilter()

    # Test cases
    test_cases = [
        {
            "input": '{"vertex_credentials": {"project_id": "test-project", "location": "us-central1", "private_key": "test-private-key"}}',
            "expected": '{"vertex_credentials": {"project_id": "test-project", "location": "us-central1", "private_key": "REDACTED"}}',
        },
        {
            "input": '{"api_key": "sk-1234567890"}',
            "expected": '{"api_key": "REDACTED"}',
        },
        {
            "input": '{"openai_api_key": "sk-1234567890"}',
            "expected": '{"openai_api_key": "REDACTED"}',
        },
        {"input": '{"password": "secret123"}', "expected": '{"password": "REDACTED"}'},
        {"input": '{"token": "abc123"}', "expected": '{"token": "REDACTED"}'},
        {
            "input": '{"api_base": "https://api.example.com"}',
            "expected": '{"api_base": "REDACTED"}',
        },
        {
            "input": '{"non_sensitive": "value", "credentials": "secret"}',
            "expected": '{"non_sensitive": "value", "credentials": "REDACTED"}',
        },
    ]

    for test_case in test_cases:
        # Create a log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=test_case["input"],
            args=(),
            exc_info=None,
        )

        # Apply the filter
        sensitive_filter.filter(record)

        # Verify the output
        assert (
            record.msg == test_case["expected"]
        ), f"Failed for input: {test_case['input']}"


def test_sensitive_data_filter_with_different_formats():
    # Create a test logger
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)

    # Create a filter
    sensitive_filter = SensitiveDataFilter()

    # Test different formats
    test_cases = [
        {"input": "api_key=sk-1234567890", "expected": "api_key=REDACTED"},
        {
            "input": "'credentials': 'secret123'",
            "expected": "'credentials': 'REDACTED'",
        },
        {"input": "\"token\": 'abc123'", "expected": "\"token\": 'REDACTED'"},
    ]

    for test_case in test_cases:
        # Create a log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=test_case["input"],
            args=(),
            exc_info=None,
        )

        # Apply the filter
        sensitive_filter.filter(record)

        # Verify the output
        assert (
            record.msg == test_case["expected"]
        ), f"Failed for input: {test_case['input']}"
