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


def test_sensitive_data_filter_with_special_characters():
    # Create a test logger
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)

    # Create a filter
    sensitive_filter = SensitiveDataFilter()

    # Test cases with special characters in keys
    test_cases = [
        {
            "input": '{"api_key": "sk-1234567890"}',
            "expected": '{"api_key": "REDACTED"}',
        },
        {
            "input": '{"api-key": "sk-1234567890"}',
            "expected": '{"api-key": "REDACTED"}',
        },
        {
            "input": '{"api/key": "sk-1234567890"}',
            "expected": '{"api/key": "REDACTED"}',
        },
        {
            "input": '{"api\\key": "sk-1234567890"}',
            "expected": '{"api\\key": "REDACTED"}',
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


def test_sensitive_data_filter_with_format_strings():
    # Create a test logger
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)

    # Create a filter
    sensitive_filter = SensitiveDataFilter()

    # Test cases with format strings
    test_cases = [
        {
            "input": "API key: %s",
            "args": ("sk-1234567890",),
            "expected": "API key: REDACTED",
        },
        {
            "input": "Credentials: %s, Token: %s",
            "args": ("secret123", "abc123"),
            "expected": "Credentials: REDACTED, Token: REDACTED",
        },
        {
            "input": "API base: %s, Key: %s",
            "args": ("https://api.example.com", "sk-1234567890"),
            "expected": "API base: REDACTED, Key: REDACTED",
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
            args=test_case["args"],
            exc_info=None,
        )

        # Apply the filter
        sensitive_filter.filter(record)

        # Verify the output
        assert (
            record.msg == test_case["expected"]
        ), f"Failed for input: {test_case['input']} with args: {test_case['args']}"


def test_sensitive_data_filter_reliability():
    # Create a test logger
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)

    # Create a SensitiveDataFilter and break its regex pattern to cause failure
    sensitive_filter = SensitiveDataFilter()
    sensitive_filter.SENSITIVE_KEYS = [
        ")"
    ]  # Invalid regex pattern that will cause failure

    # Add the filter
    logger.addFilter(sensitive_filter)

    # Try to log a message - this should not raise an exception
    try:
        logger.debug("Test message with sensitive data: api_key=sk-1234567890")
    except Exception as e:
        pytest.fail(f"Logging failed with exception: {str(e)}")

    # Clean up
    logger.removeFilter(sensitive_filter)
