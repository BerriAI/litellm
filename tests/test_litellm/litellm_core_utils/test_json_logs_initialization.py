"""
Unit tests for JSON logging initialization.
Tests that all loggers are properly configured when JSON_LOGS=True.
"""

import json
import logging
import os
import sys
from io import StringIO
from unittest import mock

import pytest


def test_json_logs_initialization():
    """Test that all loggers are properly configured when JSON_LOGS=True"""
    # Reset modules to ensure clean import
    for module in list(sys.modules.keys()):
        if module.startswith("litellm"):
            del sys.modules[module]
    
    # Set environment variable before import
    with mock.patch.dict(os.environ, {"JSON_LOGS": "True", "LITELLM_LOG": "DEBUG"}):
        # Import litellm with JSON_LOGS enabled
        import litellm
        from litellm._logging import (
            ALL_LOGGERS,
            json_logs,
            verbose_logger,
            verbose_proxy_logger,
            verbose_router_logger,
        )
        
        # Verify json_logs is enabled
        assert json_logs is True
        assert litellm.json_logs is True
        
        # Check that all loggers have handlers with JsonFormatter
        for logger in ALL_LOGGERS:
            assert len(logger.handlers) > 0, f"Logger {logger.name} has no handlers"
            assert logger.propagate is False, f"Logger {logger.name} should not propagate"
            
            # Check that at least one handler has JsonFormatter
            has_json_formatter = False
            for handler in logger.handlers:
                if handler.formatter and handler.formatter.__class__.__name__ == "JsonFormatter":
                    has_json_formatter = True
                    break
            assert has_json_formatter, f"Logger {logger.name} does not have JsonFormatter"


def test_json_logs_output_format():
    """Test that log output is valid JSON when JSON_LOGS=True"""
    # Reset modules
    for module in list(sys.modules.keys()):
        if module.startswith("litellm"):
            del sys.modules[module]
    
    with mock.patch.dict(os.environ, {"JSON_LOGS": "True", "LITELLM_LOG": "ERROR"}):
        # Capture stderr
        captured_output = StringIO()
        
        with mock.patch("sys.stderr", captured_output):
            # Import and use litellm
            import litellm
            from litellm._logging import verbose_logger
            
            # Log an error
            verbose_logger.error("Test error message")
            
            # Log with exception
            try:
                raise ValueError("Test exception")
            except Exception:
                verbose_logger.error("Error with exception", exc_info=True)
        
        # Get output
        output = captured_output.getvalue()
        lines = output.strip().split("\n")
        
        # Verify each line is valid JSON
        assert len(lines) >= 2, "Should have at least 2 log lines"
        
        for line in lines:
            if line:  # Skip empty lines
                # Parse JSON
                data = json.loads(line)
                
                # Check required fields
                assert "message" in data
                assert "level" in data
                assert "timestamp" in data
                assert data["level"] == "ERROR"
                
                # Check optional stacktrace field
                if "Error with exception" in data["message"]:
                    assert "stacktrace" in data
                    assert "ValueError: Test exception" in data["stacktrace"]


def test_root_logger_configured():
    """Test that the root logger is configured for JSON logging"""
    # Reset modules
    for module in list(sys.modules.keys()):
        if module.startswith("litellm"):
            del sys.modules[module]
    
    with mock.patch.dict(os.environ, {"JSON_LOGS": "True", "LITELLM_LOG": "ERROR"}):
        # Capture stderr
        captured_output = StringIO()
        
        with mock.patch("sys.stderr", captured_output):
            # Import litellm
            import litellm
            
            # Use root logger
            root_logger = logging.getLogger()
            root_logger.error("Root logger error")
            
            # Use a custom module logger (inherits from root)
            custom_logger = logging.getLogger("my.custom.module")
            custom_logger.error("Custom module error")
        
        # Get output
        output = captured_output.getvalue()
        lines = output.strip().split("\n")
        
        # Verify we got JSON output from both loggers
        messages = []
        for line in lines:
            if line:
                data = json.loads(line)
                messages.append(data["message"])
        
        assert "Root logger error" in messages
        assert "Custom module error" in messages


def test_json_logs_disabled():
    """Test that JSON logging is not used when JSON_LOGS is not set"""
    # Reset modules
    for module in list(sys.modules.keys()):
        if module.startswith("litellm"):
            del sys.modules[module]
    
    # Ensure JSON_LOGS is not set
    env = os.environ.copy()
    env.pop("JSON_LOGS", None)
    
    with mock.patch.dict(os.environ, env, clear=True):
        # Import litellm
        import litellm
        from litellm._logging import json_logs, verbose_logger
        
        # Verify json_logs is disabled
        assert json_logs is False
        
        # Capture output
        captured_output = StringIO()
        with mock.patch("sys.stderr", captured_output):
            verbose_logger.error("Test error without JSON")
        
        output = captured_output.getvalue().strip()
        
        # Output should not be JSON
        if output:  # May be empty if logging is disabled
            try:
                json.loads(output)
                pytest.fail("Output should not be valid JSON when JSON_LOGS is disabled")
            except json.JSONDecodeError:
                pass  # Expected