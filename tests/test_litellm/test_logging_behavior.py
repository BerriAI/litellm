import os
import tempfile
import re
import json
from pathlib import Path
from datetime import datetime

import pytest

# Import the loggers from litellm._logging
from litellm._logging import verbose_logger, verbose_proxy_logger, verbose_router_logger


class TestLoggingBehavior:
    """Test suite to verify logging behavior for all LiteLLM loggers."""

    def read_log_file_contents(self, log_file_path):
        """Helper method to read and return contents of log file."""
        if not os.path.exists(log_file_path):
            return ""
        
        with open(log_file_path, 'r') as f:
            return f.read()

    @pytest.fixture(autouse=True)
    def setup_log_file(self, temp_log_file):
        """Use the temp_log_file fixture to ensure proper isolation."""
        self.temp_log_path = temp_log_file
        
        # Set environment variable before importing/reloading
        original_log_file = os.environ.get("LITELLM_LOG_FILE")
        os.environ["LITELLM_LOG_FILE"] = temp_log_file
        
        # Force reload of the logging module to pick up new environment variable
        import importlib
        import litellm._logging
        importlib.reload(litellm._logging)
        
        yield
        
        # Cleanup: Restore original environment variable
        if original_log_file is not None:
            os.environ["LITELLM_LOG_FILE"] = original_log_file
        else:
            os.environ.pop("LITELLM_LOG_FILE", None)
        
        # Reload again to restore original state
        importlib.reload(litellm._logging)

    def test_verbose_logger_info_level(self):
        """Test that verbose_logger writes to file with INFO level."""
        test_message = "INFO level test message from verbose_logger"
        
        # Log at INFO level
        verbose_logger.info(test_message)
        
        # Force flush all handlers to ensure they write to disk
        for handler in verbose_logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
        
        # Read log file contents
        log_file_path = os.environ.get("LITELLM_LOG_FILE")
        assert log_file_path is not None, "LITELLM_LOG_FILE environment variable should be set"
        
        log_contents = self.read_log_file_contents(log_file_path)
        assert test_message in log_contents, f"Message '{test_message}' should be found in log file"

    def test_verbose_logger_debug_level(self):
        """Test that verbose_logger writes to file with DEBUG level."""
        test_message = "DEBUG level test message from verbose_logger"
        
        # Log at DEBUG level
        verbose_logger.debug(test_message)
        
        # Read log file contents
        log_file_path = os.environ.get("LITELLM_LOG_FILE")
        assert log_file_path is not None, "LITELLM_LOG_FILE environment variable should be set"
        
        log_contents = self.read_log_file_contents(log_file_path)
        assert test_message in log_contents, f"Message '{test_message}' should be found in log file"

    def test_verbose_proxy_logger_info_level(self):
        """Test that verbose_proxy_logger writes to file with INFO level."""
        test_message = "INFO level test message from verbose_proxy_logger"
        
        # Log at INFO level
        verbose_proxy_logger.info(test_message)
        
        # Read log file contents
        log_file_path = os.environ.get("LITELLM_LOG_FILE")
        assert log_file_path is not None, "LITELLM_LOG_FILE environment variable should be set"
        
        log_contents = self.read_log_file_contents(log_file_path)
        assert test_message in log_contents, f"Message '{test_message}' should be found in log file"

    def test_verbose_proxy_logger_debug_level(self):
        """Test that verbose_proxy_logger writes to file with DEBUG level."""
        test_message = "DEBUG level test message from verbose_proxy_logger"
        
        # Log at DEBUG level
        verbose_proxy_logger.debug(test_message)
        
        # Read log file contents
        log_file_path = os.environ.get("LITELLM_LOG_FILE")
        assert log_file_path is not None, "LITELLM_LOG_FILE environment variable should be set"
        
        log_contents = self.read_log_file_contents(log_file_path)
        assert test_message in log_contents, f"Message '{test_message}' should be found in log file"

    def test_verbose_router_logger_info_level(self):
        """Test that verbose_router_logger writes to file with INFO level."""
        test_message = "INFO level test message from verbose_router_logger"
        
        # Log at INFO level
        verbose_router_logger.info(test_message)
        
        # Read log file contents
        log_file_path = os.environ.get("LITELLM_LOG_FILE")
        assert log_file_path is not None, "LITELLM_LOG_FILE environment variable should be set"
        
        log_contents = self.read_log_file_contents(log_file_path)
        assert test_message in log_contents, f"Message '{test_message}' should be found in log file"

    def test_verbose_router_logger_debug_level(self):
        """Test that verbose_router_logger writes to file with DEBUG level."""
        test_message = "DEBUG level test message from verbose_router_logger"
        
        # Log at DEBUG level
        verbose_router_logger.debug(test_message)
        
        # Read log file contents
        log_file_path = os.environ.get("LITELLM_LOG_FILE")
        assert log_file_path is not None, "LITELLM_LOG_FILE environment variable should be set"
        
        log_contents = self.read_log_file_contents(log_file_path)
        assert test_message in log_contents, f"Message '{test_message}' should be found in log file"

    def test_log_format_includes_timestamp_and_level(self):
        """Test that log entries include timestamp and level information."""
        test_message = "Format test message"
        
        # Log at INFO level
        verbose_logger.info(test_message)
        
        # Read log file contents
        log_file_path = os.environ.get("LITELLM_LOG_FILE")
        assert log_file_path is not None, "LITELLM_LOG_FILE environment variable should be set"
        
        log_contents = self.read_log_file_contents(log_file_path)
        
        # Check for timestamp format (should be in HH:MM:SS format based on _logging.py)
        assert re.search(r'\d{2}:\d{2}:\d{2}', log_contents), "Log should contain timestamp in HH:MM:SS format"
        
        # Check for level information
        assert 'INFO' in log_contents, "Log should contain INFO level indicator"
        
        # Check for logger name
        assert 'LiteLLM' in log_contents, "Log should contain LiteLLM logger name"

    def test_multiple_loggers_write_to_same_file(self):
        """Test that all loggers write to the same file."""
        messages = {
            'verbose_logger': "Message from verbose_logger",
            'verbose_proxy_logger': "Message from verbose_proxy_logger", 
            'verbose_router_logger': "Message from verbose_router_logger"
        }
        
        # Log messages from different loggers
        verbose_logger.info(messages['verbose_logger'])
        verbose_proxy_logger.info(messages['verbose_proxy_logger'])
        verbose_router_logger.info(messages['verbose_router_logger'])
        
        # Read log file contents
        log_file_path = os.environ.get("LITELLM_LOG_FILE")
        assert log_file_path is not None, "LITELLM_LOG_FILE environment variable should be set"
        
        log_contents = self.read_log_file_contents(log_file_path)
        
        # Verify all messages are in the same file
        for message in messages.values():
            assert message in log_contents, f"Message '{message}' should be found in log file"

    def test_log_file_is_not_empty(self):
        """Test that the log file is not empty after logging."""
        # Log a message
        verbose_logger.info("Test message to ensure file is not empty")
        
        # Read log file contents
        log_file_path = os.environ.get("LITELLM_LOG_FILE")
        assert log_file_path is not None, "LITELLM_LOG_FILE environment variable should be set"
        
        log_contents = self.read_log_file_contents(log_file_path)
        
        # Verify file is not empty
        assert len(log_contents.strip()) > 0, "Log file should not be empty after logging"


class TestJSONLoggingBehavior:
    """Test suite to verify JSON logging behavior for all LiteLLM loggers."""

    def read_log_file_contents(self, log_file_path):
        """Helper method to read and return contents of log file."""
        if not os.path.exists(log_file_path):
            return ""
        
        with open(log_file_path, 'r') as f:
            return f.read()

    @pytest.fixture(autouse=True)
    def setup_json_logging(self, temp_log_file):
        """Set up JSON logging environment and ensure proper isolation."""
        self.temp_log_path = temp_log_file
        
        # Store original environment variables
        original_log_file = os.environ.get("LITELLM_LOG_FILE")
        original_json_logs = os.environ.get("JSON_LOGS")
        
        # Set environment variables for JSON logging
        os.environ["LITELLM_LOG_FILE"] = temp_log_file
        os.environ["JSON_LOGS"] = "True"
        
        # Force reload of the logging module to pick up new environment variables
        import importlib
        import litellm._logging
        importlib.reload(litellm._logging)
        
        yield
        
        # Cleanup: Restore original environment variables
        if original_log_file is not None:
            os.environ["LITELLM_LOG_FILE"] = original_log_file
        else:
            os.environ.pop("LITELLM_LOG_FILE", None)
            
        if original_json_logs is not None:
            os.environ["JSON_LOGS"] = original_json_logs
        else:
            os.environ.pop("JSON_LOGS", None)
        
        # Reload again to restore original state
        importlib.reload(litellm._logging)

    def test_verbose_logger_json_info_level(self):
        """Test that verbose_logger writes JSON formatted logs at INFO level."""
        test_message = "JSON INFO level test message from verbose_logger"
        
        # Log at INFO level
        verbose_logger.info(test_message)
        
        # Force flush all handlers to ensure they write to disk
        for handler in verbose_logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
        
        # Read log file contents
        log_file_path = os.environ.get("LITELLM_LOG_FILE")
        assert log_file_path is not None, "LITELLM_LOG_FILE environment variable should be set"
        
        log_contents = self.read_log_file_contents(log_file_path)
        assert log_contents.strip(), "Log file should not be empty"
        
        # Parse JSON and verify structure
        log_lines = [line.strip() for line in log_contents.strip().split('\n') if line.strip()]
        assert len(log_lines) > 0, "Should have at least one log line"
        
        # Find the line containing our test message
        target_log = None
        for line in log_lines:
            try:
                parsed = json.loads(line)
                if parsed.get("message") == test_message:
                    target_log = parsed
                    break
            except json.JSONDecodeError:
                continue
        
        assert target_log is not None, f"Could not find JSON log entry with message: {test_message}"
        
        # Verify JSON structure
        assert "message" in target_log, "JSON log should contain 'message' field"
        assert "level" in target_log, "JSON log should contain 'level' field"
        assert "timestamp" in target_log, "JSON log should contain 'timestamp' field"
        
        # Verify content
        assert target_log["message"] == test_message
        assert target_log["level"] == "INFO"
        
        # Verify timestamp is in ISO 8601 format
        timestamp_str = target_log["timestamp"]
        try:
            datetime.fromisoformat(timestamp_str)
        except ValueError:
            pytest.fail(f"Timestamp '{timestamp_str}' is not in valid ISO 8601 format")

    def test_verbose_logger_json_debug_level(self):
        """Test that verbose_logger writes JSON formatted logs at DEBUG level."""
        test_message = "JSON DEBUG level test message from verbose_logger"
        
        # Log at DEBUG level
        verbose_logger.debug(test_message)
        
        # Read log file contents
        log_file_path = os.environ.get("LITELLM_LOG_FILE")
        assert log_file_path is not None, "LITELLM_LOG_FILE environment variable should be set"
        
        log_contents = self.read_log_file_contents(log_file_path)
        assert log_contents.strip(), "Log file should not be empty"
        
        # Parse JSON and verify structure
        log_lines = [line.strip() for line in log_contents.strip().split('\n') if line.strip()]
        
        # Find the line containing our test message
        target_log = None
        for line in log_lines:
            try:
                parsed = json.loads(line)
                if parsed.get("message") == test_message:
                    target_log = parsed
                    break
            except json.JSONDecodeError:
                continue
        
        assert target_log is not None, f"Could not find JSON log entry with message: {test_message}"
        assert target_log["level"] == "DEBUG"

    def test_verbose_proxy_logger_json_info_level(self):
        """Test that verbose_proxy_logger writes JSON formatted logs at INFO level."""
        test_message = "JSON INFO level test message from verbose_proxy_logger"
        
        # Log at INFO level
        verbose_proxy_logger.info(test_message)
        
        # Read log file contents
        log_file_path = os.environ.get("LITELLM_LOG_FILE")
        assert log_file_path is not None, "LITELLM_LOG_FILE environment variable should be set"
        
        log_contents = self.read_log_file_contents(log_file_path)
        assert log_contents.strip(), "Log file should not be empty"
        
        # Parse JSON and verify structure
        log_lines = [line.strip() for line in log_contents.strip().split('\n') if line.strip()]
        
        # Find the line containing our test message
        target_log = None
        for line in log_lines:
            try:
                parsed = json.loads(line)
                if parsed.get("message") == test_message:
                    target_log = parsed
                    break
            except json.JSONDecodeError:
                continue
        
        assert target_log is not None, f"Could not find JSON log entry with message: {test_message}"
        
        # Verify JSON structure and content
        assert target_log["message"] == test_message
        assert target_log["level"] == "INFO"
        
        # Verify timestamp is in ISO 8601 format
        timestamp_str = target_log["timestamp"]
        try:
            datetime.fromisoformat(timestamp_str)
        except ValueError:
            pytest.fail(f"Timestamp '{timestamp_str}' is not in valid ISO 8601 format")

    def test_verbose_proxy_logger_json_debug_level(self):
        """Test that verbose_proxy_logger writes JSON formatted logs at DEBUG level."""
        test_message = "JSON DEBUG level test message from verbose_proxy_logger"
        
        # Log at DEBUG level
        verbose_proxy_logger.debug(test_message)
        
        # Read log file contents
        log_file_path = os.environ.get("LITELLM_LOG_FILE")
        assert log_file_path is not None, "LITELLM_LOG_FILE environment variable should be set"
        
        log_contents = self.read_log_file_contents(log_file_path)
        assert log_contents.strip(), "Log file should not be empty"
        
        # Parse JSON and verify structure
        log_lines = [line.strip() for line in log_contents.strip().split('\n') if line.strip()]
        
        # Find the line containing our test message
        target_log = None
        for line in log_lines:
            try:
                parsed = json.loads(line)
                if parsed.get("message") == test_message:
                    target_log = parsed
                    break
            except json.JSONDecodeError:
                continue
        
        assert target_log is not None, f"Could not find JSON log entry with message: {test_message}"
        assert target_log["level"] == "DEBUG"

    def test_verbose_router_logger_json_info_level(self):
        """Test that verbose_router_logger writes JSON formatted logs at INFO level."""
        test_message = "JSON INFO level test message from verbose_router_logger"
        
        # Log at INFO level
        verbose_router_logger.info(test_message)
        
        # Read log file contents
        log_file_path = os.environ.get("LITELLM_LOG_FILE")
        assert log_file_path is not None, "LITELLM_LOG_FILE environment variable should be set"
        
        log_contents = self.read_log_file_contents(log_file_path)
        assert log_contents.strip(), "Log file should not be empty"
        
        # Parse JSON and verify structure
        log_lines = [line.strip() for line in log_contents.strip().split('\n') if line.strip()]
        
        # Find the line containing our test message
        target_log = None
        for line in log_lines:
            try:
                parsed = json.loads(line)
                if parsed.get("message") == test_message:
                    target_log = parsed
                    break
            except json.JSONDecodeError:
                continue
        
        assert target_log is not None, f"Could not find JSON log entry with message: {test_message}"
        
        # Verify JSON structure and content
        assert target_log["message"] == test_message
        assert target_log["level"] == "INFO"
        
        # Verify timestamp is in ISO 8601 format
        timestamp_str = target_log["timestamp"]
        try:
            datetime.fromisoformat(timestamp_str)
        except ValueError:
            pytest.fail(f"Timestamp '{timestamp_str}' is not in valid ISO 8601 format")

    def test_verbose_router_logger_json_debug_level(self):
        """Test that verbose_router_logger writes JSON formatted logs at DEBUG level."""
        test_message = "JSON DEBUG level test message from verbose_router_logger"
        
        # Log at DEBUG level
        verbose_router_logger.debug(test_message)
        
        # Read log file contents
        log_file_path = os.environ.get("LITELLM_LOG_FILE")
        assert log_file_path is not None, "LITELLM_LOG_FILE environment variable should be set"
        
        log_contents = self.read_log_file_contents(log_file_path)
        assert log_contents.strip(), "Log file should not be empty"
        
        # Parse JSON and verify structure
        log_lines = [line.strip() for line in log_contents.strip().split('\n') if line.strip()]
        
        # Find the line containing our test message
        target_log = None
        for line in log_lines:
            try:
                parsed = json.loads(line)
                if parsed.get("message") == test_message:
                    target_log = parsed
                    break
            except json.JSONDecodeError:
                continue
        
        assert target_log is not None, f"Could not find JSON log entry with message: {test_message}"
        assert target_log["level"] == "DEBUG"

    def test_json_output_is_valid_json(self):
        """Test that all JSON log output can be parsed as valid JSON."""
        test_messages = [
            "JSON test message 1",
            "JSON test message 2",
            "JSON test message 3"
        ]
        
        # Log messages from all loggers
        verbose_logger.info(test_messages[0])
        verbose_proxy_logger.info(test_messages[1])
        verbose_router_logger.info(test_messages[2])
        
        # Read log file contents
        log_file_path = os.environ.get("LITELLM_LOG_FILE")
        assert log_file_path is not None, "LITELLM_LOG_FILE environment variable should be set"
        
        log_contents = self.read_log_file_contents(log_file_path)
        assert log_contents.strip(), "Log file should not be empty"
        
        # Parse each line as JSON
        log_lines = [line.strip() for line in log_contents.strip().split('\n') if line.strip()]
        parsed_logs = []
        
        for line in log_lines:
            try:
                parsed = json.loads(line)
                parsed_logs.append(parsed)
            except json.JSONDecodeError as e:
                pytest.fail(f"Failed to parse JSON log line: {line}. Error: {e}")
        
        assert len(parsed_logs) >= len(test_messages), f"Should have at least {len(test_messages)} parsed log entries"
        
        # Verify each parsed log has required fields
        for parsed_log in parsed_logs:
            assert isinstance(parsed_log, dict), "Parsed log should be a dictionary"
            assert "message" in parsed_log, "Each log should have a 'message' field"
            assert "level" in parsed_log, "Each log should have a 'level' field"
            assert "timestamp" in parsed_log, "Each log should have a 'timestamp' field"

    def test_json_timestamp_iso8601_format(self):
        """Test that JSON log timestamps are in ISO 8601 format."""
        test_message = "Timestamp format test message"
        
        # Log a message
        verbose_logger.info(test_message)
        
        # Read log file contents
        log_file_path = os.environ.get("LITELLM_LOG_FILE")
        assert log_file_path is not None, "LITELLM_LOG_FILE environment variable should be set"
        
        log_contents = self.read_log_file_contents(log_file_path)
        assert log_contents.strip(), "Log file should not be empty"
        
        # Parse JSON and verify timestamp format
        log_lines = [line.strip() for line in log_contents.strip().split('\n') if line.strip()]
        
        # Find the line containing our test message
        target_log = None
        for line in log_lines:
            try:
                parsed = json.loads(line)
                if parsed.get("message") == test_message:
                    target_log = parsed
                    break
            except json.JSONDecodeError:
                continue
        
        assert target_log is not None, f"Could not find JSON log entry with message: {test_message}"
        
        timestamp_str = target_log["timestamp"]
        
        # Verify timestamp can be parsed as ISO 8601
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp_str)
            assert isinstance(parsed_timestamp, datetime), "Parsed timestamp should be a datetime object"
        except ValueError as e:
            pytest.fail(f"Timestamp '{timestamp_str}' is not in valid ISO 8601 format. Error: {e}")
        
        # Verify timestamp format matches expected pattern (YYYY-MM-DDTHH:MM:SS.ffffff)
        import re
        iso8601_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?$'
        assert re.match(iso8601_pattern, timestamp_str), f"Timestamp '{timestamp_str}' does not match ISO 8601 pattern"

    def test_json_logs_contain_expected_fields(self):
        """Test that JSON logs contain all expected fields with correct types."""
        test_message = "Field validation test message"
        
        # Log a message
        verbose_logger.info(test_message)
        
        # Read log file contents
        log_file_path = os.environ.get("LITELLM_LOG_FILE")
        assert log_file_path is not None, "LITELLM_LOG_FILE environment variable should be set"
        
        log_contents = self.read_log_file_contents(log_file_path)
        assert log_contents.strip(), "Log file should not be empty"
        
        # Parse JSON and verify fields
        log_lines = [line.strip() for line in log_contents.strip().split('\n') if line.strip()]
        
        # Find the line containing our test message
        target_log = None
        for line in log_lines:
            try:
                parsed = json.loads(line)
                if parsed.get("message") == test_message:
                    target_log = parsed
                    break
            except json.JSONDecodeError:
                continue
        
        assert target_log is not None, f"Could not find JSON log entry with message: {test_message}"
        
        # Verify required fields exist and have correct types
        assert "message" in target_log, "JSON log should contain 'message' field"
        assert "level" in target_log, "JSON log should contain 'level' field"
        assert "timestamp" in target_log, "JSON log should contain 'timestamp' field"
        
        assert isinstance(target_log["message"], str), "'message' field should be a string"
        assert isinstance(target_log["level"], str), "'level' field should be a string"
        assert isinstance(target_log["timestamp"], str), "'timestamp' field should be a string"
        
        # Verify field values
        assert target_log["message"] == test_message
        assert target_log["level"] in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], "Level should be a valid log level"

    def test_multiple_json_loggers_write_to_same_file(self):
        """Test that all loggers write JSON formatted logs to the same file."""
        messages = {
            'verbose_logger': "JSON message from verbose_logger",
            'verbose_proxy_logger': "JSON message from verbose_proxy_logger",
            'verbose_router_logger': "JSON message from verbose_router_logger"
        }
        
        # Log messages from different loggers
        verbose_logger.info(messages['verbose_logger'])
        verbose_proxy_logger.info(messages['verbose_proxy_logger'])
        verbose_router_logger.info(messages['verbose_router_logger'])
        
        # Read log file contents
        log_file_path = os.environ.get("LITELLM_LOG_FILE")
        assert log_file_path is not None, "LITELLM_LOG_FILE environment variable should be set"
        
        log_contents = self.read_log_file_contents(log_file_path)
        assert log_contents.strip(), "Log file should not be empty"
        
        # Parse all JSON logs
        log_lines = [line.strip() for line in log_contents.strip().split('\n') if line.strip()]
        parsed_logs = []
        
        for line in log_lines:
            try:
                parsed = json.loads(line)
                parsed_logs.append(parsed)
            except json.JSONDecodeError:
                continue
        
        # Find logs for each message
        found_messages = set()
        for parsed_log in parsed_logs:
            message = parsed_log.get("message", "")
            if message in messages.values():
                found_messages.add(message)
        
        # Verify all messages are found in JSON format
        for message in messages.values():
            assert message in found_messages, f"Message '{message}' should be found in JSON logs"