import unittest
import logging
import json
from datetime import datetime
from litellm.proxy.json_logging import JsonFormatter, AccessJsonFormatter

class TestJsonFormatter(unittest.TestCase):
    def setUp(self):
        self.formatter = JsonFormatter()

    def test_basic_log_format(self):
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        formatted = self.formatter.format(record)
        log_entry = json.loads(formatted)

        self.assertEqual(log_entry["level"], "INFO")
        self.assertEqual(log_entry["message"], "Test message")
        self.assertEqual(log_entry["logger_name"], "test_logger")
        self.assertIn("timestamp", log_entry)
        self.assertIn("process", log_entry)
        self.assertIn("thread", log_entry)

    def test_log_with_exception(self):
        try:
            raise ValueError("Test error")
        except ValueError as e:
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Test error occurred",
                args=(),
                exc_info=(type(e), e, e.__traceback__)
            )
            formatted = self.formatter.format(record)
            log_entry = json.loads(formatted)

            self.assertEqual(log_entry["level"], "ERROR")
            self.assertEqual(log_entry["message"], "Test error occurred")
            self.assertIn("exception", log_entry)
            self.assertEqual(log_entry["exception"]["type"], "ValueError")
            self.assertEqual(log_entry["exception"]["message"], "Test error")
            self.assertIsInstance(log_entry["exception"]["traceback"], list)

    def test_log_with_extra_data(self):
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.extra_data = {"user_id": "123", "action": "login"}
        formatted = self.formatter.format(record)
        log_entry = json.loads(formatted)

        self.assertEqual(log_entry["user_id"], "123")
        self.assertEqual(log_entry["action"], "login")

class TestAccessJsonFormatter(unittest.TestCase):
    def setUp(self):
        self.formatter = AccessJsonFormatter()

    def test_access_log_format(self):
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="%s - %s %s HTTP/%s %d",
            args=("127.0.0.1", "GET", "/test", "1.1", 200),
            exc_info=None
        )
        formatted = self.formatter.format(record)
        log_entry = json.loads(formatted)

        self.assertEqual(log_entry["level"], "INFO")
        self.assertEqual(log_entry["client_addr"], "127.0.0.1")
        self.assertEqual(log_entry["method"], "GET")
        self.assertEqual(log_entry["full_path"], "/test")
        self.assertEqual(log_entry["http_version"], "1.1")
        self.assertEqual(log_entry["status_code"], 200)
        self.assertIn("timestamp", log_entry)
        self.assertIn("process", log_entry)
        self.assertIn("thread", log_entry)
