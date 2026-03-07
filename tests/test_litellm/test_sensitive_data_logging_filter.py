"""
Tests for _SensitiveDataFilter - verifies that sensitive data is redacted
from DEBUG log messages before they are emitted.
"""

import logging

import pytest

from litellm._logging import _SensitiveDataFilter


@pytest.fixture
def filter_instance():
    return _SensitiveDataFilter()


@pytest.fixture
def logger_with_filter(filter_instance):
    """Returns a logger with the filter attached and captures emitted records."""
    lg = logging.getLogger("test.sensitive_filter")
    lg.setLevel(logging.DEBUG)
    lg.filters.clear()
    lg.addFilter(filter_instance)

    records = []

    class CapturingHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = CapturingHandler()
    lg.handlers = [handler]
    lg.propagate = False
    return lg, records


# ---------------------------------------------------------------------------
# _mask_message unit tests
# ---------------------------------------------------------------------------


def test_masks_api_key_in_dict(filter_instance):
    """api_key value inside a dict repr is redacted."""
    msg = "config: {'api_key': 'sk-1234567890abcdef', 'model': 'gpt-4'}"
    result = filter_instance._mask_message(msg)
    assert "sk-1234567890abcdef" not in result
    assert "api_key" in result
    assert "model" in result


def test_masks_authorization_header(filter_instance):
    """Authorization header value inside a dict repr is redacted."""
    msg = "Pass through endpoint sending request to\nURL https://api.example.com\nheaders: {'authorization': 'Bearer sk-realkey123456', 'content-type': 'application/json'}\nbody: {}"
    result = filter_instance._mask_message(msg)
    assert "sk-realkey123456" not in result
    assert "content-type" in result


def test_masks_master_key_in_general_settings(filter_instance):
    """master_key inside general_settings dict is redacted (proxy_server.py:3149 case)."""
    msg = "_alerting_callbacks: {'polling_via_cache': False, 'store_model_in_db': True, 'master_key': 'sk-realmasterkey1234', 'alerting': ['email']}"
    result = filter_instance._mask_message(msg)
    assert "sk-realmaster" not in result  # middle should be masked
    assert "polling_via_cache" in result
    assert "alerting" in result


def test_masks_langsmith_api_key_in_team_settings(filter_instance):
    """langsmith_api_key inside default_team_settings list is redacted (proxy_server.py:2740 case)."""
    msg = " setting litellm.default_team_settings=[{'team_id': 'abc123', 'langsmith_api_key': 'lsv2_sk_029e2e6f21df4d98bf596eb9ca982b53', 'langsmith_project': 'test'}]"
    result = filter_instance._mask_message(msg)
    assert "lsv2_sk_029e2e6f21df4d98bf596eb9ca982b53" not in result
    assert "team_id" in result
    assert "langsmith_project" in result


def test_non_sensitive_fields_unchanged(filter_instance):
    """Non-sensitive dict fields pass through unmodified."""
    msg = "deployment: {'model_name': 'gpt-4', 'litellm_params': {'model': 'gpt-4'}}"
    result = filter_instance._mask_message(msg)
    assert result == msg


def test_message_without_dict_unchanged(filter_instance):
    """Messages with no dict repr are not modified."""
    msg = "Starting proxy server on port 4000"
    assert filter_instance._mask_message(msg) == msg


def test_multiple_dicts_in_message(filter_instance):
    """Both dicts in a message with multiple dicts are processed."""
    msg = "req: {'api_key': 'sk-secret99'} resp: {'status': 'ok', 'token': 'tok-abc123456'}"
    result = filter_instance._mask_message(msg)
    assert "sk-secret99" not in result
    assert "tok-abc123456" not in result
    assert "status" in result


def test_non_parseable_dict_skipped_safely(filter_instance):
    """If the dict contains non-literal values, the message passes through without crashing."""
    msg = "obj: {'fn': <function foo at 0x7f0>, 'api_key': 'sk-safe'}"
    # ast.literal_eval will fail, filter should not crash
    result = filter_instance._mask_message(msg)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Integration tests via the logger
# ---------------------------------------------------------------------------


def test_filter_redacts_on_debug_record(logger_with_filter):
    """DEBUG records have sensitive dict values redacted."""
    lg, records = logger_with_filter
    lg.debug(
        "headers: {'authorization': 'Bearer sk-supersecret', 'x-request-id': '123'}"
    )
    assert len(records) == 1
    assert "sk-supersecret" not in records[0].getMessage()
    assert "x-request-id" in records[0].getMessage()


def test_filter_does_not_affect_info_records(logger_with_filter):
    """INFO records are passed through without modification."""
    lg, records = logger_with_filter
    lg.info("config: {'api_key': 'sk-shouldnotbetouched'}")
    assert len(records) == 1
    assert "sk-shouldnotbetouched" in records[0].getMessage()


def test_filter_works_with_percent_style_args(logger_with_filter):
    """Filter handles logger.debug('msg %s', dict) style calls correctly."""
    lg, records = logger_with_filter
    lg.debug("settings: %s", {"master_key": "sk-percentstyle99", "env": "prod"})
    assert len(records) == 1
    assert "sk-percentstyle99" not in records[0].getMessage()
    assert "env" in records[0].getMessage()


def test_filter_never_drops_records(logger_with_filter):
    """Filter always returns True — no log records are suppressed."""
    lg, records = logger_with_filter
    lg.debug("normal message, no dict")
    lg.debug("{'api_key': 'sk-dropped?'}")
    assert len(records) == 2
