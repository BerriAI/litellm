import logging
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from litellm._logging import (
    JsonFormatter,
    _secret_filter,
    verbose_logger,
    verbose_proxy_logger,
    verbose_router_logger,
)
from litellm.litellm_core_utils.secret_redaction import redact_string

SECRET = "sk-proj-abc123def456ghi789jklmnopqrst"


@pytest.fixture(autouse=True)
def _enable_redaction():
    """Ensure secret redaction is on (the default) for all tests in this module."""
    with patch("litellm._logging._ENABLE_SECRET_REDACTION", True):
        yield


def _capture_logger_output(fn):
    """Run fn with all litellm loggers wired to a StringIO buffer, return output."""
    buf = StringIO()
    h = logging.StreamHandler(buf)
    h.addFilter(_secret_filter)
    loggers = [verbose_logger, verbose_proxy_logger, verbose_router_logger]
    saved = [(lg, lg.handlers[:], lg.level) for lg in loggers]
    for lg in loggers:
        lg.handlers.clear()
        lg.addHandler(h)
        lg.setLevel(logging.DEBUG)
    try:
        fn()
        return buf.getvalue()
    finally:
        for lg, handlers, level in saved:
            lg.handlers.clear()
            for old_h in handlers:
                lg.addHandler(old_h)
            lg.setLevel(level)


def test_redact_string_catches_secret_patterns():
    """Core regex patterns redact known secret formats."""
    cases = [
        "Bearer eyJhbGciOiJSUzI1NiJ9.payload.sig",
        "api_key=a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
        "password=supersecretpassword123",
        "postgresql://admin:s3cretpass@db.example.com:5432/mydb",
        SECRET,
    ]
    for secret in cases:
        result = redact_string("msg: " + secret)
        assert secret not in result, f"{secret!r} was not redacted"
        assert "REDACTED" in result

    normal = "Loaded model gpt-4 with 3 replicas on us-east-1"
    assert redact_string(normal) == normal


def test_filter_redacts_secrets_in_logger_output():
    def log_messages():
        verbose_logger.debug("Key: " + SECRET)
        verbose_logger.debug("Normal message with no secrets")

    output = _capture_logger_output(log_messages)
    assert SECRET not in output
    assert "REDACTED" in output
    assert "Normal message with no secrets" in output


def test_filter_redacts_percent_style_args():
    """Secrets passed as %-style args should be redacted."""

    def log_messages():
        verbose_logger.debug("key=%s region=%s", SECRET, "us-east-1")

    output = _capture_logger_output(log_messages)
    assert SECRET not in output
    assert "us-east-1" in output


def test_filter_redacts_non_string_args():
    """Secrets inside dicts/lists passed as %-style args should be redacted."""

    def log_messages():
        verbose_logger.debug("Config: %s", {"nested": {"key": SECRET}})
        verbose_logger.debug("Keys: %s", [SECRET])

    output = _capture_logger_output(log_messages)
    assert SECRET not in output
    assert "REDACTED" in output


def test_filter_redacts_exception_tracebacks():
    """Secrets embedded in exception messages must be redacted in tracebacks."""

    def log_messages():
        try:
            raise ValueError(f"Auth failed with key {SECRET}")
        except ValueError:
            verbose_logger.exception("Something went wrong")

    output = _capture_logger_output(log_messages)
    assert SECRET not in output
    assert "REDACTED" in output
    assert "Something went wrong" in output


def test_filter_redacts_extra_fields():
    """Secrets passed via extra={...} must be redacted on the record."""
    record = logging.LogRecord(
        name="test",
        level=logging.DEBUG,
        pathname="",
        lineno=0,
        msg="request completed",
        args=(),
        exc_info=None,
    )
    record.api_key = SECRET
    record.region = "us-east-1"

    _secret_filter.filter(record)

    assert SECRET not in record.api_key
    assert "REDACTED" in record.api_key
    assert record.region == "us-east-1"


def test_disable_redaction_passes_secrets_through():
    """When LITELLM_DISABLE_REDACT_SECRETS=true, secrets pass through."""
    with patch("litellm._logging._ENABLE_SECRET_REDACTION", False):
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg="key=" + SECRET,
            args=(),
            exc_info=None,
        )
        _secret_filter.filter(record)
        assert "sk-proj-" in record.msg


def test_x_api_key_regex_does_not_consume_json_delimiters():
    """x-api-key pattern must stop before closing quotes/braces so JSON stays valid."""
    # Simulates a JSON log line containing an x-api-key header value
    json_line = '{"headers": {"x-api-key": "secret123"}, "status": 200}'
    result = redact_string(json_line)
    # The secret value should be redacted
    assert "secret123" not in result
    assert "REDACTED" in result
    # Closing delimiter must survive so the line is still valid-ish JSON
    assert '"status": 200' in result
    assert "}" in result


def test_json_excepthook_redacts_secrets():
    """Unhandled exceptions in JSON mode must have secrets redacted."""
    buf = StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(JsonFormatter())
    h.addFilter(_secret_filter)

    # Capture what the excepthook would emit
    record = logging.LogRecord(
        name="LiteLLM",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg=f"Connection failed with key {SECRET}",
        args=(),
        exc_info=None,
    )
    # Simulate the filter + formatter pipeline
    _secret_filter.filter(record)
    output = h.formatter.format(record)
    assert SECRET not in output
    assert "REDACTED" in output


def test_json_excepthook_redacts_traceback_secrets():
    """Unhandled exception tracebacks in JSON mode must have secrets redacted."""
    buf = StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(JsonFormatter())
    h.addFilter(_secret_filter)

    try:
        raise RuntimeError(f"Failed to auth with {SECRET}")
    except RuntimeError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="LiteLLM",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg=str(exc_info[1]),
        args=(),
        exc_info=exc_info,
    )
    _secret_filter.filter(record)
    output = h.formatter.format(record)
    assert SECRET not in output
    assert "REDACTED" in output


def test_key_name_redaction_catches_secrets_in_dict_repr():
    """Secrets inside dict repr strings are redacted based on key names."""
    cases = [
        # Python dict repr (the exact leak format from the bug report)
        "param_name=general_settings, param_value={'master_key': 'my-random-secret-key-1234', 'enable_jwt_auth': True}",
        # database_url
        "'database_url': 'postgres://admin:password@db.example.com:5432/litellm'",
        # JSON format
        '"database_url": "postgres://admin:password@db.example.com:5432/litellm"',
        # access_token
        "'access_token': 'some-opaque-token-value'",
        # refresh_token
        "refresh_token=my-refresh-tok-12345",
        # auth_token
        "'auth_token': 'random-auth-value'",
        # slack_webhook_url
        "'slack_webhook_url': 'https://hooks.slack.com/services/T00/B00/xxx'",
    ]
    for secret_line in cases:
        result = redact_string(secret_line)
        assert "REDACTED" in result, f"Key-name redaction missed: {secret_line!r}"

    # Non-sensitive keys should NOT be redacted
    safe = "'enable_jwt_auth': True, 'store_model_in_db': True"
    assert redact_string(safe) == safe


def test_key_name_redaction_in_general_settings_dict():
    """End-to-end: secrets inside a general_settings dict dump are redacted
    when logged through the named litellm loggers."""

    def log_messages():
        general_settings = {
            "master_key": "my-random-secret-key-1234",
            "database_url": "postgres://admin:password@db.example.com:5432/litellm",
            "enable_jwt_auth": True,
            "store_model_in_db": True,
        }
        verbose_proxy_logger.debug(
            f"param_name=general_settings, param_value={general_settings}"
        )

    output = _capture_logger_output(log_messages)
    assert "my-random-secret-key-1234" not in output
    assert "REDACTED" in output
    # Non-sensitive values should survive
    assert "enable_jwt_auth" in output


# ── Tests for issue #24902: Gemini API key leak via URL query param ──────────


def test_redact_url_query_param_key_question_mark():
    """?key=<secret> in a URL is fully redacted (Gemini-style API auth)."""
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=AIzaSyFakeKeyABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = redact_string(url)
    assert "AIzaSyFakeKeyABCDEFGHIJKLMNOPQRSTUVWXYZ" not in result
    assert "REDACTED" in result
    # The base URL prefix should be preserved
    assert "generativelanguage.googleapis.com" in result


def test_redact_url_query_param_key_ampersand():
    """&key=<secret> inside a multi-param URL is redacted while other params survive."""
    url = (
        "https://example.com/api?model=gemini-pro&key=SomeSecretKeyABCDEF12345&alt=sse"
    )
    result = redact_string(url)
    assert "SomeSecretKeyABCDEF12345" not in result
    assert "REDACTED" in result
    # Other query params must be preserved
    assert "model=gemini-pro" in result
    assert "alt=sse" in result


def test_redact_url_query_param_key_short_value_not_redacted():
    """Short values (< 8 chars) after key= are not redacted to avoid false positives."""
    url = "https://example.com/api?key=abc"
    result = redact_string(url)
    assert result == url


def test_redact_string_applied_to_httpx_error_message():
    """Simulates the Gemini leak path: httpx raise_for_status() includes URL with ?key=..."""
    import httpx

    # Build a fake URL with an embedded API key (the Gemini auth pattern)
    fake_key = "AIzaSyFakeGeminiKey1234567890ABCDE"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={fake_key}"
    request = httpx.Request("POST", url)
    response = httpx.Response(400, request=request)

    # httpx.Response.raise_for_status() includes the full URL in the error message
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        response.raise_for_status()

    error_str = redact_string(str(exc_info.value))
    assert fake_key not in error_str
    assert "REDACTED" in error_str


def test_redact_url_query_param_key_in_logger_output():
    """End-to-end: Gemini-style URL with ?key=... is redacted in logger output."""
    fake_key = "AIzaSyFakeGeminiKeyXYZ1234567890ABCDE"
    url = (
        f"https://generativelanguage.googleapis.com/v1/models/gemini-pro?key={fake_key}"
    )

    def log_messages():
        verbose_logger.debug("Request failed: %s", url)

    output = _capture_logger_output(log_messages)
    assert fake_key not in output
    assert "REDACTED" in output


def test_exception_mapping_respects_redaction_opt_out():
    """When LITELLM_DISABLE_REDACT_SECRETS=true, exception messages pass through unredacted."""
    import httpx

    from litellm.litellm_core_utils.exception_mapping_utils import exception_type

    fake_key = "AIzaSyFakeGeminiKey1234567890ABCDE"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro?key={fake_key}"
    request = httpx.Request("POST", url)
    response = httpx.Response(400, request=request)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        response.raise_for_status()
    original_exc = exc_info.value

    with patch(
        "litellm.litellm_core_utils.exception_mapping_utils._ENABLE_SECRET_REDACTION",
        False,
    ):
        with pytest.raises(Exception) as mapped_exc_info:
            exception_type(
                model="gemini/gemini-pro",
                original_exception=original_exc,
                custom_llm_provider="gemini",
            )
    # Key must survive in the mapped exception when redaction is disabled
    assert fake_key in str(mapped_exc_info.value)
# ── GCP service-account / Vertex credential redaction ──


_SAMPLE_SA_JSON = (
    '{"type": "service_account", "project_id": "my-proj-123", '
    '"private_key_id": "abc123def", '
    '"private_key": "-----BEGIN PRIVATE KEY-----\\nMIIEvQIBADANBgkq\\n-----END PRIVATE KEY-----\\n", '
    '"client_email": "sa@my-proj.iam.gserviceaccount.com", '
    '"client_id": "123456789"}'
)


def test_pem_private_key_redacted_in_json():
    result = _redact_string(_SAMPLE_SA_JSON)
    assert "MIIEvQIBADA" not in result
    assert "-----BEGIN" not in result


def test_pem_private_key_redacted_in_dict_repr():
    import json

    sa = json.loads(_SAMPLE_SA_JSON)
    result = _redact_string(str(sa))
    assert "MIIEvQIBADA" not in result


def test_service_account_blob_fully_redacted():
    result = _redact_string(f"Got={_SAMPLE_SA_JSON}")
    assert "my-proj-123" not in result
    assert "sa@my-proj.iam.gserviceaccount.com" not in result
    assert "abc123def" not in result
    assert "MIIEvQIBADA" not in result


def test_vertex_error_message_no_credential_leak():
    """The old Vertex error format leaked the full credential JSON.
    The new format must not contain any credential material."""
    new_msg = (
        "Unable to load vertex credentials from environment. "
        "Ensure the JSON is valid (check for unescaped newlines in private_key). "
        "Parse error: JSONDecodeError"
    )
    result = _redact_string(new_msg)
    assert result == new_msg  # nothing to redact


def test_vertex_traceback_redacts_pem():
    traceback_text = (
        "Traceback (most recent call last):\n"
        '  File "vertex_llm_base.py", line 95\n'
        "    json_obj = json.loads(credentials)\n"
        "json.decoder.JSONDecodeError: Invalid control character\n"
        "Failed to load vertex credentials. Error: "
        "Unable to load vertex credentials from environment. "
        f"Got={_SAMPLE_SA_JSON}"
    )
    result = _redact_string(traceback_text)
    assert "MIIEvQIBADA" not in result
    assert "-----BEGIN" not in result


def test_gcp_oauth_token_redacted():
    result = _redact_string("access token ya29.c.c0ASRK0GZvXlongtokenhere")
    assert "ya29." not in result
    assert "REDACTED" in result


def test_non_pem_private_key_value_redacted():
    result = _redact_string("'private_key': 'some-non-pem-secret-value'")
    assert "some-non-pem-secret" not in result


def test_normal_vertex_log_not_redacted():
    msg = "Vertex: Loading vertex credentials, is_file_path=True, current dir /app"
    assert _redact_string(msg) == msg
