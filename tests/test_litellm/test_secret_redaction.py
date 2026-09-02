import logging
import logging.config
import sys
from collections.abc import Callable
from io import StringIO
from unittest.mock import patch

import pytest

from litellm._logging import (
    JsonFormatter,
    _redact_string,
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


def test_redact_string_catches_minimum_length_virtual_key():
    """Regression test for LIT-4355: keys at the enforced 16-char minimum
    (MINIMUM_CUSTOM_KEY_LENGTH) must be treated as key-shaped by the scrubber."""
    minimum_length_key = "sk-abcdefghijklm"
    assert len(minimum_length_key) == 16
    result = redact_string("msg: " + minimum_length_key)
    assert minimum_length_key not in result
    assert "REDACTED" in result


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


def test_filter_preserves_uvicorn_color_message_args():
    """Regression test: uvicorn's startup banner logs a plain message plus a
    colorized `extra={"color_message": ...}` copy of the same "%s://%s:%d" template,
    both meant to be filled in from record.args. uvicorn's own ColourizedFormatter
    re-substitutes color_message against record.args when writing to a TTY, instead
    of using the already-formatted record.msg.

    Before this fix, the filter cleared record.args after substituting only
    record.msg, so color_message was rendered with args=None and the raw
    "%s://%s:%d" placeholders were printed instead of the real host/port.
    """
    from uvicorn.logging import DefaultFormatter

    addr_format = "%s://%s:%d"
    plain_message = f"Uvicorn running on {addr_format} (Press CTRL+C to quit)"
    color_message = f"Uvicorn running on {addr_format} (Press CTRL+C to quit)"

    logger = logging.getLogger("uvicorn.error")
    saved_handlers, saved_level = logger.handlers[:], logger.level
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    formatter = DefaultFormatter("%(levelprefix)s %(message)s")
    formatter.use_colors = True
    handler.setFormatter(formatter)
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    try:
        logger.info(
            plain_message,
            "http",
            "0.0.0.0",
            4000,
            extra={"color_message": color_message},
        )
        output = buf.getvalue()
    finally:
        logger.handlers = saved_handlers
        logger.setLevel(saved_level)

    assert "%s" not in output and "%d" not in output, f"unsubstituted placeholders leaked: {output!r}"
    assert "http://0.0.0.0:4000" in output


def test_filter_redacts_secrets_substituted_into_color_message():
    """The color_message substitution runs before the extra-field redaction
    loop, so a secret arriving through record.args lands in color_message and
    must still be scrubbed. Substituting after that loop would ship the secret
    to any colorized handler."""
    record = logging.LogRecord(
        name="uvicorn.error",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="connecting with %s",
        args=(SECRET,),
        exc_info=None,
    )
    record.color_message = "connecting with %s"

    _secret_filter.filter(record)

    assert SECRET not in record.color_message
    assert "REDACTED" in record.color_message


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


def test_xai_key_redaction_catches_proxy_log_and_config_dump():
    """xai_key is redacted in proxy log and config dump formats."""
    cases = [
        ("setting litellm.xai_key=xai-test-secret-123456", "xai-test-secret-123456"),
        ("'xai_key': 'xai-test-secret-123456'", "xai-test-secret-123456"),
    ]
    for secret_line, secret in cases:
        result = redact_string(secret_line)
        assert secret not in result
        assert "REDACTED" in result, f"xai_key redaction missed: {secret_line!r}"


def test_module_level_provider_key_redaction_catches_proxy_log_format():
    """Provider module-level keys are redacted when logged by proxy startup."""
    cases = [
        ("setting litellm.groq_key=gsk-test-secret-123456", "gsk-test-secret-123456"),
        (
            "setting litellm.openai_key=openai-test-secret-123456",
            "openai-test-secret-123456",
        ),
    ]
    for secret_line, secret in cases:
        result = redact_string(secret_line)
        assert secret not in result
        assert "REDACTED" in result, f"Module-level key redaction missed: {secret_line!r}"

    safe = "cache_key=cache-value-123456"
    assert redact_string(safe) == safe


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
        verbose_proxy_logger.debug(f"param_name=general_settings, param_value={general_settings}")

    output = _capture_logger_output(log_messages)
    assert "my-random-secret-key-1234" not in output
    assert "REDACTED" in output
    # Non-sensitive values should survive
    assert "enable_jwt_auth" in output


# ── GCP service-account / Vertex credential redaction ──


_SAMPLE_SA_JSON = (
    '{"type": "service_account", "project_id": "my-proj-123", '
    '"private_key_id": "abc123def", '
    '"private_key": "-----BEGIN PRIVATE KEY-----\\nMIIEvQIBADANBgkq\\n-----END PRIVATE KEY-----\\n", '
    '"client_email": "sa@my-proj.iam.gserviceaccount.com", '
    '"client_id": "123456789"}'
)


def test_pem_private_key_redacted_in_json():
    result = redact_string(_SAMPLE_SA_JSON)
    assert "MIIEvQIBADA" not in result
    assert "-----BEGIN" not in result


def test_pem_private_key_redacted_in_dict_repr():
    import json

    sa = json.loads(_SAMPLE_SA_JSON)
    result = redact_string(str(sa))
    assert "MIIEvQIBADA" not in result


def test_service_account_blob_fully_redacted():
    result = redact_string(f"Got={_SAMPLE_SA_JSON}")
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
    result = redact_string(traceback_text)
    assert "MIIEvQIBADA" not in result
    assert "-----BEGIN" not in result


def test_gcp_oauth_token_redacted():
    result = redact_string("access token ya29.c.c0ASRK0GZvXlongtokenhere")
    assert "ya29." not in result
    assert "REDACTED" in result


def test_non_pem_private_key_value_redacted():
    result = redact_string("'private_key': 'some-non-pem-secret-value'")
    assert "some-non-pem-secret" not in result


def test_normal_vertex_log_not_redacted():
    msg = "Vertex: Loading vertex credentials, is_file_path=True, current dir /app"
    assert redact_string(msg) == msg


THIRD_PARTY_LOGGERS = (
    "apscheduler.executors.default",
    "apscheduler.scheduler",
    "asyncio",
    "backoff",
    "httpx",
    "uvicorn.error",
)


def _capture_from_logger(logger_name: str, emit: Callable[[logging.Logger], None]) -> str:
    """Emit via `logger_name` and return only that logger's output as seen by a root handler.

    A handler on the root logger stands in for a log-shipping sink litellm does not own.
    The name predicate keeps the assertion scoped to the logger under test, so records
    from any other logger cannot decide the result.

    This idiom only reaches root when nothing between the logger and root stops
    propagation. `callHandlers` re-checks `propagate` at every level as it walks up, so
    an ANCESTOR with `propagate = False` ends the walk early and this returns an empty
    string no matter what was emitted; forcing it on the logger under test, as done
    below, is not enough. For a logger whose ancestors are configured that way, attach
    the capture handler to the logger itself and assert on that instead, and always
    assert the captured output is non-empty so a silent miss cannot pass.
    """
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.addFilter(lambda record: record.name == logger_name)
    lg = logging.getLogger(logger_name)
    saved = (lg.level, lg.propagate, logging.root.level)
    lg.setLevel(logging.DEBUG)
    lg.propagate = True
    logging.root.setLevel(logging.DEBUG)
    logging.root.addHandler(handler)
    try:
        emit(lg)
        return buf.getvalue()
    finally:
        logging.root.removeHandler(handler)
        lg.setLevel(saved[0])
        lg.propagate = saved[1]
        logging.root.setLevel(saved[2])


def test_third_party_logger_messages_are_redacted():
    for logger_name in THIRD_PARTY_LOGGERS:
        output = _capture_from_logger(logger_name, lambda lg: lg.error("value %s", SECRET))

        assert output.strip(), f"no record captured for {logger_name}"
        assert SECRET not in output, f"{logger_name} leaked a secret"
        assert "REDACTED" in output, f"{logger_name} was not redacted"


def test_third_party_logger_tracebacks_are_redacted():
    def emit(lg: logging.Logger) -> None:
        try:
            raise ValueError("value " + SECRET)
        except ValueError:
            lg.error("call failed", exc_info=True)

    for logger_name in THIRD_PARTY_LOGGERS:
        output = _capture_from_logger(logger_name, emit)

        assert output.strip(), f"no record captured for {logger_name}"
        assert SECRET not in output, f"{logger_name} leaked a secret in a traceback"
        assert "REDACTED" in output, f"{logger_name} traceback was not redacted"


UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


def test_redaction_survives_uvicorn_logging_reconfiguration():
    """Proxy startup hands uvicorn a logging config, and `dictConfig` replaces the handlers
    of every logger it names. Redaction is attached to the logger rather than to a handler
    so that it outlives that; moving it onto a handler would fail here.

    The config below is written out rather than imported from the one litellm ships on
    purpose. What is under test is `dictConfig` semantics, so any config that names the
    loggers exercises it; importing the real one would add coupling without adding
    coverage. The capture reads the reconfigured logger's own handler because the config
    sets `propagate = False`, which is where uvicorn's handler sits in a running proxy.
    """
    saved = tuple(
        (logging.getLogger(name), logging.getLogger(name).handlers[:], logging.getLogger(name).level)
        for name in UVICORN_LOGGERS
    )
    uvicorn_shaped_config: dict[str, object] = {
        "version": 1,
        "disable_existing_loggers": False,
        "handlers": {"default": {"class": "logging.StreamHandler"}},
        "loggers": {name: {"handlers": ["default"], "level": "INFO", "propagate": False} for name in UVICORN_LOGGERS},
    }
    try:
        logging.config.dictConfig(uvicorn_shaped_config)

        for logger_name in THIRD_PARTY_LOGGERS:
            filters = logging.getLogger(logger_name).filters
            assert _secret_filter in filters, f"{logger_name} lost redaction across reconfiguration"

        buf = StringIO()
        reconfigured = logging.getLogger("uvicorn.error")
        reconfigured.addHandler(logging.StreamHandler(buf))
        reconfigured.setLevel(logging.DEBUG)
        reconfigured.error("value %s", SECRET)
        output = buf.getvalue()

        assert output.strip(), "no record captured for uvicorn.error"
        assert SECRET not in output, "uvicorn.error leaked a secret after reconfiguration"
        assert "REDACTED" in output, "uvicorn.error was not redacted after reconfiguration"
    finally:
        for lg, handlers, level in saved:
            lg.handlers[:] = handlers
            lg.setLevel(level)
            lg.propagate = True


def test_aws_credential_redaction_catches_quoted_values():
    """AWS creds appear as quoted dict-repr values, not just bare key=value."""
    cases = (
        "{'aws_secret_access_key': 'wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY'}",
        '{"aws_session_token": "IQoJb3JpZ2luX2VjEaCXVzLWVhc3QtMSJHMEUCIQ"}',
        "aws_session_token: 'FwoGZXIvYXdzEBYaDHh4eHh4eHh4eHh4eCLLAe'",
        "aws_secret_access_key=wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY",
        "{'aws_access_key_id': 'not-an-akia-shaped-value'}",
    )
    for secret_line in cases:
        result = redact_string(secret_line)
        assert "REDACTED" in result, f"AWS redaction missed: {secret_line!r}"
        assert "wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY" not in result
        assert "IQoJb3JpZ2luX2VjEaCXVzLWVhc3QtMSJHMEUCIQ" not in result

    safe = "'aws_region_name': 'us-east-1'"
    assert redact_string(safe) == safe


@pytest.mark.parametrize(
    "extra",
    (
        {"api_base": {f"https://host/v1?key={SECRET}"}},
        {"blob": {"authorization": f"Bearer {SECRET}"}},
        {"blob": [f"Bearer {SECRET}"]},
        {"blob": ({"nested": {"deep": SECRET}},)},
    ),
    ids=("set", "dict", "list", "nested"),
)
def test_json_formatter_redacts_non_string_extra_values(extra):
    """SecretRedactionFilter only scrubs str attrs, so containers must be caught on render."""
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(_secret_filter)

    logger = logging.getLogger("test_json_extra_redaction")
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        logger.warning("request sent", extra=extra)
    finally:
        logger.handlers = []

    output = buf.getvalue()
    assert output.strip(), "no record captured"
    assert SECRET not in output, f"non-string extra leaked a secret: {output}"
    assert "REDACTED" in output
