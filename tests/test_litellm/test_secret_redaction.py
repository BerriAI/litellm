import json
import logging
import logging.config
import sys
from collections.abc import Callable
from io import StringIO
from unittest.mock import patch

import pytest

from litellm._logging import (
    JsonFormatter,
    _get_uvicorn_json_log_config,
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


@pytest.fixture
def uvicorn_logger_state():
    """Snapshot and restore the process-wide uvicorn loggers around a `dictConfig` call.

    `dictConfig` swaps their handlers, flips `propagate`, and appends filters. The repo
    conftest only snapshots ALL_LOGGERS (root plus the three litellm loggers), so without
    this every later test in the process inherits stdout handlers and `propagate = False`.
    """
    saved = tuple(
        (lg, lg.handlers[:], lg.filters[:], lg.level, lg.propagate) for lg in map(logging.getLogger, UVICORN_LOGGERS)
    )
    yield
    for lg, handlers, filters, level, propagate in saved:
        lg.handlers[:] = handlers
        lg.filters[:] = filters
        lg.setLevel(level)
        lg.propagate = propagate


@pytest.fixture
def bare_uvicorn_loggers(uvicorn_logger_state):
    """Strip the uvicorn loggers so only the config under test can supply redaction.

    Clearing the filters is what makes the `uvicorn.error` case mean anything:
    `_redact_third_party_loggers()` already attached `_secret_filter` to that logger at
    import time, so a test that skipped this step would still pass with the config's own
    filter entry deleted. Once cleared, the config is the only possible source of
    redaction, which is also the situation in a fresh uvicorn worker.
    """
    for lg in map(logging.getLogger, UVICORN_LOGGERS):
        lg.handlers.clear()
        lg.filters.clear()
    yield


def _apply_uvicorn_json_log_config(*, dictconfig_accepts_filter_instances: bool = True) -> None:
    """Configure logging exactly as the proxy does when it starts uvicorn with JSON logs.

    Call this from the test body, never from a fixture. The config's handlers write to
    `ext://sys.stdout`, which `dictConfig` resolves to whatever `sys.stdout` is bound to at
    that instant, and pytest swaps that object between the setup and call phases: resolving
    it during setup captures the stream `capsys` is about to replace, and every assertion
    then reads an empty buffer.

    The flag is passed explicitly rather than left to the production default so these tests
    describe one runtime's behavior instead of the interpreter that happens to run them.
    """
    logging.config.dictConfig(
        _get_uvicorn_json_log_config(dictconfig_accepts_filter_instances=dictconfig_accepts_filter_instances)
    )


def _emit_uvicorn_access_line(path: str, status: int = 200) -> None:
    """Log the record uvicorn's access logger emits, argument-for-argument.

    uvicorn builds it as a template plus a 5-tuple (see `uvicorn.protocols.http`), never as
    a pre-rendered string, so a test that passed one string would not exercise the same path.
    """
    logging.getLogger("uvicorn.access").info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:52814", "GET", path, "1.1", status)


def test_redaction_survives_uvicorn_logging_reconfiguration(uvicorn_logger_state):
    """Proxy startup hands uvicorn a logging config, and `dictConfig` replaces the handlers
    of every logger it names. Redaction is attached to the logger rather than to a handler
    so that it outlives that; moving it onto a handler would fail here.

    The config below is written out rather than imported from the one litellm ships on
    purpose. What is under test is `dictConfig` semantics, so any config that names the
    loggers exercises it; importing the real one would add coupling without adding
    coverage. The capture reads the reconfigured logger's own handler because the config
    sets `propagate = False`, which is where uvicorn's handler sits in a running proxy.
    """
    uvicorn_shaped_config: dict[str, object] = {
        "version": 1,
        "disable_existing_loggers": False,
        "handlers": {"default": {"class": "logging.StreamHandler"}},
        "loggers": {name: {"handlers": ["default"], "level": "INFO", "propagate": False} for name in UVICORN_LOGGERS},
    }
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


@pytest.mark.parametrize("logger_name", UVICORN_LOGGERS)
def test_uvicorn_log_config_redacts_on_every_uvicorn_logger(logger_name, bare_uvicorn_loggers, capsys):
    """Every logger the proxy's uvicorn log config names must redact.

    `uvicorn.access` is the one that carries request paths, but `uvicorn` and `uvicorn.error`
    log connection and startup detail that can carry a credential too, and only `uvicorn.error`
    is covered by `_redact_third_party_loggers()`. The level is set explicitly because the
    config takes its level from `LITELLM_LOG`, and level is not what is under test.
    """
    _apply_uvicorn_json_log_config()
    lg = logging.getLogger(logger_name)
    lg.setLevel(logging.INFO)

    lg.error("token=%s", SECRET)

    out = capsys.readouterr().out
    assert out.strip(), f"no record captured for {logger_name}"
    assert SECRET not in out, f"{logger_name} leaked a secret through the uvicorn log config"
    assert "REDACTED" in out, f"{logger_name} was not redacted by the uvicorn log config"
    assert json.loads(out.strip())["component"] == logger_name


@pytest.mark.parametrize(
    "path, secret",
    [
        (f"/key/info?key={SECRET}", SECRET),
        (
            "/gemini/v1beta/models/gemini-2.5-flash:generateContent?key=AIzaSyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r",
            "AIzaSyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r",
        ),
    ],
    ids=["virtual_key_on_key_info", "google_api_key_on_gemini_passthrough"],
)
def test_uvicorn_access_log_redacts_credential_in_query_string(path, secret, bare_uvicorn_loggers, capsys):
    """Routes that take a credential as a query parameter must not persist it to the access log.

    `/key/info` declares `key` as a `fastapi.Query` parameter and the gemini passthrough takes
    `?key=<google api key>`, so both land in uvicorn's request line verbatim. Asserting the
    surrounding line survives keeps the log useful and fails a filter that just blanks the record.
    """
    _apply_uvicorn_json_log_config()

    _emit_uvicorn_access_line(path)

    line = capsys.readouterr().out.strip()
    assert line, "no record captured for uvicorn.access"
    message = json.loads(line)["message"]
    assert secret not in message, f"uvicorn.access leaked a credential: {message}"
    assert "REDACTED" in message
    assert path.split("?")[0] in message
    assert "GET" in message
    assert "200" in message


def test_uvicorn_access_log_leaves_clean_request_lines_intact(bare_uvicorn_loggers, capsys):
    """A request with nothing to redact is logged unchanged, so over-redaction fails here."""
    _apply_uvicorn_json_log_config()

    _emit_uvicorn_access_line("/health/liveliness")

    message = json.loads(capsys.readouterr().out.strip())["message"]
    assert message == '127.0.0.1:52814 - "GET /health/liveliness HTTP/1.1" 200'


def test_uvicorn_log_config_declares_secret_filter_on_every_logger():
    """No logger may be added to the config without redaction.

    The behavioral tests above only cover the loggers that exist today; this one fails when a
    fourth is added unfiltered.
    """
    loggers = _get_uvicorn_json_log_config(dictconfig_accepts_filter_instances=True)["loggers"]
    unfiltered = tuple(name for name, cfg in loggers.items() if _secret_filter not in cfg.get("filters", ()))

    assert unfiltered == (), f"uvicorn log config leaves these loggers unredacted: {unfiltered}"


def test_uvicorn_log_config_omits_filter_instances_below_python_311():
    """`dictConfig` resolves each `filters` entry as an id into a top-level "filters" section
    until Python 3.11, and raises `ValueError` on an instance. uvicorn applies this config from
    `Config.__init__`, so shipping instances at the floor of the supported range would stop the
    proxy from starting rather than leave it logging unredacted. Empty is what makes that safe:
    `common_logger_config` only resolves `filters` when the value is truthy.
    """
    loggers = _get_uvicorn_json_log_config(dictconfig_accepts_filter_instances=False)["loggers"]
    populated = tuple(name for name, cfg in loggers.items() if cfg["filters"])

    assert populated == (), f"these loggers ship a filter dictConfig cannot resolve below 3.11: {populated}"


def _dictconfig_resolves_filter_instances() -> bool:
    """Whether this interpreter's `dictConfig` accepts a filter instance in a `filters` list.

    Answered by trying it rather than by restating the version literal the gate uses, so a gate
    that drifts from what the runtime actually supports fails the test below instead of agreeing
    with itself.
    """
    probe = logging.getLogger("test.dictconfig_filter_instance_probe")
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "loggers": {probe.name: {"filters": [_secret_filter]}},
    }
    try:
        logging.config.dictConfig(config)
    except ValueError:
        return False
    finally:
        probe.filters.clear()
    return True


def test_uvicorn_log_config_ships_filters_exactly_when_dictconfig_resolves_them():
    """The version gate itself, which every other test here bypasses by passing the flag.

    The assertion is against what this interpreter's `dictConfig` actually does, so a gate that
    disagrees with the runtime executing it fails here. A boundary that is wrong only on a
    version this run is not using cannot be caught without that interpreter, which is what the
    `dictconfig_accepts_filter_instances=False` tests stand in for.
    """
    expected = (_secret_filter,) if _dictconfig_resolves_filter_instances() else ()
    loggers = _get_uvicorn_json_log_config()["loggers"]
    mismatched = tuple(name for name, cfg in loggers.items() if cfg["filters"] != expected)

    assert mismatched == (), f"version gate disagrees with this interpreter's dictConfig for: {mismatched}"


def test_uvicorn_log_config_without_filter_instances_still_logs(bare_uvicorn_loggers, capsys):
    """Dropping the filters must cost redaction and nothing else.

    The sub-3.11 config still has to be one uvicorn can apply, with the same handlers and the
    same JSON formatter, so a request line comes out whole on those runtimes too.
    """
    _apply_uvicorn_json_log_config(dictconfig_accepts_filter_instances=False)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)

    _emit_uvicorn_access_line("/health/liveliness")

    message = json.loads(capsys.readouterr().out.strip())["message"]
    assert message == '127.0.0.1:52814 - "GET /health/liveliness HTTP/1.1" 200'
