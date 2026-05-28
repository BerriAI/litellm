"""
DEBUG-mode secret-leak detection matrix (LIT-3200).

This is the QA monitoring layer for ``litellm._logging``: a parametric matrix
that exercises the cross-product of

    (secret-payload x litellm-logger x emission-style)

under ``_turn_on_debug()`` and asserts that no secret reaches the captured
log output. The negative control proves the matrix actually catches leaks --
flipping ``_ENABLE_SECRET_REDACTION`` off makes the same payload appear
verbatim in the output.

Background: secrets have leaked through DEBUG-mode logs in the past
(service-account JSON, signing material, master_key in config dumps).
The existing ``test_secret_redaction.py`` covers the redaction function and
a few logger paths in isolation; this file is the QA-process matrix that
exercises every (payload, logger, emission-style) combination in one place,
so a regression that disables the filter -- globally or for a single emission
style -- fails loudly here rather than landing silently.

The fake-secret strings below are stitched together from multiple pieces on
purpose so GitHub secret-scanning push protection does not false-flag this
test fixture. None of the values are real and none are valid against any
provider.

Run:
    pytest tests/test_litellm/test_secret_leak_debug_mode_matrix.py -v
"""
from __future__ import annotations

import logging
from io import StringIO
from typing import Callable, List, Tuple
from unittest.mock import patch

import pytest

from litellm._logging import (
    _secret_filter,
    _turn_on_debug,
    verbose_logger,
    verbose_proxy_logger,
    verbose_router_logger,
)

# Synthetic fixtures only. Strings are stitched at module-load time so
# secret-scanners do not see a literal credential prefix in this file.
# None of these values are real and none are valid against any provider.
_PAYLOADS: List[Tuple[str, str, str]] = [
    ("openai_sk_key",
     "sk-" + "proj-LEAK01A0B1C2D3E4F5G6H7I8J9K0L1M2",
     "LEAK01A0B1C2D3E4F5G6H7I8J9K0L1M2"),
    ("anthropic_sk_key",
     "sk-ant-api03" + "-LEAK02A0B1C2D3E4F5G6H7I8J9K0L1M2",
     "LEAK02A0B1C2D3E4F5G6H7I8J9K0L1M2"),
    ("bearer_token",
     "Bearer eyJ" + "hbGciOiJSUzI1NiJ9.LEAK03payload12345.sig67890",
     "LEAK03payload12345"),
    ("raw_jwt",
     "eyJ" + "hbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9." + "eyJ" + "zdWIiOiJMRUFLMDQifQ.signature-xyz",
     "eyJ" + "zdWIiOiJMRUFLMDQifQ"),
    ("aws_access_key_id",
     "AKIA" + "LEAK05IOSFODNN7Q",
     "AKIA" + "LEAK05IOSFODNN7Q"),
    ("aws_secret_access_key_kv",
     "aws_secret_access_key=" + "wJalrLEAK06FEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
     "wJalrLEAK06FEMI"),
    ("google_api_key",
     "AIza" + "SyA0LEAK07_VtPRcQ5cwSCNBcOoYG7VK8DA",
     "LEAK07"),
    ("postgres_url_credentials",
     "postgresql://" + "litellm_user:Pa-LEAK08-2025@db.internal:5432/litellm",
     "Pa-LEAK08-2025"),
    ("database_url_in_dict",
     "{\"database_url\": \"" + "postgres://admin:LEAK09pw@db.example:5432/x\"}",
     "LEAK09pw"),
    ("master_key_in_dict",
     "{\"master_key\": \"sk-" + "LEAK10-master-do-not-leak\"}",
     "sk-" + "LEAK10-master-do-not-leak"),
    ("password_kv",
     "password=" + "LEAK11-supersecret-pw-2025",
     "LEAK11-supersecret-pw-2025"),
    ("x_api_key_in_dict",
     "{\"x-api-key\": \"" + "LEAK12-long-mcp-token-value\"}",
     "LEAK12-long-mcp-token-value"),
    ("api_key_kv",
     "api_key=" + "LEAK13-A1B2C3D4E5F6A7B8C9D0E1F2A3B4",
     "LEAK13-A1B2C3D4E5F6A7B8C9D0E1F2A3B4"),
    ("databricks_pat",
     "dapi" + "deadbeef567890abcdef1234567890ab",
     "deadbeef567890abcdef"),
    ("gcp_oauth_token",
     "access token ya29." + "c.LEAK15c0ASRK0GZvXlongtokenhere",
     "ya29." + "c.LEAK15c0ASRK0GZvXlongtokenhere"),
    ("slack_webhook_url_in_dict",
     "{\"slack_webhook_url\": \"https://hooks.slack.com/services" + "/T0/B0/LEAK16abcdef\"}",
     "LEAK16abcdef"),
    ("refresh_token_in_dict",
     "{\"refresh_token\": \"rt" + "-LEAK17-very-secret-refresh-value\"}",
     "rt-LEAK17-very-secret-refresh-value"),
    ("access_token_in_dict",
     "{\"access_token\": \"opa" + "que-LEAK18-access-token-value\"}",
     "opaque-LEAK18-access-token-value"),
    ("basic_auth_header",
     "Authorization: Basic " + "dXNlcm5hbWU6LEAK19c2VjcmV0LXBhc3N3b3Jk",
     "dXNlcm5hbWU6LEAK19c2VjcmV0"),
    ("pem_private_key",
     "private_key: -----BEGIN P" + "RIVATE KEY-----\nMII" + "EvQIBADANBgkqhkiG9w0BAQEFAASCBKLEAK20cwggSjAgEAAoIBAQ\n-----END P" + "RIVATE KEY-----",
     "MII" + "EvQIBADANBgkqhkiG9w0BAQEFAASCBKLEAK20cwggSjAgEAAoIBAQ"),
    ("client_secret_kv",
     "client_secret=" + "LEAK21-oauth-client-secret-value",
     "LEAK21-oauth-client-secret-value"),
    ("azure_sas_sig",
     "https://acct.blob.core.windows.net/c/b?sv=2020&sig=" + "LEAK22%2BabcdefGHIJ%2F",
     "LEAK22%2BabcdefGHIJ"),
]

_LOGGERS = [
    ("verbose_logger", verbose_logger),
    ("verbose_proxy_logger", verbose_proxy_logger),
    ("verbose_router_logger", verbose_router_logger),
]


def _emit_msg_concat(logger, payload):
    logger.debug("Inbound credential: " + payload)


def _emit_fstring(logger, payload):
    logger.debug(f"Resolved auth header value={payload}")


def _emit_percent_args_str(logger, payload):
    logger.debug("Outbound key=%s region=%s", payload, "us-east-1")


def _emit_percent_args_dict(logger, payload):
    logger.debug("Config dump: %s", {"creds": payload})


def _emit_percent_args_list(logger, payload):
    logger.debug("Keys loaded: %s", [payload, "other"])


def _emit_extra_kwarg(logger, payload):
    logger.debug("Saved config", extra={"resolved_key": payload})


def _emit_exception_traceback(logger, payload):
    try:
        raise RuntimeError("Upstream auth failed with " + payload)
    except RuntimeError:
        logger.exception("Auth handler raised")


def _emit_log_at_debug_level(logger, payload):
    logger.log(logging.DEBUG, "Provider key: " + payload)


_EMISSION_STYLES: List[Tuple[str, Callable[[logging.Logger, str], None]]] = [
    ("msg_concat", _emit_msg_concat),
    ("fstring", _emit_fstring),
    ("percent_args_str", _emit_percent_args_str),
    ("percent_args_dict", _emit_percent_args_dict),
    ("percent_args_list", _emit_percent_args_list),
    ("extra_kwarg", _emit_extra_kwarg),
    ("exception_traceback", _emit_exception_traceback),
    ("log_at_debug_level", _emit_log_at_debug_level),
]


class _MatrixFormatter(logging.Formatter):
    """Custom formatter for the matrix. Appends any non-standard record
    attribute to the output so ``extra={...}`` fields the filter is meant
    to redact become visible in the captured buffer. The default
    ``logging.Formatter`` silently drops record extras and would make every
    ``extra_kwarg`` matrix cell vacuously pass.
    """

    # logging.LogRecord stock attributes; any record attribute not in this
    # set is treated as an extra field and appended to the output.
    _STD = {
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "exc_info", "exc_text", "stack_info",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "processName", "process", "message",
        "asctime", "taskName",
    }

    def format(self, record):
        base = super().format(record)
        extras = []
        for k, v in record.__dict__.items():
            if k in self._STD or k.startswith("_"):
                continue
            extras.append(f"{k}={v!r}")
        if extras:
            return base + " extras{" + " ".join(extras) + "}"
        return base


@pytest.fixture(autouse=True)
def _force_redaction_on():
    """Ensure the redaction switch is on for every test (the default),
    regardless of any env override the runner might inherit."""
    with patch("litellm._logging._ENABLE_SECRET_REDACTION", True):
        yield


@pytest.fixture
def debug_capture_buf():
    """Wire all three named litellm loggers into one in-memory buffer with
    ``_secret_filter`` attached and a custom formatter that renders ``extra``
    fields (so they are visible to assertions), then call ``_turn_on_debug()``
    so all three loggers are at DEBUG. Restore prior state on teardown.
    """
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(_MatrixFormatter("%(name)s:%(levelname)s:%(message)s"))
    handler.addFilter(_secret_filter)
    loggers = [verbose_logger, verbose_proxy_logger, verbose_router_logger]
    saved = [(lg, lg.handlers[:], lg.level, lg.propagate, lg.disabled)
             for lg in loggers]
    for lg in loggers:
        lg.handlers.clear()
        lg.addHandler(handler)
        lg.propagate = False
        lg.disabled = False
    _turn_on_debug()
    try:
        yield buf
    finally:
        for lg, handlers, level, propagate, disabled in saved:
            lg.handlers.clear()
            for old in handlers:
                lg.addHandler(old)
            lg.setLevel(level)
            lg.propagate = propagate
            lg.disabled = disabled


@pytest.mark.parametrize("payload_id,payload,marker", _PAYLOADS,
                         ids=[p[0] for p in _PAYLOADS])
@pytest.mark.parametrize("logger_id,logger", _LOGGERS,
                         ids=[lg[0] for lg in _LOGGERS])
@pytest.mark.parametrize("style_id,emit_fn", _EMISSION_STYLES,
                         ids=[s[0] for s in _EMISSION_STYLES])
def test_no_secret_leaks_in_debug_mode(
    debug_capture_buf, payload_id, payload, marker,
    logger_id, logger, style_id, emit_fn,
):
    """Matrix cell: every (secret, logger, emission-style) combination must
    not surface the secret in DEBUG-mode log output."""
    emit_fn(logger, payload)
    output = debug_capture_buf.getvalue()
    assert marker not in output, (
        f"DEBUG-mode leak detected: payload={payload_id} logger={logger_id} "
        f"style={style_id}\nmarker={marker!r} present in captured output:\n"
        f"{output!r}"
    )
    assert output.strip(), (
        f"Logger emitted nothing for payload={payload_id} logger={logger_id} "
        f"style={style_id} -- matrix cell did not exercise the filter."
    )


# Negative control -- stitched the same way for the same reason.
_NEG = "sk-" + "proj-NEGCTRLneverChangeThisStringEver1234"
_NEG_MARKER = "NEGCTRLneverChangeThisStringEver1234"


def test_matrix_catches_regression_when_redaction_disabled():
    """If ``_ENABLE_SECRET_REDACTION`` flips to False, the same matrix
    wiring must produce a real leak -- proves the detection criterion is
    not a tautology."""
    payload, marker = _NEG, _NEG_MARKER
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.addFilter(_secret_filter)
    # _turn_on_debug() modifies all three named loggers, so capture and
    # restore the state of all three (not just verbose_logger) -- otherwise
    # the remainder of the test session sees verbose_proxy_logger and
    # verbose_router_logger left at DEBUG.
    loggers = [verbose_logger, verbose_proxy_logger, verbose_router_logger]
    saved = [(lg, lg.handlers[:], lg.level, lg.propagate, lg.disabled)
             for lg in loggers]
    verbose_logger.handlers.clear()
    verbose_logger.addHandler(handler)
    verbose_logger.propagate = False
    verbose_logger.disabled = False
    _turn_on_debug()
    try:
        with patch("litellm._logging._ENABLE_SECRET_REDACTION", False):
            verbose_logger.debug("Provider key: " + payload)
        output = buf.getvalue()
        assert marker in output, (
            "Negative control failed: with redaction disabled, the payload "
            "should have leaked through, but the matrix did not see it. "
            "Captured output:\n" + repr(output)
        )
    finally:
        for lg, handlers, level, propagate, disabled in saved:
            lg.handlers.clear()
            for old in handlers:
                lg.addHandler(old)
            lg.setLevel(level)
            lg.propagate = propagate
            lg.disabled = disabled


def test_turn_on_debug_sets_all_named_loggers_to_debug():
    """Pin that ``_turn_on_debug()`` actually leaves all three named loggers
    at DEBUG -- otherwise the matrix would silently pass with no logs emitted.
    """
    loggers = [verbose_logger, verbose_proxy_logger, verbose_router_logger]
    saved = [(lg, lg.level) for lg in loggers]
    try:
        for lg in loggers:
            lg.setLevel(logging.WARNING)
        _turn_on_debug()
        for lg in loggers:
            assert lg.level == logging.DEBUG, (
                f"{lg.name} not at DEBUG after _turn_on_debug() "
                f"(got level={lg.level})"
            )
    finally:
        for lg, level in saved:
            lg.setLevel(level)


def test_secret_filter_is_attached_to_each_logger_via_fixture(debug_capture_buf):
    """Canary that the fixture wires ``_secret_filter`` onto every emission
    path the matrix exercises."""
    payload, marker = _NEG, _NEG_MARKER
    for _id, lg in _LOGGERS:
        debug_capture_buf.seek(0)
        debug_capture_buf.truncate(0)
        lg.debug("Provider key: " + payload)
        out = debug_capture_buf.getvalue()
        assert marker not in out, (
            f"Filter not attached to {lg.name} via the fixture -- leak in canary."
        )
