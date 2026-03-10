import logging
import os
import sys
from io import StringIO

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm._logging import (
    SecretRedactionFilter,
    _redact_string,
    _secret_filter,
    verbose_logger,
    verbose_proxy_logger,
    verbose_router_logger,
)


def test_redact_string_covers_all_secret_formats():
    secrets = {
        "AWS access key": "AKIAIOSFODNN7EXAMPLE",
        "AWS temp key": "ASIAISAMPLEKEYID1234",
        "AWS secret": "aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "AWS session token": "aws_session_token: FwoGZXIvYXdzEBYaDHqa0AP1RIF0re2EXAMPLETOKEN1234567890",
        "Bearer token": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig",
        "OpenAI key": "sk-proj-abc123def456ghi789jklmnopqrst",
        "x-api-key header": "x-api-key: mysecretapikey123",
        "api-key header": "api-key: mysecretapikey123",
        "Azure key": "api_key=a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
        "Google key": "AIzaSyA1234567890abcdefghijklmnopqrstuvwx",
        "Anthropic x-ak": "x-ak-abcdefghijklmnopqrstuvwxyz12345",
        "Dict repr with secrets": "Request Headers: {'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig', 'x-api-key': 'sk-ant-api03-abcdefghijklmnopqrstuvwxyz'}",
    }
    for label, secret in secrets.items():
        result = _redact_string(f"msg: {secret}")
        assert secret not in result, f"{label} was not redacted"
        assert "REDACTED" in result, f"{label} missing REDACTED marker"

    normal = "Loaded model gpt-4 with 3 replicas on us-east-1"
    assert _redact_string(normal) == normal


def test_filter_redacts_secrets_in_logger_output():
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
        verbose_logger.debug(f"Credentials: AKIAIOSFODNN7EXAMPLE")
        verbose_proxy_logger.debug(f"Headers: Bearer eyJhbGciOiJSUzI1NiJ9.payload.sig")
        verbose_router_logger.debug(f"Key: sk-proj-abc123def456ghi789jklmnopqrst")
        verbose_logger.debug("Normal message with no secrets")

        output = buf.getvalue()
        assert "AKIAIOSFODNN7EXAMPLE" not in output
        assert "eyJhbGciOiJSUzI1NiJ9" not in output
        assert "sk-proj-abc123def456ghi789jklmnopqrst" not in output
        assert "REDACTED" in output
        assert "Normal message with no secrets" in output
    finally:
        for lg, handlers, level in saved:
            lg.handlers.clear()
            for old_h in handlers:
                lg.addHandler(old_h)
            lg.setLevel(level)
