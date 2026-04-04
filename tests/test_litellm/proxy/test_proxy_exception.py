"""Tests for ProxyException.__str__.

Related issue: https://github.com/BerriAI/litellm/issues/22644
"""

from litellm.proxy._types import ProxyException


def test_proxy_exception_str_returns_message():
    exc = ProxyException(
        message="This is an error",
        type="invalid_request_error",
        param="model",
        code=400,
    )
    assert str(exc) == "This is an error"


def test_proxy_exception_str_with_non_string_message():
    exc = ProxyException(
        message=42,
        type="server_error",
        param=None,
        code=500,
    )
    # __init__ already calls str(message)
    assert str(exc) == "42"
