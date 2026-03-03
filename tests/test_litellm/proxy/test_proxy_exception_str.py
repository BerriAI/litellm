"""
Tests for ProxyException string representation.

Verifies that str(ProxyException(...)) returns the exception message
instead of an empty string. See https://github.com/BerriAI/litellm/issues/22644
"""

from litellm.proxy._types import ProxyException


class TestProxyExceptionStr:
    """Tests for ProxyException.__str__() behavior."""

    def test_proxy_exception_str_returns_message(self):
        """str(ProxyException(...)) should return the message, not empty string."""
        exc = ProxyException(
            message="auth failed",
            type="bad_request_error",
            param="key_alias",
            code=400,
        )
        assert str(exc) == "auth failed"

    def test_proxy_exception_str_empty_message(self):
        """str() should work correctly with an empty message."""
        exc = ProxyException(
            message="",
            type="bad_request_error",
            param="key_alias",
            code=400,
        )
        assert str(exc) == ""

    def test_proxy_exception_str_unicode(self):
        """str() should handle unicode messages correctly."""
        msg = "Error: \u2018invalid key\u2019 please retry \U0001f512"
        exc = ProxyException(
            message=msg,
            type="bad_request_error",
            param="key_alias",
            code=400,
        )
        assert str(exc) == msg

    def test_proxy_exception_repr(self):
        """repr() should include the message text."""
        exc = ProxyException(
            message="something broke",
            type="internal_error",
            param=None,
            code=500,
        )
        assert "something broke" in repr(exc)

    def test_proxy_exception_to_dict_unchanged(self):
        """to_dict() output format must remain exactly the same after the fix."""
        exc = ProxyException(
            message="model not found",
            type="invalid_request_error",
            param="model",
            code=404,
        )
        result = exc.to_dict()
        assert result == {
            "message": "model not found",
            "type": "invalid_request_error",
            "param": "model",
            "code": "404",
        }

    def test_proxy_exception_to_dict_with_provider_fields(self):
        """to_dict() should include provider_specific_fields when present."""
        exc = ProxyException(
            message="rate limited",
            type="rate_limit_error",
            param=None,
            code=429,
            provider_specific_fields={"retry_after": 30},
        )
        result = exc.to_dict()
        assert result["provider_specific_fields"] == {"retry_after": 30}
        assert result["message"] == "rate limited"

    def test_proxy_exception_chaining(self):
        """raise ProxyException from ValueError should preserve __cause__."""
        inner = ValueError("inner error")
        try:
            try:
                raise inner
            except ValueError:
                raise ProxyException(
                    message="outer error",
                    type="bad_request_error",
                    param="test",
                    code=400,
                ) from inner
        except ProxyException as exc:
            assert str(exc) == "outer error"
            assert exc.__cause__ is inner

    def test_proxy_exception_catch_and_stringify(self):
        """Simulates the real-world pattern: try/except that formats str(e) into a log."""
        log_output = ""
        try:
            raise ProxyException(
                message="key expired",
                type="authentication_error",
                param="api_key",
                code=401,
            )
        except ProxyException as exc:
            log_output = "Error - {}".format(str(exc))

        assert log_output == "Error - key expired"
