import logging

import pytest


class TestScrubSecrets:
    """Tests for the _scrub_secrets() helper."""

    def test_api_key_value_redacted(self):
        from litellm._logging import _scrub_secrets

        result = _scrub_secrets("api_key=sk-abcdef123456789")
        assert "sk-abcdef" not in result
        assert "[REDACTED]" in result

    def test_aws_secret_key_redacted(self):
        from litellm._logging import _scrub_secrets

        result = _scrub_secrets("aws_secret_key: AKIAIOSFODNN7EXAMPLE")
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert "[REDACTED]" in result

    def test_encryption_key_redacted(self):
        from litellm._logging import _scrub_secrets

        result = _scrub_secrets("encryption_key = my-very-secret-key-123")
        assert "my-very-secret-key-123" not in result
        assert "[REDACTED]" in result

    def test_bearer_token_redacted(self):
        # CredentialScrubberFilter matches key=value / key:value format.
        # "Authorization: Bearer <token>" (space-separated) is handled by the
        # existing SecretRedactionFilter; here we test the key:value log pattern.
        from litellm._logging import _scrub_secrets

        result = _scrub_secrets("auth_token: eyJhbGciOiJSUzI1NiJ9abcdef")
        assert "eyJhbGciOiJSUzI1NiJ9abcdef" not in result
        assert "[REDACTED]" in result

    def test_non_secret_field_not_redacted(self):
        from litellm._logging import _scrub_secrets

        result = _scrub_secrets("model=gpt-4o, endpoint=https://api.openai.com")
        assert "gpt-4o" in result
        assert "api.openai.com" in result

    def test_short_value_not_redacted(self):
        # Values < 6 chars are not secrets (avoids redacting booleans/short flags)
        from litellm._logging import _scrub_secrets

        result = _scrub_secrets("api_key=abc")
        assert "abc" in result


class TestCredentialScrubberFilter:
    """Tests for CredentialScrubberFilter.filter() — covers every branch."""

    def _make_record(self, msg, args=None):
        # Always construct with args=() then set record.args directly.
        # Passing a dict to LogRecord.__init__ as args triggers a KeyError on
        # Python 3.13 (args[0] lookup on a str-keyed dict).
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
        if args is not None:
            record.args = args  # type: ignore[assignment]
        return record

    def test_string_msg_with_secret_is_redacted(self):
        from litellm._logging import CredentialScrubberFilter

        f = CredentialScrubberFilter()
        record = self._make_record("api_key=sk-secret123456789")
        f.filter(record)
        assert "sk-secret123456789" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_non_string_msg_untouched(self):
        # Branch: msg exists but is not a str — must not crash
        from litellm._logging import CredentialScrubberFilter

        f = CredentialScrubberFilter()
        record = self._make_record(12345)
        f.filter(record)
        assert record.msg == 12345

    def test_none_msg_untouched(self):
        # Branch: record.msg is falsy
        from litellm._logging import CredentialScrubberFilter

        f = CredentialScrubberFilter()
        record = self._make_record(None)
        f.filter(record)
        assert record.msg is None

    def test_dict_args_secret_redacted(self):
        # Branch: record.args is a dict whose string value contains a key=value secret
        from litellm._logging import CredentialScrubberFilter

        f = CredentialScrubberFilter()
        record = self._make_record("config %s", {"info": "api_key=sk-secret123456789"})
        f.filter(record)
        assert "sk-secret123456789" not in str(record.args)

    def test_dict_args_non_string_value_untouched(self):
        # Branch: record.args is a dict with a non-string value — must pass through
        from litellm._logging import CredentialScrubberFilter

        f = CredentialScrubberFilter()
        record = self._make_record("config %s", {"count": 42})
        f.filter(record)
        assert record.args == {"count": 42}  # type: ignore[comparison-overlap]

    def test_tuple_args_secret_redacted(self):
        # Branch: record.args is a tuple whose string element contains key=value secret
        from litellm._logging import CredentialScrubberFilter

        f = CredentialScrubberFilter()
        record = self._make_record("config: %s", ("api_key=sk-secret123456789",))
        f.filter(record)
        assert "sk-secret123456789" not in str(record.args)

    def test_tuple_args_mixed_types_non_string_passthrough(self):
        # Branch: tuple contains non-string elements — ints/None pass through untouched
        from litellm._logging import CredentialScrubberFilter

        f = CredentialScrubberFilter()
        record = self._make_record(
            "vals=%s %s %s", (42, None, "api_key=sk-secret123456789")
        )
        f.filter(record)
        args = record.args
        assert isinstance(args, tuple)
        assert args[0] == 42
        assert args[1] is None
        assert "sk-secret123456789" not in str(args[2])

    def test_no_args_no_crash(self):
        # Branch: record.args is falsy (empty tuple)
        from litellm._logging import CredentialScrubberFilter

        f = CredentialScrubberFilter()
        record = self._make_record("plain message with no args")
        f.filter(record)
        assert record.msg == "plain message with no args"

    def test_filter_always_returns_true(self):
        # Filter must never drop log records
        from litellm._logging import CredentialScrubberFilter

        f = CredentialScrubberFilter()
        record = self._make_record("api_key=sk-secret123456789")
        result = f.filter(record)
        assert result is True

    def test_filter_registered_on_verbose_logger(self):
        import litellm._logging as log_module

        filter_types = [type(f).__name__ for f in log_module.verbose_logger.filters]
        assert "CredentialScrubberFilter" in filter_types

    def test_filter_registered_on_verbose_proxy_logger(self):
        import litellm._logging as log_module

        filter_types = [
            type(f).__name__ for f in log_module.verbose_proxy_logger.filters
        ]
        assert "CredentialScrubberFilter" in filter_types

    def test_filter_registered_on_verbose_router_logger(self):
        import litellm._logging as log_module

        filter_types = [
            type(f).__name__ for f in log_module.verbose_router_logger.filters
        ]
        assert "CredentialScrubberFilter" in filter_types
