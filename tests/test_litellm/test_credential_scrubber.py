import logging


class TestScrubSecrets:
    """Tests for the _scrub_secrets() helper."""

    def test_known_patterns_are_redacted(self):
        from litellm._logging import _scrub_secrets

        for text, secret in [
            ("api_key=sk-abcdef123456789", "sk-abcdef"),
            ("aws_secret_key: AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE"),
            ("access_token: eyJhbGciOiJSUzI1NiJ9abcdef", "eyJhbGciOiJSUzI1NiJ9abcdef"),
        ]:
            assert secret not in _scrub_secrets(text)

    def test_non_secrets_pass_through(self):
        # Non-secret fields and values < 6 chars are not redacted
        from litellm._logging import _scrub_secrets

        result = _scrub_secrets(
            "model=gpt-4o, endpoint=https://api.openai.com, api_key=abc"
        )
        assert "gpt-4o" in result
        assert "api.openai.com" in result
        assert "abc" in result  # < 6 chars, not redacted

    def test_pem_value_not_partially_redacted(self):
        from litellm._logging import _scrub_secrets

        pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----"
        result = _scrub_secrets(f"private_key = {pem}")
        assert "MIIE" not in result or "-----BEGIN" in result


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
        result = f.filter(record)
        assert result is True  # filter must never drop records
        assert "sk-secret123456789" not in record.msg

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

    def test_filter_respects_disable_redaction_env_var(self):
        # Branch: _ENABLE_SECRET_REDACTION is False — filter must pass record unmodified
        from unittest.mock import patch

        from litellm._logging import CredentialScrubberFilter

        with patch("litellm._logging._ENABLE_SECRET_REDACTION", False):
            f = CredentialScrubberFilter()
            record = self._make_record("api_key=sk-secret123456789")
            f.filter(record)
            assert "sk-secret123456789" in record.msg

    def test_dict_args_secret_key_name_redacts_raw_value(self):
        # Branch: dict key is itself a secret field name — raw value must be redacted
        # even without a key=value pattern inside the value string.
        from litellm._logging import CredentialScrubberFilter

        f = CredentialScrubberFilter()
        record = self._make_record("config %s", {"api_key": "sk-rawsecretvalue123"})
        f.filter(record)
        assert "sk-rawsecretvalue123" not in str(record.args)
        assert "[REDACTED]" in str(record.args)

    def test_dict_args_secret_key_non_string_value_redacted(self):
        # Branch: dict key is a secret name but value is not a str (e.g. bytes).
        # Key-name check must fire regardless of value type.
        from litellm._logging import CredentialScrubberFilter

        f = CredentialScrubberFilter()
        record = self._make_record("config %s", {"api_key": b"sk-bytessecret123"})
        f.filter(record)
        assert "sk-bytessecret123" not in str(record.args)
        assert "[REDACTED]" in str(record.args)

    def test_exc_traceback_secret_redacted(self):
        # Branch: record.exc_info — secret in exception message is scrubbed from exc_text
        import sys

        from litellm._logging import CredentialScrubberFilter

        try:
            raise ValueError("api_key=sk-secretinexception12345")
        except ValueError:
            ei = sys.exc_info()
        f = CredentialScrubberFilter()
        record = self._make_record("error occurred")
        record.exc_info = ei
        f.filter(record)
        assert record.exc_text is not None
        assert "sk-secretinexception12345" not in record.exc_text

    def test_extra_fields_secret_redacted(self):
        # Branch: extra fields via logger.debug("msg", extra={...})
        # Secret-named key → value replaced with [REDACTED]
        # Non-secret key with embedded key=value secret → value scrubbed inline
        from litellm._logging import CredentialScrubberFilter

        f = CredentialScrubberFilter()
        record = self._make_record("msg")
        record.api_key = "sk-extrasecret123456789"  # type: ignore[attr-defined]
        record.debug_info = "api_key=sk-embeddedsecret12345"  # type: ignore[attr-defined]
        f.filter(record)
        assert getattr(record, "api_key") == "[REDACTED]"
        assert "sk-embeddedsecret12345" not in getattr(record, "debug_info", "")
