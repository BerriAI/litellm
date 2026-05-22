import logging
import sys
from decimal import Decimal
from unittest.mock import patch

import litellm._logging as log_module
from litellm._logging import CredentialScrubberFilter, _scrub_secrets


class TestScrubSecrets:

    def test_known_patterns_are_redacted(self):
        for text, secret in [
            ("api_key=sk-abcdef123456789", "sk-abcdef"),
            ("aws_secret_key: AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE"),
            ("access_token: eyJhbGciOiJSUzI1NiJ9abcdef", "eyJhbGciOiJSUzI1NiJ9abcdef"),
        ]:
            assert secret not in _scrub_secrets(text)

    def test_non_secrets_pass_through(self):
        result = _scrub_secrets(
            "model=gpt-4o, endpoint=https://api.openai.com, api_key=abc"
        )
        assert "gpt-4o" in result
        assert "api.openai.com" in result
        assert "abc" in result  # < 6 chars, not redacted

    def test_pem_value_not_partially_redacted(self):
        pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----"
        result = _scrub_secrets(f"private_key = {pem}")
        # PEM body must not appear as an orphan without its surrounding structure.
        # Header-only (body redacted) is acceptable; body-only (orphan fragment) is not.
        if "MIIE" in result:
            assert "-----BEGIN RSA PRIVATE KEY-----" in result


class TestCredentialScrubberFilter:

    def _make_record(self, msg, args=None):
        # Construct with args=() then set record.args directly — passing a dict
        # to LogRecord.__init__ triggers a KeyError on Python 3.13.
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
        f = CredentialScrubberFilter()
        record = self._make_record("api_key=sk-secret123456789")
        assert f.filter(record) is True
        assert "sk-secret123456789" not in record.msg

    def test_non_string_msg_untouched(self):
        f = CredentialScrubberFilter()
        record = self._make_record(12345)
        f.filter(record)
        assert record.msg == 12345

    def test_none_msg_untouched(self):
        f = CredentialScrubberFilter()
        record = self._make_record(None)
        f.filter(record)
        assert record.msg is None

    def test_dict_args_secret_redacted(self):
        f = CredentialScrubberFilter()
        record = self._make_record("config %s", {"info": "api_key=sk-secret123456789"})
        f.filter(record)
        assert "sk-secret123456789" not in str(record.args)

    def test_dict_args_non_secret_int_value_preserved(self):
        # Non-secret key + int value: original type must survive (no str() coercion).
        f = CredentialScrubberFilter()
        record = self._make_record("config %s", {"count": 42})
        f.filter(record)
        assert record.args == {"count": 42}  # type: ignore[comparison-overlap]

    def test_dict_args_non_string_key_no_crash(self):
        # Non-string dict keys (unusual but legal) must not raise TypeError.
        f = CredentialScrubberFilter()
        record = self._make_record("config %s", {1: "api_key=sk-secret123456789"})
        f.filter(record)  # must not raise
        assert "sk-secret123456789" not in str(record.args)

    def test_dict_args_nested_dict_value_scrubbed(self):
        # Non-string nested dict under a non-secret key must still be scrubbed.
        f = CredentialScrubberFilter()
        record = self._make_record(
            "headers %s", {"headers": {"Authorization": "Bearer sk-nestednested1234"}}
        )
        f.filter(record)
        assert "sk-nestednested1234" not in str(record.args)

    def test_tuple_args_secret_redacted(self):
        f = CredentialScrubberFilter()
        record = self._make_record("config: %s", ("api_key=sk-secret123456789",))
        f.filter(record)
        assert "sk-secret123456789" not in str(record.args)

    def test_tuple_args_mixed_types_non_string_passthrough(self):
        f = CredentialScrubberFilter()
        record = self._make_record(
            "vals=%s %s %s %s",
            (
                42,
                None,
                "api_key=sk-secret123456789",
                {"api_key": "sk-dictval12345678901"},
            ),
        )
        f.filter(record)
        args = record.args
        assert isinstance(args, tuple)
        assert args[0] == 42
        assert args[1] is None
        assert "sk-secret123456789" not in str(args[2])
        assert "sk-dictval12345678901" not in str(args[3])

    def test_tuple_args_numeric_type_preserved_when_no_secret(self):
        # Decimal has no secret: original object must be kept so %f/%d still works.
        f = CredentialScrubberFilter()
        record = self._make_record("count=%s", (Decimal("42"),))
        f.filter(record)
        assert isinstance(record.args[0], Decimal)  # type: ignore[index]

    def test_bare_exception_arg_scrubbed(self):
        # verbose_logger.error("msg", exc) sets record.args = exc (not a tuple).
        exc = ValueError("api_key=sk-bareexception12345")
        f = CredentialScrubberFilter()
        record = self._make_record("error %s")
        record.args = exc  # type: ignore[assignment]
        f.filter(record)
        assert "sk-bareexception12345" not in str(record.args)

    def test_no_args_no_crash(self):
        f = CredentialScrubberFilter()
        record = self._make_record("plain message with no args")
        f.filter(record)
        assert record.msg == "plain message with no args"

    def test_filter_registered_on_all_loggers(self):
        for logger in (
            log_module.verbose_logger,
            log_module.verbose_proxy_logger,
            log_module.verbose_router_logger,
        ):
            assert any(
                type(f).__name__ == "CredentialScrubberFilter" for f in logger.filters
            )

    def test_filter_respects_disable_redaction_env_var(self):
        with patch("litellm._logging._ENABLE_SECRET_REDACTION", False):
            f = CredentialScrubberFilter()
            record = self._make_record("api_key=sk-secret123456789")
            f.filter(record)
            assert "sk-secret123456789" in record.msg

    def test_dict_args_secret_key_name_redacts_raw_value(self):
        f = CredentialScrubberFilter()
        record = self._make_record("config %s", {"api_key": "sk-rawsecretvalue123"})
        f.filter(record)
        assert "sk-rawsecretvalue123" not in str(record.args)
        assert "REDACTED" in str(record.args)

    def test_dict_args_secret_key_non_string_value_redacted(self):
        # Key-name check fires for non-string values too (e.g. bytes).
        f = CredentialScrubberFilter()
        record = self._make_record("config %s", {"api_key": b"sk-bytessecret123"})
        f.filter(record)
        assert "sk-bytessecret123" not in str(record.args)
        assert "REDACTED" in str(record.args)

    def test_exc_traceback_secret_redacted(self):
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
        f = CredentialScrubberFilter()
        record = self._make_record("msg")
        record.api_key = "sk-extrasecret123456789"  # type: ignore[attr-defined]
        record.debug_info = "api_key=sk-embeddedsecret12345"  # type: ignore[attr-defined]
        f.filter(record)
        assert getattr(record, "api_key") == "REDACTED"
        assert "sk-embeddedsecret12345" not in getattr(record, "debug_info", "")

    def test_exc_format_error_is_silently_ignored(self):
        # When formatException() raises, filter must not propagate the error.
        try:
            raise ValueError("api_key=sk-secretinexception12345")
        except ValueError:
            ei = sys.exc_info()
        f = CredentialScrubberFilter()
        record = self._make_record("error occurred")
        record.exc_info = ei
        with patch(
            "logging.Formatter.formatException", side_effect=RuntimeError("fmt failed")
        ):
            f.filter(record)
        assert record.exc_text is None
