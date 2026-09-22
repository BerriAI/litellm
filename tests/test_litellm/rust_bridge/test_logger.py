import logging
from typing import Final

import pytest

import litellm
from litellm._logging import redact_secrets, session_id_var, trace_id_var, verbose_logger
from litellm.rust_bridge import logger


def test_native_records_preserve_metadata_and_redact_before_custom_handlers(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(verbose_logger, "handlers", [])
    secret: Final = "sk-" + "a" * 48
    message: Final = f"Authorization: Bearer {secret}"
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        logger.emit(
            logging.WARNING, message, "native.rs", 42, "litellm_http", {"retry": True, "api_key": secret}, ("", "")
        )

    record: Final = caplog.records[0]
    assert len(caplog.records) == 1
    assert record.getMessage() == redact_secrets(message)
    assert secret not in record.getMessage()
    assert (record.pathname, record.lineno, record.funcName) == ("native.rs", 42, "litellm_http")
    assert record.__dict__["rust_fields"]["retry"] is True
    assert secret not in str(record.__dict__["rust_fields"])


def test_native_context_is_scoped_and_respects_correlation_setting(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "request_correlation_in_logs", True)
    context_before: Final = logger.context()
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        logger.emit(logging.WARNING, "native", "native.rs", 1, "litellm_http", {}, ("session", "trace"))
        monkeypatch.setattr(litellm, "request_correlation_in_logs", False)
        logger.emit(logging.WARNING, "disabled", "native.rs", 2, "litellm_http", {}, ("hidden", "hidden"))

    first, second = caplog.records
    assert (first.__dict__["session_id"], first.__dict__["trace_id"]) == ("session", "trace")
    assert "session_id" not in second.__dict__
    assert "trace_id" not in second.__dict__
    assert (session_id_var.get(), trace_id_var.get()) == context_before


def test_native_logging_observes_level_changes(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR, logger="LiteLLM"):
        assert not logger.enabled(logging.WARNING)
        logger.emit(logging.WARNING, "filtered", "native.rs", 1, "litellm_http", {}, ("", ""))
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        assert logger.enabled(logging.WARNING)
        logger.emit(logging.WARNING, "visible", "native.rs", 1, "litellm_http", {}, ("", ""))

    assert [record.getMessage() for record in caplog.records] == ["visible"]
