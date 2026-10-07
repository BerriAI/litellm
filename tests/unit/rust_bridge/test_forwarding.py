import logging
import sys
from io import StringIO
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

from litellm._logging import session_id_var, set_session_id, set_trace_id, trace_id_var
from litellm._logging import diagnostic_snapshot
from litellm.rust_bridge import logger as native_logger
from litellm.rust_bridge.forwarding import NATIVE_ORIGIN, DiagnosticEmitter, ForwardingHandler, install

_FIELDS: Final = TypeAdapter(dict[str, JsonValue])


class RecordingEmitter:
    def __init__(self) -> None:
        self.events: tuple[tuple[int, str, str], ...] = ()

    def emit(self, severity: int, message: str, fields: str) -> None:
        self.events = (*self.events, (severity, message, fields))


def test_forwarding_preserves_original_record_and_python_handler_output() -> None:
    emitter: Final = RecordingEmitter()
    output: Final = StringIO()
    legacy: Final = logging.StreamHandler(output)
    logger: Final = logging.Logger("LiteLLM.custom", logging.DEBUG)
    logger.addHandler(ForwardingHandler(lambda: emitter, lambda: True))
    logger.addHandler(legacy)
    arguments: Final = ("kept",)
    extra: Final = {"nested": {"attempt": 3, "retry": True}}
    try:
        raise ValueError("original exception")
    except ValueError:
        exception: Final = sys.exc_info()
    record: Final = logger.makeRecord(
        logger.name, 35, "caller.py", 19, "message %s", arguments, exception, "caller", extra
    )
    record.created = 1_700_000_000.5  # rebind-ok: a historical timestamp verifies projection independently of the clock
    session_token: Final = set_session_id("session")
    trace_token: Final = set_trace_id("trace")
    try:
        logger.handle(record)
    finally:
        trace_id_var.reset(trace_token)
        session_id_var.reset(session_token)

    severity, message, serialized = emitter.events[0]
    fields: Final = _FIELDS.validate_json(serialized)
    assert (severity, message) == (35, "message kept")
    assert fields["logger.name"] == "LiteLLM.custom"
    assert fields["python.levelno"] == 35
    assert fields["python.created"] == 1_700_000_000.5
    assert (fields["code.filepath"], fields["code.lineno"], fields["code.function"]) == ("caller.py", 19, "caller")
    assert (fields["session_id"], fields["trace_id"]) == ("session", "trace")
    assert fields["extra"] == extra
    assert "ValueError: original exception" in str(fields["exception.stacktrace"])
    assert record.args is arguments
    assert record.exc_info is exception
    assert record.__dict__["nested"] is extra["nested"]
    assert output.getvalue().startswith("message kept\nTraceback")
    assert output.getvalue().endswith("ValueError: original exception\n")


@pytest.mark.parametrize("active,available", ((False, True), (True, False)))
def test_disabled_or_missing_native_keeps_python_output(active: bool, available: bool) -> None:
    emitter: Final = RecordingEmitter()
    output: Final = StringIO()
    logger: Final = logging.Logger("LiteLLM.custom")
    logger.addHandler(ForwardingHandler(lambda: emitter if available else None, lambda: active))
    logger.addHandler(logging.StreamHandler(output))
    logger.warning("visible %s", "python")
    assert output.getvalue() == "visible python\n"
    assert emitter.events == ()


class FailedEmitter:
    def emit(self, severity: int, message: str, fields: str) -> None:
        raise RuntimeError("export failed")


def test_native_failure_does_not_interrupt_subsequent_python_handlers() -> None:
    emitter: Final[DiagnosticEmitter] = FailedEmitter()
    handler: Final = ForwardingHandler(lambda: emitter, lambda: True)
    output: Final = StringIO()
    logger: Final = logging.Logger("LiteLLM.custom")
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(output))
    logger.error("still visible")
    assert output.getvalue() == "still visible\n"
    assert handler.failures == 1


def test_native_origin_and_logger_rejection_prevent_duplicate_delivery() -> None:
    emitter: Final = RecordingEmitter()
    handler: Final = ForwardingHandler(lambda: emitter, lambda: True)
    logger: Final = logging.Logger("LiteLLM.custom")
    logger.addHandler(handler)
    logger.warning("from rust", extra={"_litellm_native_origin": NATIVE_ORIGIN})
    logger.addFilter(lambda record: False)
    logger.warning("rejected by host filter")
    assert emitter.events == ()


def test_install_is_additive_and_idempotent_without_changing_levels_or_propagation() -> None:
    output: Final = StringIO()
    legacy: Final = logging.StreamHandler(output)
    logger: Final = logging.Logger("custom", logging.ERROR)
    logger.addHandler(legacy)
    install((logger,))
    install((logger,))
    logger.warning("filtered")
    logger.error("visible")
    assert output.getvalue() == "visible\n"
    assert logger.handlers[0] is legacy
    assert len(tuple(handler for handler in logger.handlers if isinstance(handler, ForwardingHandler))) == 1
    assert logger.level == logging.ERROR
    assert logger.propagate is True


@pytest.mark.parametrize("propagate", (True, False))
def test_install_avoids_forwarding_twice_and_respects_nonpropagating_children(propagate: bool) -> None:
    emitter: Final = RecordingEmitter()
    parent: Final = logging.Logger("parent")
    child: Final = logging.Logger("parent.child")
    child.parent = parent
    child.propagate = propagate
    parent.addHandler(ForwardingHandler(lambda: emitter, lambda: True))
    child.addHandler(ForwardingHandler(lambda: emitter, lambda: True))
    install((child, parent))
    child.warning("once")
    assert len(emitter.events) == 1
    assert emitter.events[0][1] == "once"


def test_extra_serialization_can_log_without_recursively_forwarding() -> None:
    emitter: Final = RecordingEmitter()
    logger: Final = logging.Logger("custom")
    logger.addHandler(ForwardingHandler(lambda: emitter, lambda: True))

    class LoggingValue:
        def __str__(self) -> str:
            logger.warning("during serialization")
            return "value"

    logger.warning("outer", extra={"value": LoggingValue()})
    assert len(emitter.events) == 1
    assert emitter.events[0][1] == "outer"


def test_native_compatibility_redaction_preserves_the_origin_guard() -> None:
    from litellm._logging import verbose_logger

    emitter: Final = RecordingEmitter()
    handler: Final = ForwardingHandler(lambda: emitter, lambda: True)
    verbose_logger.addHandler(handler)
    try:
        native_logger.emit(logging.ERROR, "from rust", "native.rs", 1, "litellm_test", {}, ("", ""))
    finally:
        verbose_logger.removeHandler(handler)
    assert emitter.events == ()


def test_owned_projection_schema_preserves_source_fields_and_excludes_internal_markers() -> None:
    record: Final = logging.Logger("custom").makeRecord(
        "custom",
        logging.WARNING,
        "caller.py",
        19,
        "message %s",
        ("kept",),
        None,
        "caller",
        {"nested": {"retry": True}, "_litellm_native_origin": NATIVE_ORIGIN},
    )
    record.created = 1_700_000_000.5  # rebind-ok: fixed source time makes the projection contract deterministic
    session_token: Final = set_session_id("session")
    trace_token: Final = set_trace_id("trace")
    try:
        message, serialized = diagnostic_snapshot(record)
    finally:
        trace_id_var.reset(trace_token)
        session_id_var.reset(session_token)
    assert message == "message kept"
    assert _FIELDS.validate_json(serialized) == {
        "source.target": "custom",
        "source.timestamp": 1_700_000_000.5,
        "source.language": "python",
        "logger.name": "custom",
        "python.levelno": logging.WARNING,
        "python.levelname": "WARNING",
        "python.created": 1_700_000_000.5,
        "code.filepath": "caller.py",
        "code.lineno": 19,
        "code.function": "caller",
        "exception.stacktrace": None,
        "stack": None,
        "session_id": "session",
        "trace_id": "trace",
        "extra": {"nested": {"retry": True}},
    }
