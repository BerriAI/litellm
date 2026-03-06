"""
Tests for litellm._logging — verifies that the library follows Python logging
best practices and does not pollute the root or external loggers.

Fixes https://github.com/BerriAI/litellm/issues/16284
"""

import logging
import subprocess
import sys
import textwrap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_in_subprocess(code: str) -> str:
    """Run *code* in a fresh Python process so logger state is pristine."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoggerNamespace:
    """Loggers must live under the 'litellm' namespace."""

    def test_verbose_logger_name(self):
        from litellm._logging import verbose_logger

        assert verbose_logger.name == "litellm"

    def test_verbose_proxy_logger_name(self):
        from litellm._logging import verbose_proxy_logger

        assert verbose_proxy_logger.name == "litellm.proxy"

    def test_verbose_router_logger_name(self):
        from litellm._logging import verbose_router_logger

        assert verbose_router_logger.name == "litellm.router"


class TestNullHandlerDefault:
    """The library must ship with only a NullHandler on the parent logger."""

    def test_litellm_logger_has_null_handler(self):
        from litellm._logging import verbose_logger

        handler_types = [type(h) for h in verbose_logger.handlers]
        assert logging.NullHandler in handler_types

    def test_litellm_logger_no_stream_handler_by_default(self):
        """Importing litellm must NOT attach a StreamHandler."""
        out = _run_in_subprocess("""
            import logging, litellm
            from litellm._logging import verbose_logger
            has_stream = any(
                isinstance(h, logging.StreamHandler)
                and not isinstance(h, logging.NullHandler)
                for h in verbose_logger.handlers
            )
            print(has_stream)
        """)
        assert out == "False"


class TestRootLoggerNotTouched:
    """Importing litellm must NOT add handlers to the root logger."""

    def test_root_logger_no_handlers_after_import(self):
        out = _run_in_subprocess("""
            import logging
            root_before = len(logging.getLogger().handlers)
            import litellm
            root_after = len(logging.getLogger().handlers)
            print(root_after - root_before)
        """)
        assert out == "0", f"Root logger gained handlers after import: delta={out}"

    def test_root_logger_not_in_all_loggers(self):
        from litellm._logging import ALL_LOGGERS

        root = logging.getLogger()
        assert root not in ALL_LOGGERS


class TestExternalLoggersSuppressedAtImport:
    """Importing litellm must suppress noisy third-party loggers (httpx, apscheduler)."""

    def test_httpx_logger_suppressed_at_import(self):
        out = _run_in_subprocess("""
            import logging
            import litellm
            print(logging.getLogger("httpx").level >= logging.WARNING)
        """)
        assert out == "True"

    def test_apscheduler_logger_suppressed_at_import(self):
        out = _run_in_subprocess("""
            import logging
            import litellm
            print(logging.getLogger("apscheduler.scheduler").level >= logging.WARNING)
        """)
        assert out == "True"


class TestSetVerboseStillWorks:
    """_turn_on_debug must add a handler so users can actually see output."""

    def setup_method(self):
        """Snapshot logger state before each test."""
        from litellm._logging import verbose_logger, verbose_proxy_logger, verbose_router_logger

        self._orig_level = verbose_logger.level
        self._orig_proxy_level = verbose_proxy_logger.level
        self._orig_router_level = verbose_router_logger.level
        self._orig_handlers = list(verbose_logger.handlers)

    def teardown_method(self):
        """Restore logger state after each test."""
        from litellm._logging import verbose_logger, verbose_proxy_logger, verbose_router_logger

        # Remove any StreamHandlers added during test (keep NullHandler)
        verbose_logger.handlers[:] = [
            h for h in verbose_logger.handlers
            if isinstance(h, logging.NullHandler)
        ]
        # Re-add original handlers that aren't already present
        for h in self._orig_handlers:
            if h not in verbose_logger.handlers:
                verbose_logger.handlers.append(h)
        verbose_logger.setLevel(self._orig_level)
        verbose_proxy_logger.setLevel(self._orig_proxy_level)
        verbose_router_logger.setLevel(self._orig_router_level)

    def test_turn_on_debug_adds_stream_handler(self):
        from litellm._logging import _turn_on_debug, verbose_logger

        _turn_on_debug()
        has_stream = any(
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.NullHandler)
            for h in verbose_logger.handlers
        )
        assert has_stream

    def test_turn_on_debug_sets_debug_level(self):
        from litellm._logging import (
            _turn_on_debug,
            verbose_logger,
            verbose_proxy_logger,
            verbose_router_logger,
        )

        _turn_on_debug()
        assert verbose_logger.level == logging.DEBUG
        assert verbose_proxy_logger.level == logging.DEBUG
        assert verbose_router_logger.level == logging.DEBUG

    def test_turn_on_debug_idempotent(self):
        """Calling _turn_on_debug twice must not duplicate handlers."""
        from litellm._logging import _turn_on_debug, verbose_logger

        _turn_on_debug()
        count_before = len(verbose_logger.handlers)
        _turn_on_debug()
        count_after = len(verbose_logger.handlers)
        assert count_after == count_before


class TestChildLoggerPropagation:
    """Child loggers (proxy, router) should propagate to the parent by default."""

    def test_proxy_logger_propagates(self):
        """Test in subprocess to isolate from JSON_LOGS environment."""
        out = _run_in_subprocess("""
            from litellm._logging import verbose_proxy_logger
            print(verbose_proxy_logger.propagate)
        """)
        assert out == "True"

    def test_router_logger_propagates(self):
        """Test in subprocess to isolate from JSON_LOGS environment."""
        out = _run_in_subprocess("""
            from litellm._logging import verbose_router_logger
            print(verbose_router_logger.propagate)
        """)
        assert out == "True"


class TestLegacyLoggerBackwardCompat:
    """Handlers attached to the old logger names receive records from the new canonical loggers."""

    def test_handler_on_old_name_receives_canonical_records(self):
        """A handler on 'LiteLLM' should receive records emitted to 'litellm'."""
        import logging

        from litellm._logging import verbose_logger

        old = logging.getLogger("LiteLLM")
        captured = []
        original_verbose_logger_level = verbose_logger.level

        class _Capture(logging.Handler):
            def emit(self, record):
                captured.append(record)

        handler = _Capture()
        old.addHandler(handler)
        old.setLevel(logging.DEBUG)
        try:
            verbose_logger.warning("backward-compat-test")
            assert any("backward-compat-test" in r.getMessage() for r in captured), (
                "Handler on old 'LiteLLM' logger did not receive record from canonical 'litellm'"
            )
        finally:
            old.removeHandler(handler)
            verbose_logger.setLevel(original_verbose_logger_level)

    def test_old_and_canonical_share_handlers(self):
        """Old and canonical loggers must share the same handlers list object."""
        import logging

        from litellm._logging import verbose_logger

        old = logging.getLogger("LiteLLM")
        assert old.handlers is verbose_logger.handlers

    def test_set_level_on_alias_propagates_to_canonical(self):
        """setLevel on the legacy alias must also set the canonical logger's level."""
        import logging

        from litellm._logging import verbose_logger

        old = logging.getLogger("LiteLLM")
        original_level = verbose_logger.level
        try:
            old.setLevel(logging.DEBUG)
            assert verbose_logger.level == logging.DEBUG, (
                "setLevel on alias 'LiteLLM' did not propagate to canonical 'litellm'"
            )
        finally:
            verbose_logger.setLevel(original_level)


class TestSdkJsonLogsInit:
    """Verify SDK users with JSON_LOGS=true get JSON handlers at import time."""

    def test_json_logs_attaches_handler_at_import(self):
        """When JSON_LOGS=true, importing litellm must attach a StreamHandler."""
        import subprocess
        import sys

        # Must test in a subprocess because module-level code only runs once.
        result = subprocess.run(
            [sys.executable, "-c", """
import os
os.environ["JSON_LOGS"] = "true"
import logging
from litellm._logging import verbose_logger
stream_handlers = [
    h for h in verbose_logger.handlers
    if isinstance(h, logging.StreamHandler)
    and not isinstance(h, logging.NullHandler)
]
assert len(stream_handlers) >= 1, "No StreamHandler found for JSON_LOGS=true SDK path"
print("OK")
"""],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"SDK JSON_LOGS=true test failed: {result.stderr}"
        )
