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


class TestExternalLoggersNotConfigured:
    """Importing litellm must NOT modify external loggers at import time."""

    def test_httpx_logger_not_modified_at_import(self):
        out = _run_in_subprocess("""
            import logging
            httpx_level_before = logging.getLogger("httpx").level
            import litellm
            httpx_level_after = logging.getLogger("httpx").level
            print(httpx_level_before == httpx_level_after)
        """)
        assert out == "True"

    def test_apscheduler_logger_not_modified_at_import(self):
        out = _run_in_subprocess("""
            import logging
            lvl_before = logging.getLogger("apscheduler.scheduler").level
            import litellm
            lvl_after = logging.getLogger("apscheduler.scheduler").level
            print(lvl_before == lvl_after)
        """)
        assert out == "True"


class TestSetVerboseStillWorks:
    """_turn_on_debug must add a handler so users can actually see output."""

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
        from litellm._logging import verbose_proxy_logger

        assert verbose_proxy_logger.propagate is True

    def test_router_logger_propagates(self):
        from litellm._logging import verbose_router_logger

        assert verbose_router_logger.propagate is True
