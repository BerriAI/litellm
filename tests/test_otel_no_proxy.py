"""
Tests for OTel tracing without proxy dependencies.

Verifies that the OpenTelemetry integration can be imported, initialized,
and used without requiring litellm[proxy] dependencies.

Fixes https://github.com/BerriAI/litellm/issues/13081
"""

import ast
import enum
import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


class TestSpanAttributesLocation(unittest.TestCase):
    """Test that SpanAttributes is importable from the new shared location."""

    def test_import_from_types_integrations_otel(self):
        """SpanAttributes should be importable from litellm.types.integrations.otel."""
        from litellm.types.integrations.otel import SpanAttributes

        assert issubclass(SpanAttributes, enum.Enum)
        assert SpanAttributes.LLM_SYSTEM.value == "gen_ai.system"

    def test_backward_compat_import_from_proxy_types(self):
        """SpanAttributes should still be importable from litellm.proxy._types."""
        from litellm.proxy._types import SpanAttributes

        assert SpanAttributes.LLM_SYSTEM.value == "gen_ai.system"

    def test_same_class_both_locations(self):
        """Both import paths should resolve to the exact same class."""
        from litellm.proxy._types import SpanAttributes as ProxySpanAttributes
        from litellm.types.integrations.otel import SpanAttributes as OtelSpanAttributes

        assert ProxySpanAttributes is OtelSpanAttributes

    def test_all_expected_attributes_present(self):
        """SpanAttributes should contain all expected OTel semantic convention attributes."""
        from litellm.types.integrations.otel import SpanAttributes

        expected = [
            "LLM_SYSTEM",
            "LLM_REQUEST_MODEL",
            "LLM_REQUEST_MAX_TOKENS",
            "LLM_REQUEST_TEMPERATURE",
            "LLM_COMPLETIONS",
            "LLM_PROMPTS",
            "LLM_RESPONSE_MODEL",
            "LLM_USAGE_COMPLETION_TOKENS",
            "LLM_USAGE_PROMPT_TOKENS",
            "LLM_REQUEST_FUNCTIONS",
            "GEN_AI_INPUT_MESSAGES",
            "GEN_AI_OUTPUT_MESSAGES",
            "GEN_AI_USAGE_INPUT_TOKENS",
            "GEN_AI_USAGE_OUTPUT_TOKENS",
        ]
        for attr in expected:
            assert hasattr(SpanAttributes, attr), f"Missing attribute: {attr}"

    def test_span_attributes_is_str_enum(self):
        """SpanAttributes members should be usable as strings."""
        from litellm.types.integrations.otel import SpanAttributes

        val = SpanAttributes.LLM_REQUEST_MODEL
        assert isinstance(val, str)
        assert val == "gen_ai.request.model"


class TestOtelImportWithoutProxy(unittest.TestCase):
    """Test that the OTel integration module can be imported without proxy deps."""

    def test_opentelemetry_module_importable(self):
        """litellm.integrations.opentelemetry should import without error."""
        mod = importlib.import_module("litellm.integrations.opentelemetry")
        assert hasattr(mod, "OpenTelemetry")
        assert hasattr(mod, "OpenTelemetryConfig")

    def test_otel_no_runtime_proxy_types_import_for_span_attributes(self):
        """
        The OTel integration should import SpanAttributes from
        litellm.types.integrations.otel, NOT from litellm.proxy._types.

        Uses ast module for robust source analysis instead of string heuristics.
        """
        source_path = importlib.util.find_spec(
            "litellm.integrations.opentelemetry"
        ).origin
        with open(source_path, "r") as f:
            tree = ast.parse(f.read())

        # Walk the AST to find imports of SpanAttributes from proxy._types
        # that are NOT inside an `if TYPE_CHECKING:` guard.
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "litellm.proxy._types":
                continue
            if not any(alias.name == "SpanAttributes" for alias in node.names):
                continue
            # Found a proxy._types SpanAttributes import — check if it's
            # guarded by TYPE_CHECKING by inspecting parent If nodes.
            if not self._is_inside_type_checking(tree, node):
                self.fail(
                    f"Line {node.lineno}: runtime import of SpanAttributes "
                    f"from litellm.proxy._types (should use "
                    f"litellm.types.integrations.otel)"
                )

    @staticmethod
    def _is_inside_type_checking(tree: ast.Module, target: ast.AST) -> bool:
        """Check if an AST node is inside an `if TYPE_CHECKING:` block."""
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            # Match `if TYPE_CHECKING:` pattern
            test = node.test
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                # Check if target is anywhere inside this If body
                for child in ast.walk(node):
                    if child is target:
                        return True
        return False

    def test_langtrace_imports_from_shared_location(self):
        """langtrace.py should import SpanAttributes from the shared location."""
        import litellm.integrations.langtrace as langtrace_mod

        from litellm.types.integrations.otel import SpanAttributes as SharedSpanAttributes

        # Runtime check: langtrace's module-level SpanAttributes should be
        # the shared enum, not a proxy-specific copy.
        self.assertIs(
            langtrace_mod.SpanAttributes,
            SharedSpanAttributes,
            "langtrace should import SpanAttributes from "
            "litellm.types.integrations.otel",
        )


class TestOtelInitializationWithoutProxy(unittest.TestCase):
    """Test that OpenTelemetry class can be initialized without proxy server."""

    @patch("litellm.integrations.opentelemetry.OpenTelemetry._init_tracing")
    @patch("litellm.integrations.opentelemetry.OpenTelemetry._init_metrics")
    @patch("litellm.integrations.opentelemetry.OpenTelemetry._init_logs")
    def test_otel_init_without_proxy_server(
        self, mock_logs, mock_metrics, mock_tracing
    ):
        """OpenTelemetry should initialize even when proxy_server can't be imported."""
        from litellm.integrations.opentelemetry import OpenTelemetry

        # Temporarily make proxy_server un-importable
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == "litellm.proxy.proxy_server" or (
                name == "litellm.proxy" and args and "proxy_server" in str(args)
            ):
                raise ImportError("Simulated: proxy_server not available")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            otel = OpenTelemetry()
            assert otel is not None
            assert otel.config is not None

    def test_proxy_init_uses_debug_not_warning(self):
        """_init_otel_logger_on_litellm_proxy should use debug log, not warning."""
        source = importlib.util.find_spec(
            "litellm.integrations.opentelemetry"
        ).origin
        with open(source, "r") as f:
            content = f.read()

        # Find the _init_otel_logger_on_litellm_proxy method
        method_start = content.find("def _init_otel_logger_on_litellm_proxy")
        assert method_start != -1
        # Get the next method (or end of file)
        next_def = content.find("\n    def ", method_start + 1)
        method_body = content[method_start:next_def] if next_def != -1 else content[method_start:]

        # Should NOT use verbose_logger.warning for the ImportError case
        assert "verbose_logger.warning" not in method_body, (
            "_init_otel_logger_on_litellm_proxy should use debug, not warning"
        )
        assert "verbose_logger.debug" in method_body

    def test_set_tools_attributes_uses_shared_span_attributes(self):
        """set_tools_attributes should work with SpanAttributes from shared location."""
        from litellm.types.integrations.otel import SpanAttributes

        # Verify the attributes used in set_tools_attributes exist
        assert hasattr(SpanAttributes, "LLM_REQUEST_FUNCTIONS")
        assert SpanAttributes.LLM_REQUEST_FUNCTIONS.value == "llm.request.functions"

    def test_tool_calls_kv_pair_uses_shared_span_attributes(self):
        """_tool_calls_kv_pair should work with SpanAttributes from shared location."""
        from litellm.types.integrations.otel import SpanAttributes

        assert hasattr(SpanAttributes, "LLM_COMPLETIONS")
        assert SpanAttributes.LLM_COMPLETIONS.value == "gen_ai.completion"


if __name__ == "__main__":
    unittest.main()
