"""
Unit tests for Python 3.10+ typing compatibility in context management compact editor.

Fixes Issue #38076:
- Verifies that NotRequired is correctly imported either from typing (>= 3.11) or typing_extensions (< 3.11).
- Verifies that _SummaryCallKwargs TypedDict definition is valid across Python versions.
"""

import importlib
import sys


def test_compact_editor_not_required_import():
    """Verify that compact.py imports NotRequired correctly based on sys.version_info."""
    import litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact as compact_module

    assert hasattr(compact_module, "NotRequired")
    if sys.version_info >= (3, 11):
        import typing

        assert compact_module.NotRequired is typing.NotRequired
    else:
        import typing_extensions

        assert compact_module.NotRequired is typing_extensions.NotRequired


def test_compact_editor_summary_call_kwargs_definition():
    """Verify that _SummaryCallKwargs TypedDict is properly constructed."""
    from litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact import (
        _SummaryCallKwargs,
    )

    annotations = _SummaryCallKwargs.__annotations__
    assert "model" in annotations
    assert "messages" in annotations
    assert "max_tokens" in annotations
    assert "timeout" in annotations
    assert "litellm_metadata" in annotations
    assert "user" in annotations
    assert "allowed_model_region" in annotations


def test_compact_module_can_be_imported():
    """Ensure litellm compact module imports without error on the current runtime."""
    compact_module = importlib.import_module(
        "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact"
    )
    assert compact_module is not None
