"""Adapter registry. Adapter modules are imported only when their harness is used."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Final

from litellm.harness.adapters.base import HarnessAdapter, SessionContext
from litellm.harness.types import Capabilities, Harness, require_harness

ADAPTER_PATHS: Final[Mapping[Harness, tuple[str, str]]] = {
    Harness.CLAUDE_CODE: ("litellm.harness.adapters.claude_code", "ClaudeCodeAdapter"),
    Harness.CODEX: ("litellm.harness.adapters.codex", "CodexAdapter"),
    Harness.OPENCODE: ("litellm.harness.adapters.opencode", "OpenCodeAdapter"),
    Harness.DEEPAGENTS: ("litellm.harness.adapters.deepagents", "DeepAgentsAdapter"),
}


def get_adapter_class(harness: Harness) -> type[HarnessAdapter]:
    """Import and return the adapter class for harness."""
    module_path, class_name = ADAPTER_PATHS[require_harness(harness)]
    module = importlib.import_module(module_path)
    adapter_cls: type[HarnessAdapter] = getattr(module, class_name)
    return adapter_cls


def get_capabilities(harness: Harness) -> Capabilities:
    """What harness supports, as declared by its adapter."""
    return get_adapter_class(harness).capabilities


__all__ = [
    "ADAPTER_PATHS",
    "HarnessAdapter",
    "SessionContext",
    "get_adapter_class",
    "get_capabilities",
]
