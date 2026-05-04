"""
Per-request mapping from OpenAI-safe tool names back to client-supplied names.

OpenAI's Chat Completions API requires tools[].function.name to match
``^[a-zA-Z0-9_-]+$``. For providers that enforce this, we rewrite outbound tool
names and keep sanitized -> original in a ContextVar dict for this completion only
(never a process-wide cache).

Restore in ``convert_dict_to_response`` only affects names present in the current
request's mapping, so other providers and concurrent requests are unaffected.
"""

from __future__ import annotations

import contextvars
from typing import Dict, Optional

_CTX: contextvars.ContextVar[Optional[Dict[str, str]]] = contextvars.ContextVar(
    "litellm_openai_tool_name_mapping", default=None
)

# Providers where the upstream Chat Completions API applies OpenAI tool-name validation.
_OPENAI_TOOL_NAME_VALIDATION_PROVIDERS = frozenset(
    {
        "openai",
        "azure",
        "azure_ai",
        "custom_openai",
        "text-completion-openai",
        "groq",
        "deepinfra",
        "together_ai",
        "fireworks_ai",
        "nvidia_nim",
        "github_copilot",
        "perplexity",
        "xai",
    }
)


def should_sanitize_openai_tool_names(litellm_provider: str) -> bool:
    return litellm_provider in _OPENAI_TOOL_NAME_VALIDATION_PROVIDERS


def begin_openai_tool_name_mapping_scope() -> None:
    """Reset mapping for this completion (call once at the start of completion())."""
    _CTX.set({})


def _store(sanitized: str, original: str) -> None:
    if sanitized == original:
        return
    m = _CTX.get()
    if m is None:
        m = {}
        _CTX.set(m)
    m[sanitized] = original


def get_openai_tool_name(response_tool_name: str) -> str:
    m = _CTX.get()
    if m is not None and response_tool_name in m:
        return m[response_tool_name]
    return response_tool_name


def restore_openai_tool_name_for_user(sanitized_name: Optional[str]) -> Optional[str]:
    if sanitized_name is None:
        return None
    return get_openai_tool_name(sanitized_name)


def store_openai_tool_name_mapping(sanitized: str, original: str) -> None:
    """Record a rewrite when ``validate_and_fix_openai_tools`` sanitizes a name."""
    _store(sanitized, original)
