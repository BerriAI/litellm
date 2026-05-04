"""
Per-request mapping between client-supplied tool names and their OpenAI-safe forms.

OpenAI's Chat Completions API requires tools[].function.name to match
``^[a-zA-Z0-9_-]+$``. For providers that enforce this, we rewrite outbound tool
names and keep two ContextVar dicts for this completion only (never a process-wide
cache):

  _CTX      – sanitized  → original   (used on the response path)
  _CTX_REV  – original   → sanitized  (used to rewrite tool_choice before the request)

Using the pre-built reverse map for tool_choice guarantees that collision suffixes
(e.g. "a_b_1") added by _make_unique_openai_tool_name are respected instead of
being recomputed independently.
"""

from __future__ import annotations

import contextvars
from typing import Dict, Optional

_CTX: contextvars.ContextVar[Optional[Dict[str, str]]] = contextvars.ContextVar(
    "litellm_openai_tool_name_mapping", default=None
)
_CTX_REV: contextvars.ContextVar[Optional[Dict[str, str]]] = contextvars.ContextVar(
    "litellm_openai_tool_name_mapping_rev", default=None
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


def begin_openai_tool_name_mapping_scope(*, force_reset: bool = False) -> None:
    """Initialise (or reset) both mapping dicts for one completion request.

    Idempotent by default: if the ContextVar already holds a dict this is a
    no-op.  This matters for async streaming where ``acompletion`` calls this
    function in the *outer* async context before ``contextvars.copy_context()``
    so that both the executor and the stream consumer share the same dict
    objects by reference.  The inner ``completion()`` call then sees a non-None
    dict and skips the reset, preserving the shared reference.

    Pass ``force_reset=True`` only when you explicitly want a clean slate
    (e.g. in tests).
    """
    if force_reset or _CTX.get() is None:
        _CTX.set({})
    if force_reset or _CTX_REV.get() is None:
        _CTX_REV.set({})


def _store(sanitized: str, original: str) -> None:
    if sanitized == original:
        return
    m = _CTX.get()
    if m is None:
        m = {}
        _CTX.set(m)
    m[sanitized] = original
    r = _CTX_REV.get()
    if r is None:
        r = {}
        _CTX_REV.set(r)
    r[original] = sanitized


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


def get_sanitized_tool_name(original_name: str) -> str:
    """Return the sanitized (outbound) name for *original_name* if it was rewritten.

    Used to rewrite ``tool_choice.function.name`` after ``validate_and_fix_openai_tools``
    has run, so that collision suffixes added by ``_make_unique_openai_tool_name`` are
    respected rather than recomputed independently.
    """
    r = _CTX_REV.get()
    if r is not None and original_name in r:
        return r[original_name]
    return original_name
