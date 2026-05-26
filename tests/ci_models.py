"""
Centralized CI model identifiers, resolved at import time from environment
variables with sensible defaults.

Many tests in this repository pass a literal model identifier directly to
``litellm.completion`` / ``litellm.embedding`` (e.g. ``model="gpt-4"``).
When a provider deprecates a model, every hardcoded reference must be
updated across the suite -- a noisy, error-prone code change for what is
operationally a config swap.

This module centralizes the common CI model names so the swap becomes a
single environment-variable override at CI invocation time::

    # Default: tests use gpt-4o.
    pytest tests/llm_translation/

    # Swap every test that imports GPT_4O without touching code:
    CI_MODEL_GPT_4O="gpt-4o-2024-08-06" pytest tests/llm_translation/

Usage::

    from tests.ci_models import GPT_4O, CLAUDE_SONNET_4_5, get_ci_model

    response = litellm.completion(
        model=GPT_4O,
        messages=[{"role": "user", "content": "hi"}],
    )

    # For model names without a dedicated constant:
    model = get_ci_model("CI_MODEL_MY_CUSTOM", default="anthropic/claude-haiku")

Each constant ``X`` is resolved exactly once at module import as
``os.environ.get("CI_MODEL_X", "<default>")``. Tests that need to flip the
value mid-run should call :func:`get_ci_model` instead, which reads
``os.environ`` on every call.
"""

from __future__ import annotations

import os
import sys
from typing import Set, Tuple

_CI_MODEL_PREFIX = "CI_MODEL_"

# (constant_name, env_var, default_model_id)
#
# Defaults reflect what these tests use today; keep them in sync with the
# most-frequently hardcoded identifiers in the suite.
_DEFAULTS: Tuple[Tuple[str, str, str], ...] = (
    # ---- OpenAI chat ----
    ("GPT_3_5_TURBO", "CI_MODEL_GPT_3_5_TURBO", "gpt-3.5-turbo"),
    ("GPT_3_5_TURBO_INSTRUCT", "CI_MODEL_GPT_3_5_TURBO_INSTRUCT", "gpt-3.5-turbo-instruct"),
    ("GPT_4", "CI_MODEL_GPT_4", "gpt-4"),
    ("GPT_4O", "CI_MODEL_GPT_4O", "gpt-4o"),
    ("GPT_4O_MINI", "CI_MODEL_GPT_4O_MINI", "gpt-4o-mini"),
    ("GPT_4_1_MINI", "CI_MODEL_GPT_4_1_MINI", "gpt-4.1-mini"),
    ("GPT_5", "CI_MODEL_GPT_5", "gpt-5"),
    ("GPT_5_MINI", "CI_MODEL_GPT_5_MINI", "gpt-5-mini"),
    ("GPT_5_4", "CI_MODEL_GPT_5_4", "gpt-5.4"),
    ("GPT_5_5", "CI_MODEL_GPT_5_5", "gpt-5.5"),
    # ---- OpenAI embeddings ----
    ("TEXT_EMBEDDING_ADA_002", "CI_MODEL_TEXT_EMBEDDING_ADA_002", "text-embedding-ada-002"),
    ("TEXT_EMBEDDING_3_SMALL", "CI_MODEL_TEXT_EMBEDDING_3_SMALL", "text-embedding-3-small"),
    # ---- Azure (prefixed routes used in CI) ----
    ("AZURE_GPT_4_1_MINI", "CI_MODEL_AZURE_GPT_4_1_MINI", "azure/gpt-4.1-mini"),
    # ---- Anthropic ----
    ("CLAUDE_SONNET_4_5", "CI_MODEL_CLAUDE_SONNET_4_5", "claude-sonnet-4-5-20250929"),
    ("CLAUDE_SONNET_4", "CI_MODEL_CLAUDE_SONNET_4", "claude-sonnet-4-20250514"),
    ("CLAUDE_3_5_SONNET", "CI_MODEL_CLAUDE_3_5_SONNET", "claude-3-5-sonnet-20240620"),
    ("ANTHROPIC_CLAUDE_SONNET_4_5", "CI_MODEL_ANTHROPIC_CLAUDE_SONNET_4_5", "anthropic/claude-sonnet-4-5-20250929"),
    # ---- Bedrock (Anthropic via Bedrock) ----
    ("BEDROCK_CLAUDE_HAIKU_4_5", "CI_MODEL_BEDROCK_CLAUDE_HAIKU_4_5", "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"),
    ("ANTHROPIC_CLAUDE_HAIKU_4_5", "CI_MODEL_ANTHROPIC_CLAUDE_HAIKU_4_5", "anthropic.claude-haiku-4-5-20251001-v1:0"),
    # ---- Google / Gemini ----
    ("GEMINI_1_5_PRO", "CI_MODEL_GEMINI_1_5_PRO", "gemini-1.5-pro"),
    ("GEMINI_2_5_FLASH", "CI_MODEL_GEMINI_2_5_FLASH", "gemini-2.5-flash"),
    ("GEMINI_GEMINI_2_5_FLASH", "CI_MODEL_GEMINI_GEMINI_2_5_FLASH", "gemini/gemini-2.5-flash"),
)


def get_ci_model(env_var: str, default: str) -> str:
    """Resolve a CI model identifier from an env var, falling back to ``default``.

    Reads ``os.environ`` on every call (unlike the module-level constants,
    which are resolved once at import).

    Parameters
    ----------
    env_var:
        Environment variable name to consult. Must start with the
        ``CI_MODEL_`` prefix so the override is discoverable via
        ``env | grep ^CI_MODEL_``.
    default:
        Model identifier to return if ``env_var`` is unset or empty.

    Returns
    -------
    str
        The resolved model identifier.

    Raises
    ------
    ValueError
        If ``env_var`` does not start with ``CI_MODEL_``.
    """
    if not env_var.startswith(_CI_MODEL_PREFIX):
        raise ValueError(
            f"CI model env var must start with {_CI_MODEL_PREFIX!r}; "
            f"got {env_var!r}"
        )
    value = os.environ.get(env_var)
    if not value:
        return default
    return value


def _populate_module_constants() -> None:
    """Resolve every entry in ``_DEFAULTS`` and bind it on this module."""
    module = sys.modules[__name__]
    seen_constants: Set[str] = set()
    seen_env_vars: Set[str] = set()
    for constant_name, env_var, default in _DEFAULTS:
        if constant_name in seen_constants:
            raise RuntimeError(
                f"Duplicate constant name in _DEFAULTS: {constant_name!r}"
            )
        if env_var in seen_env_vars:
            raise RuntimeError(
                f"Duplicate env var in _DEFAULTS: {env_var!r}"
            )
        if not env_var.startswith(_CI_MODEL_PREFIX):
            raise RuntimeError(
                f"Env var for {constant_name!r} must start with "
                f"{_CI_MODEL_PREFIX!r}; got {env_var!r}"
            )
        seen_constants.add(constant_name)
        seen_env_vars.add(env_var)
        setattr(module, constant_name, get_ci_model(env_var, default))


_populate_module_constants()


def _format_table() -> str:
    """Return a fixed-width table of (constant, env_var, default, current)."""
    rows = [("Constant", "Environment variable", "Default", "Current")]
    rows.extend(
        (cn, ev, d, os.environ.get(ev, d) or d)
        for cn, ev, d in _DEFAULTS
    )
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    lines = []
    for i, r in enumerate(rows):
        lines.append("  ".join(c.ljust(widths[k]) for k, c in enumerate(r)))
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


if __name__ == "__main__":
    print(_format_table())
