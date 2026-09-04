"""
Constants for LiteLLM Skills

Centralized constants for skills processing, code execution, and sandbox configuration.
"""

from typing import Final

LITELLM_SKILL_ID_PREFIX: Final[str] = "litellm_skill_"
"""Prefix for DB-backed skill IDs. The model-facing tool name is the skill ID
with hyphens/spaces replaced by underscores, which leaves this prefix intact."""

# Code execution loop settings
DEFAULT_MAX_ITERATIONS: Final[int] = 10
"""Maximum number of iterations for the automatic code execution loop."""

DEFAULT_SANDBOX_TIMEOUT: Final[int] = 120
"""Default timeout in seconds for sandbox code execution."""
