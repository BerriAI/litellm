"""
Test to ensure no hardcoded secrets exist in the codebase.

This catches Base64 Basic Authentication strings and other secret patterns
that would be flagged by secret scanners like GitGuardian/ggshield.
"""

import base64
import os
import re

import pytest

# Root of the litellm package
LITELLM_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "litellm")

# Regex for Base64 Basic Auth patterns: 'Basic <base64string>'
# Matches strings like: Basic YW55dGhpbmc6YW55dGhpbmc=
BASIC_AUTH_PATTERN = re.compile(
    r"""['"]Basic\s+([A-Za-z0-9+/]{16,}={0,2})['"]"""
)

# Directories/files to skip
SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".mypy_cache", ".ruff_cache"}


def _is_real_base64_credentials(match_str: str) -> bool:
    """Check if a Base64 string decodes to something that looks like credentials (user:pass)."""
    try:
        # Add padding if needed - Base64 strings may omit trailing '='
        padded = match_str + "=" * (-len(match_str) % 4)
        decoded = base64.b64decode(padded).decode("utf-8", errors="ignore")
        return ":" in decoded
    except Exception:
        return False


def _collect_python_files():
    """Collect all Python files under the litellm package."""
    python_files = []
    for root, dirs, files in os.walk(LITELLM_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if f.endswith(".py"):
                python_files.append(os.path.join(root, f))
    return python_files


def test_no_hardcoded_basic_auth_secrets():
    """Ensure no hardcoded Base64 Basic Authentication credentials exist in source code.

    This test prevents regressions like the one caught by T-Mobile's GitGuardian
    container scan, where a docstring contained a literal Base64-encoded
    'Basic YW55dGhpbmc6YW55dGhpbmc' string (anything:anything).
    """
    violations = []

    for filepath in _collect_python_files():
        with open(filepath, "r", errors="ignore") as f:
            for line_num, line in enumerate(f, start=1):
                for match in BASIC_AUTH_PATTERN.finditer(line):
                    b64_value = match.group(1)
                    if _is_real_base64_credentials(b64_value):
                        rel_path = os.path.relpath(filepath, LITELLM_ROOT)
                        violations.append(
                            f"  {rel_path}:{line_num}: {match.group(0)}"
                        )

    assert not violations, (
        "Found hardcoded Base64 Basic Auth credentials that will be flagged by "
        "secret scanners (e.g. GitGuardian/ggshield):\n"
        + "\n".join(violations)
        + "\n\nUse placeholders like '<base64(username:password)>' in comments/docs instead."
    )
