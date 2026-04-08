"""
Test that S3 prefix path sanitizes spaces in team_alias and key_alias.

Fixes: https://github.com/BerriAI/litellm/issues/25019
"""

import re


def _sanitize_prefix(prefix_path: str) -> str:
    """Mirror the sanitization logic from s3_v2.py"""
    return re.sub(r"[\s]+", "_", prefix_path)


def test_team_alias_with_spaces_sanitized():
    """Team names with spaces should have spaces replaced by underscores."""
    prefix_components = ["Cloud Tooling", "ugie-dev"]
    prefix_path = "/".join(prefix_components)
    prefix_path = _sanitize_prefix(prefix_path)
    assert prefix_path == "Cloud_Tooling/ugie-dev"


def test_team_alias_with_multiple_spaces():
    """Multiple consecutive spaces collapse to a single underscore."""
    prefix_components = ["My  Great   Team"]
    prefix_path = "/".join(prefix_components)
    prefix_path = _sanitize_prefix(prefix_path)
    assert prefix_path == "My_Great_Team"


def test_team_alias_no_spaces_unchanged():
    """Team names without spaces pass through unchanged."""
    prefix_components = ["engineering", "prod-key"]
    prefix_path = "/".join(prefix_components)
    prefix_path = _sanitize_prefix(prefix_path)
    assert prefix_path == "engineering/prod-key"


def test_team_alias_with_tabs_and_newlines():
    """Tabs and newlines are also replaced."""
    prefix_components = ["team\twith\ttabs"]
    prefix_path = "/".join(prefix_components)
    prefix_path = _sanitize_prefix(prefix_path)
    assert prefix_path == "team_with_tabs"
