"""
Tests for CORS configuration security fix.

Verifies that allow_credentials is automatically disabled when
allow_origins=["*"] (wildcard) to prevent credentialed cross-origin
requests from arbitrary origins.
"""

import pytest


def _compute_cors_config(cors_origins_env):
    """
    Mirror of the CORS config logic in proxy_server.py.
    Kept here so tests remain isolated from module-level side-effects.
    """
    if cors_origins_env is None or cors_origins_env.strip() == "":
        origins = ["*"]
    else:
        origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
    allow_cors_credentials = "*" not in origins
    return origins, allow_cors_credentials


def test_cors_wildcard_disables_credentials():
    """should disable credentials when LITELLM_CORS_ORIGINS is not set (defaults to wildcard)."""
    origins, allow_credentials = _compute_cors_config(None)
    assert origins == ["*"]
    assert allow_credentials is False


def test_cors_empty_string_disables_credentials():
    """should disable credentials when LITELLM_CORS_ORIGINS is an empty or whitespace string."""
    for empty in ("", "   ", "\t"):
        origins, allow_credentials = _compute_cors_config(empty)
        assert origins == ["*"], f"Expected wildcard for input {repr(empty)}"
        assert (
            allow_credentials is False
        ), f"Expected no credentials for input {repr(empty)}"


def test_cors_single_specific_origin_enables_credentials():
    """should enable credentials when a single explicit origin is configured."""
    origins, allow_credentials = _compute_cors_config("https://admin.example.com")
    assert origins == ["https://admin.example.com"]
    assert allow_credentials is True


def test_cors_multiple_specific_origins_enables_credentials():
    """should enable credentials and correctly parse comma-separated origins."""
    origins, allow_credentials = _compute_cors_config(
        "https://app.example.com, https://admin.example.com, https://api.example.com"
    )
    assert origins == [
        "https://app.example.com",
        "https://admin.example.com",
        "https://api.example.com",
    ]
    assert allow_credentials is True


def test_cors_wildcard_string_in_env_disables_credentials():
    """should disable credentials when LITELLM_CORS_ORIGINS is explicitly set to '*'."""
    origins, allow_credentials = _compute_cors_config("*")
    assert "*" in origins
    assert allow_credentials is False


def test_cors_origins_strips_whitespace():
    """should strip surrounding whitespace from each origin entry."""
    origins, _ = _compute_cors_config("  https://a.com  ,  https://b.com  ")
    assert origins == ["https://a.com", "https://b.com"]


def test_cors_origins_skips_blank_entries():
    """should skip blank entries caused by trailing/double commas."""
    origins, allow_credentials = _compute_cors_config("https://a.com,,https://b.com,")
    assert origins == ["https://a.com", "https://b.com"]
    assert allow_credentials is True
