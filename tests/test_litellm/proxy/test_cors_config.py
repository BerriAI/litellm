"""
Tests for CORS configuration security fix.

All tests import _get_cors_config directly from proxy_server so they exercise
real production code rather than a local mirror.
"""

import pytest


def test_cors_wildcard_disables_credentials():
    """should disable credentials when LITELLM_CORS_ORIGINS is not set (defaults to wildcard)."""
    from litellm.proxy.proxy_server import _get_cors_config

    origins, allow_credentials = _get_cors_config(cors_origins_env="")
    assert origins == ["*"]
    assert allow_credentials is False


def test_cors_empty_string_disables_credentials():
    """should disable credentials when LITELLM_CORS_ORIGINS is empty or whitespace."""
    from litellm.proxy.proxy_server import _get_cors_config

    for empty in ("", "   ", "\t"):
        origins, allow_credentials = _get_cors_config(cors_origins_env=empty)
        assert origins == ["*"], f"Expected wildcard for input {repr(empty)}"
        assert (
            allow_credentials is False
        ), f"Expected no credentials for input {repr(empty)}"


def test_cors_single_specific_origin_enables_credentials():
    """should enable credentials when a single explicit origin is configured."""
    from litellm.proxy.proxy_server import _get_cors_config

    origins, allow_credentials = _get_cors_config(
        cors_origins_env="https://admin.example.com"
    )
    assert origins == ["https://admin.example.com"]
    assert allow_credentials is True


def test_cors_multiple_specific_origins_enables_credentials():
    """should enable credentials and correctly parse comma-separated origins."""
    from litellm.proxy.proxy_server import _get_cors_config

    origins, allow_credentials = _get_cors_config(
        cors_origins_env="https://app.example.com, https://admin.example.com, https://api.example.com"
    )
    assert origins == [
        "https://app.example.com",
        "https://admin.example.com",
        "https://api.example.com",
    ]
    assert allow_credentials is True


def test_cors_wildcard_string_in_env_disables_credentials():
    """should disable credentials when LITELLM_CORS_ORIGINS is explicitly set to '*'."""
    from litellm.proxy.proxy_server import _get_cors_config

    origins, allow_credentials = _get_cors_config(cors_origins_env="*")
    assert "*" in origins
    assert allow_credentials is False


def test_cors_origins_strips_whitespace():
    """should strip surrounding whitespace from each origin entry."""
    from litellm.proxy.proxy_server import _get_cors_config

    origins, _ = _get_cors_config(
        cors_origins_env="  https://a.com  ,  https://b.com  "
    )
    assert origins == ["https://a.com", "https://b.com"]


def test_cors_origins_skips_blank_entries():
    """should skip blank entries caused by trailing/double commas."""
    from litellm.proxy.proxy_server import _get_cors_config

    origins, allow_credentials = _get_cors_config(
        cors_origins_env="https://a.com,,https://b.com,"
    )
    assert origins == ["https://a.com", "https://b.com"]
    assert allow_credentials is True


def test_cors_explicit_credentials_true_overrides_wildcard():
    """should enable credentials when LITELLM_CORS_ALLOW_CREDENTIALS=true even
    if wildcard origins are in use (opt-in for existing deployments)."""
    from litellm.proxy.proxy_server import _get_cors_config

    origins, allow_credentials = _get_cors_config(
        cors_origins_env="",
        cors_credentials_env="true",
    )
    assert "*" in origins
    assert allow_credentials is True


def test_cors_explicit_credentials_false_overrides_specific_origins():
    """should disable credentials when LITELLM_CORS_ALLOW_CREDENTIALS=false even
    if specific origins are configured."""
    from litellm.proxy.proxy_server import _get_cors_config

    origins, allow_credentials = _get_cors_config(
        cors_origins_env="https://admin.example.com",
        cors_credentials_env="false",
    )
    assert origins == ["https://admin.example.com"]
    assert allow_credentials is False


def test_cors_explicit_credentials_case_insensitive():
    """should accept TRUE/FALSE case-insensitively for LITELLM_CORS_ALLOW_CREDENTIALS."""
    from litellm.proxy.proxy_server import _get_cors_config

    _, allow_true = _get_cors_config(cors_origins_env="", cors_credentials_env="TRUE")
    _, allow_false = _get_cors_config(
        cors_origins_env="https://x.com", cors_credentials_env="FALSE"
    )
    assert allow_true is True
    assert allow_false is False


def test_cors_expose_headers_defaults_to_ui_allow_headers():
    """should return exactly LITELLM_UI_ALLOW_HEADERS when the env var is unset/empty."""
    from litellm.constants import LITELLM_UI_ALLOW_HEADERS
    from litellm.proxy.proxy_server import _get_cors_expose_headers

    for empty in ("", "   ", "\t"):
        headers = _get_cors_expose_headers(expose_headers_env=empty)
        assert headers == list(LITELLM_UI_ALLOW_HEADERS), f"Unexpected headers for {repr(empty)}"


def test_cors_expose_headers_appends_extra_headers():
    """should append configured headers to the defaults, parsed like LITELLM_CORS_ORIGINS."""
    from litellm.constants import LITELLM_UI_ALLOW_HEADERS
    from litellm.proxy.proxy_server import _get_cors_expose_headers

    headers = _get_cors_expose_headers(
        expose_headers_env="  x-litellm-response-cost , x-litellm-model-api-base ,,"
    )
    assert headers == [
        *LITELLM_UI_ALLOW_HEADERS,
        "x-litellm-response-cost",
        "x-litellm-model-api-base",
    ]


def test_cors_expose_headers_dedupes_preserving_order():
    """should not duplicate a header already present in the defaults."""
    from litellm.constants import LITELLM_UI_ALLOW_HEADERS
    from litellm.proxy.proxy_server import _get_cors_expose_headers

    dup = LITELLM_UI_ALLOW_HEADERS[0]
    headers = _get_cors_expose_headers(expose_headers_env=f"{dup}, x-litellm-response-cost")
    assert headers == [*LITELLM_UI_ALLOW_HEADERS, "x-litellm-response-cost"]
    assert headers.count(dup) == 1


def test_proxy_server_cors_invariant():
    """should verify that proxy_server module-level origins and allow_cors_credentials
    are consistent — catches any future drift in the module-level call to _get_cors_config.
    """
    import os

    import litellm.proxy.proxy_server as proxy_server

    if os.getenv("LITELLM_CORS_ALLOW_CREDENTIALS") is None:
        assert proxy_server.allow_cors_credentials == (
            "*" not in proxy_server.origins
        ), (
            f"Invariant broken: allow_cors_credentials={proxy_server.allow_cors_credentials} "
            f"but origins={proxy_server.origins}. "
            "When origins contains '*', allow_credentials must be False."
        )
