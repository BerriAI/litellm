import pytest

from litellm.proxy.utils import (
    _get_docs_url,
    _get_openapi_url,
    _get_redoc_url,
    get_custom_url,
    get_proxy_base_url,
    get_server_root_path,
    join_paths,
    normalize_route_for_root_path,
)


def normalize(value):
    return value


def _clear_url_env(monkeypatch):
    for var in (
        "REDOC_URL",
        "NO_REDOC",
        "DOCS_URL",
        "NO_DOCS",
        "OPENAPI_URL",
        "NO_OPENAPI",
        "PROXY_BASE_URL",
        "SERVER_ROOT_PATH",
    ):
        monkeypatch.delenv(var, raising=False)


def test_get_redoc_url_default(monkeypatch):
    _clear_url_env(monkeypatch)
    summary = {
        "result": _get_redoc_url(),
        "redoc_url_env": None,
        "no_redoc_env": None,
    }
    assert summary == {
        "result": "/redoc",
        "redoc_url_env": None,
        "no_redoc_env": None,
    }


def test_get_redoc_url_custom_env(monkeypatch):
    _clear_url_env(monkeypatch)
    monkeypatch.setenv("REDOC_URL", "/custom-redoc")
    summary = {
        "result": _get_redoc_url(),
        "redoc_url_env": "/custom-redoc",
        "default_overridden": True,
    }
    assert summary == {
        "result": "/custom-redoc",
        "redoc_url_env": "/custom-redoc",
        "default_overridden": True,
    }


def test_get_redoc_url_disabled_returns_none_error_path(monkeypatch):
    _clear_url_env(monkeypatch)
    monkeypatch.setenv("NO_REDOC", "True")
    assert _get_redoc_url() is None


def test_get_docs_url_default(monkeypatch):
    _clear_url_env(monkeypatch)
    summary = {
        "result": _get_docs_url(),
        "no_docs": None,
        "docs_url": None,
    }
    assert summary == {
        "result": "/",
        "no_docs": None,
        "docs_url": None,
    }


def test_get_docs_url_custom_env(monkeypatch):
    _clear_url_env(monkeypatch)
    monkeypatch.setenv("DOCS_URL", "/api-docs")
    summary = {
        "result": _get_docs_url(),
        "env": "/api-docs",
        "default_overridden": True,
    }
    assert summary == {
        "result": "/api-docs",
        "env": "/api-docs",
        "default_overridden": True,
    }


def test_get_docs_url_disabled_returns_none_error_path(monkeypatch):
    _clear_url_env(monkeypatch)
    monkeypatch.setenv("NO_DOCS", "True")
    assert _get_docs_url() is None


def test_get_openapi_url_default(monkeypatch):
    _clear_url_env(monkeypatch)
    summary = {
        "result": _get_openapi_url(),
        "no_openapi": None,
        "openapi_url": None,
    }
    assert summary == {
        "result": "/openapi.json",
        "no_openapi": None,
        "openapi_url": None,
    }


def test_get_openapi_url_custom_env(monkeypatch):
    _clear_url_env(monkeypatch)
    monkeypatch.setenv("OPENAPI_URL", "/api-schema")
    summary = {
        "result": _get_openapi_url(),
        "env": "/api-schema",
        "default_overridden": True,
    }
    assert summary == {
        "result": "/api-schema",
        "env": "/api-schema",
        "default_overridden": True,
    }


def test_get_openapi_url_disabled_returns_none_error_path(monkeypatch):
    _clear_url_env(monkeypatch)
    monkeypatch.setenv("NO_OPENAPI", "True")
    assert _get_openapi_url() is None


@pytest.mark.parametrize(
    "base, route, expected",
    [
        ("https://proxy.example.com", "/v1/chat", "https://proxy.example.com/v1/chat"),
        ("https://proxy.example.com/", "/v1/chat", "https://proxy.example.com/v1/chat"),
        ("https://proxy.example.com", "v1/chat", "https://proxy.example.com/v1/chat"),
        ("https://proxy.example.com", "", "https://proxy.example.com"),
        ("", "/v1/chat", "/v1/chat"),
        ("", "", "/"),
    ],
)
def test_join_paths_happy_path(base, route, expected):
    result = join_paths(base, route)
    assert {
        "input_base": base,
        "input_route": route,
        "result": result,
        "expected": expected,
    } == {
        "input_base": base,
        "input_route": route,
        "result": expected,
        "expected": expected,
    }


def test_join_paths_avoids_duplicating_route_suffix():
    summary = {
        "result": join_paths("https://api.example.com/v1/chat", "/v1/chat"),
        "base": "https://api.example.com/v1/chat",
        "route": "/v1/chat",
    }
    assert summary == {
        "result": "https://api.example.com/v1/chat",
        "base": "https://api.example.com/v1/chat",
        "route": "/v1/chat",
    }


def test_join_paths_invalid_input_raises():
    with pytest.raises(AttributeError):
        join_paths(None, "/v1/chat")


def test_get_proxy_base_url_returns_env_when_set(monkeypatch):
    _clear_url_env(monkeypatch)
    monkeypatch.setenv("PROXY_BASE_URL", "https://litellm.test")
    summary = {
        "result": get_proxy_base_url(),
        "env": "https://litellm.test",
        "is_set": True,
    }
    assert summary == {
        "result": "https://litellm.test",
        "env": "https://litellm.test",
        "is_set": True,
    }


def test_get_proxy_base_url_error_path_returns_none_when_unset(monkeypatch):
    _clear_url_env(monkeypatch)
    assert get_proxy_base_url() is None


def test_get_server_root_path_returns_env(monkeypatch):
    _clear_url_env(monkeypatch)
    monkeypatch.setenv("SERVER_ROOT_PATH", "/proxy")
    summary = {
        "result": get_server_root_path(),
        "env": "/proxy",
        "is_set": True,
    }
    assert summary == {
        "result": "/proxy",
        "env": "/proxy",
        "is_set": True,
    }


def test_get_server_root_path_error_path_default_empty_string(monkeypatch):
    _clear_url_env(monkeypatch)
    assert get_server_root_path() == ""


def test_get_custom_url_with_proxy_base_and_root_and_route(monkeypatch):
    _clear_url_env(monkeypatch)
    monkeypatch.setenv("PROXY_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("SERVER_ROOT_PATH", "/proxy")
    result = get_custom_url("https://request.example.com", "/v1/chat")
    summary = {
        "result": result,
        "base_used": "PROXY_BASE_URL",
        "root_path": "/proxy",
        "route": "/v1/chat",
    }
    assert summary == {
        "result": "https://api.example.com/proxy/v1/chat",
        "base_used": "PROXY_BASE_URL",
        "root_path": "/proxy",
        "route": "/v1/chat",
    }


def test_get_custom_url_falls_back_to_request_base(monkeypatch):
    _clear_url_env(monkeypatch)
    result = get_custom_url("https://request.example.com", "/v1/chat")
    summary = {
        "result": result,
        "base_used": "request_base_url",
        "root_path": "",
        "route": "/v1/chat",
    }
    assert summary == {
        "result": "https://request.example.com/v1/chat",
        "base_used": "request_base_url",
        "root_path": "",
        "route": "/v1/chat",
    }


def test_get_custom_url_no_route_uses_root_path(monkeypatch):
    _clear_url_env(monkeypatch)
    monkeypatch.setenv("SERVER_ROOT_PATH", "/proxy")
    result = get_custom_url("https://request.example.com", route=None)
    summary = {
        "result": result,
        "base_used": "request_base_url",
        "root_path": "/proxy",
        "route": None,
    }
    assert summary == {
        "result": "https://request.example.com/proxy",
        "base_used": "request_base_url",
        "root_path": "/proxy",
        "route": None,
    }


def test_get_custom_url_error_path_invalid_base_raises(monkeypatch):
    _clear_url_env(monkeypatch)
    with pytest.raises(AttributeError):
        get_custom_url(None, "/v1/chat")


def test_normalize_route_for_root_path_strips_prefix(monkeypatch):
    _clear_url_env(monkeypatch)
    monkeypatch.setenv("SERVER_ROOT_PATH", "/proxy")
    summary = {
        "result": normalize_route_for_root_path("/proxy/v1/chat"),
        "root_path": "/proxy",
        "input": "/proxy/v1/chat",
    }
    assert summary == {
        "result": "/v1/chat",
        "root_path": "/proxy",
        "input": "/proxy/v1/chat",
    }


def test_normalize_route_for_root_path_returns_route_when_no_root(monkeypatch):
    _clear_url_env(monkeypatch)
    summary = {
        "result": normalize_route_for_root_path("/v1/chat"),
        "root_path": "",
        "input": "/v1/chat",
    }
    assert summary == {
        "result": "/v1/chat",
        "root_path": "",
        "input": "/v1/chat",
    }


def test_normalize_route_for_root_path_error_path_when_route_not_under_root(
    monkeypatch,
):
    _clear_url_env(monkeypatch)
    monkeypatch.setenv("SERVER_ROOT_PATH", "/proxy")
    assert normalize_route_for_root_path("/other/v1/chat") is None


# ---------------------------------------------------------------------------
# get_custom_url under a per-request prefix (SERVER_ROOT_PATHS):
# when a request lives under a dynamic prefix, request.base_url already
# contains it. Appending the SERVER_ROOT_PATH scalar on top produced
# double-prefixed SSO callback / login URLs (a path that doesn't exist on
# the deployment). These pin that the emitted URL now stays under one
# prefix — the one the request actually arrived on.
# ---------------------------------------------------------------------------


def test_get_custom_url_uses_per_request_prefix_when_middleware_ran(monkeypatch):
    """The middleware stashes the effective per-request prefix in a ContextVar.
    ``get_custom_url`` reads that in preference to the SERVER_ROOT_PATH scalar,
    and ``join_paths``'s tail-dedup collapses the append so a request whose
    ``base_url`` already ends in ``/tenant-a`` does not become
    ``/tenant-a/legacy/route``."""
    _clear_url_env(monkeypatch)
    monkeypatch.setenv("SERVER_ROOT_PATH", "/legacy")

    from litellm.proxy.middleware.per_request_root_path_middleware import (
        _request_root_path_var,
    )

    token = _request_root_path_var.set("/tenant-a")
    try:
        # request.base_url already carries the tenant prefix; the scalar
        # SERVER_ROOT_PATH must not be re-appended on top.
        result = get_custom_url(
            request_base_url="https://request.example.com/tenant-a/",
            route="/v1/chat",
        )
    finally:
        _request_root_path_var.reset(token)

    assert result == "https://request.example.com/tenant-a/v1/chat"


def test_get_custom_url_no_double_prefix_when_both_env_vars_configured(monkeypatch):
    """Regression guard for the review point: with SERVER_ROOT_PATH also set
    (scalar-legacy) and the request matched by SERVER_ROOT_PATHS (per-request),
    the emitted URL is under one prefix — the per-request one — never both."""
    _clear_url_env(monkeypatch)
    monkeypatch.setenv("SERVER_ROOT_PATH", "/legacy")
    monkeypatch.setenv("SERVER_ROOT_PATHS", "/tenant-a")

    from litellm.proxy.middleware.per_request_root_path_middleware import (
        _request_root_path_var,
    )

    token = _request_root_path_var.set("/tenant-a")
    try:
        # No "/legacy" ever appears — the fix pins the reviewer's expected
        # behavior: only one prefix should apply per request.
        result = get_custom_url("https://api.example.com/tenant-a", "/sso/callback")
    finally:
        _request_root_path_var.reset(token)

    assert result == "https://api.example.com/tenant-a/sso/callback"
    assert "/legacy" not in result


def test_get_custom_url_scalar_only_still_stamps_root_path(monkeypatch):
    """Pre-middleware deployments have not opted into SERVER_ROOT_PATHS at all;
    the ContextVar stays unset and the SERVER_ROOT_PATH scalar owns the answer
    — the behavior every existing scalar-only deployment relies on."""
    _clear_url_env(monkeypatch)
    monkeypatch.setenv("SERVER_ROOT_PATH", "/legacy")

    result = get_custom_url("https://request.example.com", "/v1/chat")

    assert result == "https://request.example.com/legacy/v1/chat"
