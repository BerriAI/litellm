"""Behavior pins for proxy_server OpenAPI customization + CORS helpers.

Pins covered:
- ``_generate_stable_operation_id``
- ``_strip_operation_id_method_suffix``
- ``ensure_unique_openapi_operation_ids``
- ``_inject_websocket_stubs_into_openapi_schema``
- ``get_openapi_schema``
- ``custom_openapi``
- ``mount_swagger_ui``
- ``_get_cors_config``
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from litellm.proxy.proxy_server import (
    _generate_stable_operation_id,
    _get_cors_config,
    _inject_websocket_stubs_into_openapi_schema,
    _strip_operation_id_method_suffix,
    custom_openapi,
    ensure_unique_openapi_operation_ids,
    get_openapi_schema,
    mount_swagger_ui,
)

from .conftest import normalize

# ---------------------------------------------------------------------------
# _generate_stable_operation_id
# ---------------------------------------------------------------------------


def test_generate_stable_operation_id_single_method_appends_suffix():
    route = SimpleNamespace(
        name="list_models",
        path_format="/v1/models",
        methods={"GET"},
    )
    observed = {
        "operation_id": _generate_stable_operation_id(route),
        "name": route.name,
        "path": route.path_format,
    }
    assert normalize(observed) == {
        "operation_id": "list_models_v1_models_get",
        "name": "list_models",
        "path": "/v1/models",
    }


def test_generate_stable_operation_id_multi_method_no_suffix():
    route = SimpleNamespace(
        name="multi_op",
        path_format="/v1/things/{id}",
        methods={"GET", "POST"},
    )
    observed = {
        "operation_id": _generate_stable_operation_id(route),
        "method_count": len(route.methods),
        "has_method_suffix": _generate_stable_operation_id(route).endswith(
            ("_get", "_post")
        ),
    }
    assert normalize(observed) == {
        "operation_id": "multi_op_v1_things__id_",
        "method_count": 2,
        "has_method_suffix": False,
    }


def test_generate_stable_operation_id_missing_attrs_raises_error():
    bad_route = SimpleNamespace()  # missing name/path_format/methods
    with pytest.raises(AttributeError):
        _generate_stable_operation_id(bad_route)


# ---------------------------------------------------------------------------
# _strip_operation_id_method_suffix
# ---------------------------------------------------------------------------


def test_strip_operation_id_method_suffix_removes_known_method():
    observed = {
        "with_get": _strip_operation_id_method_suffix("list_models_v1_models_get"),
        "with_post": _strip_operation_id_method_suffix("create_thing_post"),
        "with_delete": _strip_operation_id_method_suffix("drop_thing_delete"),
    }
    assert observed == {
        "with_get": "list_models_v1_models",
        "with_post": "create_thing",
        "with_delete": "drop_thing",
    }


def test_strip_operation_id_method_suffix_invalid_suffix_unchanged():
    # "foo" is not a known HTTP method; "nounderscore" has no separator at all.
    observed = {
        "unknown_suffix": _strip_operation_id_method_suffix("operation_foo"),
        "no_underscore": _strip_operation_id_method_suffix("nounderscore"),
        "empty": _strip_operation_id_method_suffix(""),
    }
    assert observed == {
        "unknown_suffix": "operation_foo",
        "no_underscore": "nounderscore",
        "empty": "",
    }


# ---------------------------------------------------------------------------
# ensure_unique_openapi_operation_ids
# ---------------------------------------------------------------------------


def test_ensure_unique_openapi_operation_ids_rewrites_duplicates():
    schema = {
        "paths": {
            "/a": {"get": {"operationId": "dup_get"}},
            "/b": {"get": {"operationId": "dup_get"}},
            "/c": {"post": {"operationId": "unique_post"}},
        }
    }
    result = ensure_unique_openapi_operation_ids(schema)
    observed = {
        "a_get": result["paths"]["/a"]["get"]["operationId"],
        "b_get": result["paths"]["/b"]["get"]["operationId"],
        "c_post": result["paths"]["/c"]["post"]["operationId"],
        "ids_are_distinct": len(
            {
                result["paths"]["/a"]["get"]["operationId"],
                result["paths"]["/b"]["get"]["operationId"],
                result["paths"]["/c"]["post"]["operationId"],
            }
        )
        == 3,
    }
    assert normalize(observed) == {
        "a_get": "dup_get",
        "b_get": "dup_get_2",
        "c_post": "unique_post",
        "ids_are_distinct": True,
    }


def test_ensure_unique_openapi_operation_ids_respects_reserved():
    # operationId already ends with "_get" (an HTTP method), so the suffix is
    # stripped before re-appending the current method, yielding "reserved_get".
    schema = {
        "paths": {
            "/a": {"get": {"operationId": "reserved_get"}},
        }
    }
    reserved = {"reserved_get"}
    result = ensure_unique_openapi_operation_ids(
        schema, reserved_operation_ids=reserved
    )
    observed = {
        "rewritten": result["paths"]["/a"]["get"]["operationId"],
        "still_includes_original": "reserved_get" in reserved,
        "reserved_grew": len(reserved) > 1,
    }
    assert normalize(observed) == {
        "rewritten": "reserved_get_2",
        "still_includes_original": True,
        "reserved_grew": True,
    }


def test_ensure_unique_openapi_operation_ids_missing_paths_invalid_returns_empty():
    """No ``paths`` key — function must not crash and must return the schema as-is."""
    schema = {"info": {"title": "x"}}
    result = ensure_unique_openapi_operation_ids(schema)
    assert result is schema
    assert "paths" not in result


# ---------------------------------------------------------------------------
# _inject_websocket_stubs_into_openapi_schema
# ---------------------------------------------------------------------------


def test_inject_websocket_stubs_into_openapi_schema_adds_stub():
    schema = {"paths": {}}
    route = SimpleNamespace(path="/ws/chat", name="ws_chat", dependant=None)
    result = _inject_websocket_stubs_into_openapi_schema(schema, [route])
    stub = result["paths"]["/ws/chat"]["get"]
    assert normalize(stub) == {
        "summary": "WebSocket: ws_chat",
        "description": "WebSocket connection endpoint",
        "operationId": "websocket_ws_chat",
        "parameters": [],
        "responses": {"101": {"description": "WebSocket Protocol Switched"}},
        "tags": ["WebSocket"],
    }


def test_inject_websocket_stubs_into_openapi_schema_does_not_overwrite_existing_get():
    # Existing GET on the same path must not be replaced by the stub.
    existing_get = {"summary": "real http get", "operationId": "real_get"}
    schema = {"paths": {"/ws/chat": {"get": existing_get}}}
    route = SimpleNamespace(path="/ws/chat", name="ws_chat", dependant=None)
    result = _inject_websocket_stubs_into_openapi_schema(schema, [route])
    assert result["paths"]["/ws/chat"]["get"] is existing_get


def test_inject_websocket_stubs_into_openapi_schema_missing_paths_key_raises_error():
    schema = {}  # no "paths" key — setdefault on missing schema["paths"] will KeyError
    route = SimpleNamespace(path="/ws/x", name="ws_x", dependant=None)
    with pytest.raises(KeyError):
        _inject_websocket_stubs_into_openapi_schema(schema, [route])


# ---------------------------------------------------------------------------
# get_openapi_schema
# ---------------------------------------------------------------------------


def test_get_openapi_schema_returns_well_formed_schema(monkeypatch):
    """Patch ps.app to a fresh FastAPI so we get a deterministic minimal schema
    without depending on whatever the session app currently has cached."""
    import litellm.proxy.proxy_server as ps

    fresh = FastAPI(title="pinned-title", version="0.0.1")

    @fresh.get("/ping")
    def _ping():
        return {"ok": True}

    monkeypatch.setattr(ps, "app", fresh, raising=True)
    schema = get_openapi_schema()
    observed = {
        "openapi_present": "openapi" in schema,
        "has_paths": isinstance(schema.get("paths"), dict),
        "has_info": isinstance(schema.get("info"), dict),
        "title": schema["info"]["title"],
        "ping_path_in_schema": "/ping" in schema["paths"],
    }
    assert normalize(observed) == {
        "openapi_present": True,
        "has_paths": True,
        "has_info": True,
        "title": "pinned-title",
        "ping_path_in_schema": True,
    }


def test_get_openapi_schema_returns_cached_when_present(monkeypatch):
    """When the patched app already has openapi_schema set, the function
    returns it untouched (no regeneration)."""
    import litellm.proxy.proxy_server as ps

    fresh = FastAPI()
    sentinel = {"openapi": "3.0.0", "paths": {}, "info": {"title": "cached"}}
    fresh.openapi_schema = sentinel
    monkeypatch.setattr(ps, "app", fresh, raising=True)
    result = get_openapi_schema()
    observed = {
        "is_sentinel": result is sentinel,
        "title": result["info"]["title"],
        "paths_empty": result["paths"] == {},
    }
    assert normalize(observed) == {
        "is_sentinel": True,
        "title": "cached",
        "paths_empty": True,
    }


def test_get_openapi_schema_missing_app_attribute_raises_error(monkeypatch):
    """If the module-level ``app`` is replaced by something without
    ``openapi_schema`` and without ``routes``, the function fails fast."""
    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "app", SimpleNamespace(), raising=True)
    with pytest.raises(AttributeError):
        get_openapi_schema()


# ---------------------------------------------------------------------------
# custom_openapi
# ---------------------------------------------------------------------------


def test_custom_openapi_filters_to_openai_routes(monkeypatch):
    """custom_openapi() filters paths down to the OpenAI-compatible set and
    caches the result on the patched app."""
    import litellm.proxy.proxy_server as ps

    fresh = FastAPI(title="pinned-custom", version="0.0.1")

    @fresh.get("/ping")
    def _ping():
        return {"ok": True}

    monkeypatch.setattr(ps, "app", fresh, raising=True)
    schema = custom_openapi()
    observed = {
        "openapi_present": "openapi" in schema,
        "paths_is_dict": isinstance(schema.get("paths"), dict),
        "info_title": schema["info"]["title"],
        "cached_now": fresh.openapi_schema is schema,
        "non_openai_path_filtered": "/ping" not in schema["paths"],
    }
    assert normalize(observed) == {
        "openapi_present": True,
        "paths_is_dict": True,
        "info_title": "pinned-custom",
        "cached_now": True,
        "non_openai_path_filtered": True,
    }


def test_custom_openapi_returns_cached_when_present(monkeypatch):
    import litellm.proxy.proxy_server as ps

    fresh = FastAPI()
    sentinel = {"openapi": "3.0.0", "paths": {}, "info": {"title": "cached"}}
    fresh.openapi_schema = sentinel
    monkeypatch.setattr(ps, "app", fresh, raising=True)
    result = custom_openapi()
    observed = {
        "is_sentinel": result is sentinel,
        "title": result["info"]["title"],
        "paths_empty": result["paths"] == {},
    }
    assert normalize(observed) == {
        "is_sentinel": True,
        "title": "cached",
        "paths_empty": True,
    }


def test_custom_openapi_missing_app_attribute_raises_error(monkeypatch):
    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "app", SimpleNamespace(), raising=True)
    with pytest.raises(AttributeError):
        custom_openapi()


# ---------------------------------------------------------------------------
# mount_swagger_ui
# ---------------------------------------------------------------------------


def test_mount_swagger_ui_mounts_static_route(monkeypatch):
    """mount_swagger_ui mutates the global app — patch the module's `app` to a
    fresh FastAPI() so we don't pollute the session app's mount table."""
    import litellm.proxy.proxy_server as ps
    from fastapi import applications as fa_applications

    fresh_app = FastAPI()
    monkeypatch.setattr(ps, "app", fresh_app, raising=True)
    original_get_swagger = fa_applications.get_swagger_ui_html

    try:
        mount_swagger_ui()
    finally:
        # Restore the swagger monkey-patch so other tests are unaffected.
        fa_applications.get_swagger_ui_html = original_get_swagger

    mount_names = [getattr(r, "name", None) for r in fresh_app.routes]
    observed = {
        "swagger_mounted": "swagger" in mount_names,
        "patched_get_swagger": (
            fa_applications.get_swagger_ui_html is original_get_swagger
        ),
        "route_count_positive": len(fresh_app.routes) > 0,
    }
    assert normalize(observed) == {
        "swagger_mounted": True,
        "patched_get_swagger": True,
        "route_count_positive": True,
    }


def test_mount_swagger_ui_missing_directory_raises_error(monkeypatch, tmp_path):
    """If the swagger directory is missing, StaticFiles raises RuntimeError."""
    import litellm.proxy.proxy_server as ps
    from fastapi import applications as fa_applications

    fresh_app = FastAPI()
    monkeypatch.setattr(ps, "app", fresh_app, raising=True)
    monkeypatch.setattr(
        ps, "current_dir", str(tmp_path / "does_not_exist"), raising=True
    )
    original_get_swagger = fa_applications.get_swagger_ui_html

    try:
        with pytest.raises(RuntimeError):
            mount_swagger_ui()
    finally:
        fa_applications.get_swagger_ui_html = original_get_swagger


# ---------------------------------------------------------------------------
# _get_cors_config
# ---------------------------------------------------------------------------


def test_get_cors_config_explicit_origins_and_credentials():
    origins, allow_creds = _get_cors_config(
        cors_origins_env="https://a.example,https://b.example",
        cors_credentials_env="true",
    )
    observed = {
        "origins": origins,
        "allow_credentials": allow_creds,
        "origin_count": len(origins),
    }
    assert normalize(observed) == {
        "origins": ["https://a.example", "https://b.example"],
        "allow_credentials": True,
        "origin_count": 2,
    }


def test_get_cors_config_wildcard_defaults_credentials_false(monkeypatch):
    # Clear env to ensure we test the default branch deterministically.
    monkeypatch.delenv("LITELLM_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("LITELLM_CORS_ALLOW_CREDENTIALS", raising=False)
    origins, allow_creds = _get_cors_config()
    observed = {
        "origins": origins,
        "allow_credentials": allow_creds,
        "wildcard_in_origins": "*" in origins,
    }
    assert normalize(observed) == {
        "origins": ["*"],
        "allow_credentials": False,
        "wildcard_in_origins": True,
    }


def test_get_cors_config_invalid_credentials_value_treated_as_false():
    """Anything other than the literal "true" (case-insensitive) is false —
    misconfigured strings should not silently enable credentialed CORS."""
    _, allow_creds = _get_cors_config(
        cors_origins_env="https://a.example",
        cors_credentials_env="yes-please",
    )
    assert allow_creds is False
