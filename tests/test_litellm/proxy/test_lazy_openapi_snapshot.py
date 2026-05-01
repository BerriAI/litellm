import sys
from types import ModuleType, SimpleNamespace

from litellm.proxy._lazy_openapi_snapshot import _normalize_operation_ids


def test_generate_snapshot_uses_shared_operation_id_reservations(monkeypatch):
    from litellm.proxy import _lazy_openapi_snapshot

    route_a = SimpleNamespace(path="/feature-a/items")
    route_b = SimpleNamespace(path="/feature-b/items")
    fake_app = SimpleNamespace(
        title="LiteLLM test",
        version="0.0.0",
        routes=[route_a, route_b],
    )

    fake_feature_a_module = ModuleType("fake_feature_a")
    fake_feature_b_module = ModuleType("fake_feature_b")
    monkeypatch.setitem(sys.modules, "fake_feature_a", fake_feature_a_module)
    monkeypatch.setitem(sys.modules, "fake_feature_b", fake_feature_b_module)

    fake_lazy_features_module = ModuleType("litellm.proxy._lazy_features")
    fake_lazy_features_module.LAZY_FEATURES = [
        SimpleNamespace(
            name="feature-a",
            module_path="fake_feature_a",
            path_prefixes=("/feature-a",),
            register_fn=lambda app, module: None,
        ),
        SimpleNamespace(
            name="feature-b",
            module_path="fake_feature_b",
            path_prefixes=("/feature-b",),
            register_fn=lambda app, module: None,
        ),
    ]
    monkeypatch.setitem(
        sys.modules, "litellm.proxy._lazy_features", fake_lazy_features_module
    )

    def fake_get_openapi(title, version, routes):
        path = routes[0].path
        return {
            "paths": {path: {"get": {"operationId": "shared_operation_id_get"}}},
            "components": {"schemas": {"Example": {"type": "object"}}},
        }

    def fake_ensure_unique_openapi_operation_ids(schema, reserved_operation_ids):
        for path_item in schema["paths"].values():
            operation = path_item["get"]
            operation_id = operation["operationId"]
            if operation_id in reserved_operation_ids:
                operation_id = f"{operation_id}_2"
                operation["operationId"] = operation_id
            reserved_operation_ids.add(operation_id)
        return schema

    fake_proxy_server_module = ModuleType("litellm.proxy.proxy_server")
    fake_proxy_server_module.app = fake_app
    fake_proxy_server_module.ensure_unique_openapi_operation_ids = (
        fake_ensure_unique_openapi_operation_ids
    )
    monkeypatch.setitem(
        sys.modules, "litellm.proxy.proxy_server", fake_proxy_server_module
    )
    monkeypatch.setattr("fastapi.openapi.utils.get_openapi", fake_get_openapi)

    fragments = _lazy_openapi_snapshot.generate_snapshot()

    assert (
        fragments["feature-a"]["paths"]["/feature-a/items"]["get"]["operationId"]
        == "shared_operation_id_get"
    )
    assert (
        fragments["feature-b"]["paths"]["/feature-b/items"]["get"]["operationId"]
        == "shared_operation_id_get_2"
    )
    assert fragments["feature-a"]["paths"]["/feature-a/items"]["get"]["tags"] == [
        "feature-a"
    ]
    assert fragments["feature-b"]["paths"]["/feature-b/items"]["get"]["tags"] == [
        "feature-b"
    ]


def test_normalize_operation_ids_uses_each_http_method():
    paths = {
        "/proxy/{endpoint}": {
            "delete": {"operationId": "proxy_route_proxy__endpoint__put"},
            "get": {"operationId": "proxy_route_proxy__endpoint__put"},
            "post": {"operationId": "proxy_route_proxy__endpoint__put"},
            "put": {"operationId": "proxy_route_proxy__endpoint__put"},
        }
    }

    _normalize_operation_ids(paths)

    operations = paths["/proxy/{endpoint}"]
    assert operations["delete"]["operationId"] == "proxy_route_proxy__endpoint__delete"
    assert operations["get"]["operationId"] == "proxy_route_proxy__endpoint__get"
    assert operations["post"]["operationId"] == "proxy_route_proxy__endpoint__post"
    assert operations["put"]["operationId"] == "proxy_route_proxy__endpoint__put"


def test_normalize_operation_ids_preserves_custom_ids():
    paths = {
        "/proxy/{endpoint}": {
            "get": {"operationId": "custom_operation"},
            "post": {"operationId": "custom_operation"},
        }
    }

    _normalize_operation_ids(paths)

    operations = paths["/proxy/{endpoint}"]
    assert operations["get"]["operationId"] == "custom_operation"
    assert operations["post"]["operationId"] == "custom_operation"
