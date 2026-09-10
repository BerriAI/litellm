import json
import sys
from types import ModuleType, SimpleNamespace

from litellm.proxy._lazy_features import LazyFeature
from litellm.proxy._lazy_openapi_snapshot import SnapshotResult, _normalize_operation_ids, main


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
        LazyFeature(
            name="feature-a",
            module_path="fake_feature_a",
            path_prefixes=("/feature-a",),
            register_fn=lambda app, module: None,
        ),
        LazyFeature(
            name="feature-b",
            module_path="fake_feature_b",
            path_prefixes=("/feature-b",),
            register_fn=lambda app, module: None,
        ),
    ]
    monkeypatch.setitem(sys.modules, "litellm.proxy._lazy_features", fake_lazy_features_module)

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
    fake_proxy_server_module.ensure_unique_openapi_operation_ids = fake_ensure_unique_openapi_operation_ids
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", fake_proxy_server_module)
    monkeypatch.setattr("fastapi.openapi.utils.get_openapi", fake_get_openapi)

    fragments = _lazy_openapi_snapshot.generate_snapshot().fragments

    assert fragments["feature-a"]["paths"]["/feature-a/items"]["get"]["operationId"] == "shared_operation_id_get"
    assert fragments["feature-b"]["paths"]["/feature-b/items"]["get"]["operationId"] == "shared_operation_id_get_2"
    assert fragments["feature-a"]["paths"]["/feature-a/items"]["get"]["tags"] == ["feature-a"]
    assert fragments["feature-b"]["paths"]["/feature-b/items"]["get"]["tags"] == ["feature-b"]


def test_generate_snapshot_registers_transitively_imported_modules(monkeypatch):
    """A feature module already in sys.modules (pulled in transitively by an
    earlier feature) must still get register_fn called, else its routes never
    mount and its fragment silently vanishes from the snapshot. Fragment
    collection must also honor path_suffixes, not just prefixes."""
    from litellm.proxy import _lazy_openapi_snapshot

    fake_app = SimpleNamespace(title="LiteLLM test", version="0.0.0", routes=[])

    fake_module = ModuleType("fake_transitive_feature")
    monkeypatch.setitem(sys.modules, "fake_transitive_feature", fake_module)

    def register_fn(app, module):
        app.routes.append(SimpleNamespace(path="/transitive/items"))
        app.routes.append(SimpleNamespace(path="/v1/{param}/deep/leaf"))

    fake_lazy_features_module = ModuleType("litellm.proxy._lazy_features")
    fake_lazy_features_module.LAZY_FEATURES = [
        LazyFeature(
            name="transitive",
            module_path="fake_transitive_feature",
            path_prefixes=("/transitive",),
            path_suffixes=("/deep/leaf",),
            register_fn=register_fn,
        )
    ]
    monkeypatch.setitem(sys.modules, "litellm.proxy._lazy_features", fake_lazy_features_module)

    def fake_get_openapi(title, version, routes):
        return {"paths": {route.path: {"get": {"operationId": f"op{i}_get"}} for i, route in enumerate(routes)}}

    fake_proxy_server_module = ModuleType("litellm.proxy.proxy_server")
    fake_proxy_server_module.app = fake_app
    fake_proxy_server_module.ensure_unique_openapi_operation_ids = lambda schema, reserved_operation_ids: schema
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", fake_proxy_server_module)
    monkeypatch.setattr("fastapi.openapi.utils.get_openapi", fake_get_openapi)

    fragments = _lazy_openapi_snapshot.generate_snapshot().fragments

    assert fragments["transitive"]["paths"]["/transitive/items"]["get"]["tags"] == ["transitive"]
    assert "/v1/{param}/deep/leaf" in fragments["transitive"]["paths"]


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


def test_generate_snapshot_reports_features_whose_import_fails(monkeypatch):
    from litellm.proxy import _lazy_openapi_snapshot

    fake_app = SimpleNamespace(title="LiteLLM test", version="0.0.0", routes=[])

    fake_module = ModuleType("fake_importable_feature")
    monkeypatch.setitem(sys.modules, "fake_importable_feature", fake_module)

    def register_fn(app, module):
        app.routes.append(SimpleNamespace(path="/importable/items"))

    fake_lazy_features_module = ModuleType("litellm.proxy._lazy_features")
    fake_lazy_features_module.LAZY_FEATURES = [
        LazyFeature(
            name="importable",
            module_path="fake_importable_feature",
            path_prefixes=("/importable",),
            register_fn=register_fn,
        ),
        LazyFeature(
            name="broken",
            module_path="litellm.proxy.this_module_does_not_exist",
            path_prefixes=("/broken",),
        ),
    ]
    monkeypatch.setitem(sys.modules, "litellm.proxy._lazy_features", fake_lazy_features_module)

    def fake_get_openapi(title, version, routes):
        return {"paths": {route.path: {"get": {"operationId": "importable_get"}} for route in routes}}

    fake_proxy_server_module = ModuleType("litellm.proxy.proxy_server")
    fake_proxy_server_module.app = fake_app
    fake_proxy_server_module.ensure_unique_openapi_operation_ids = lambda schema, reserved_operation_ids: schema
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", fake_proxy_server_module)
    monkeypatch.setattr("fastapi.openapi.utils.get_openapi", fake_get_openapi)

    result = _lazy_openapi_snapshot.generate_snapshot()

    assert result.skipped == ("broken",)
    assert sorted(result.fragments) == ["importable"]


def test_main_refuses_to_write_a_snapshot_missing_skipped_features(tmp_path, capsys):
    snapshot_file = tmp_path / "snapshot.json"
    result = SnapshotResult(fragments={"importable": {"paths": {}, "components": {"schemas": {}}}}, skipped=("broken",))

    assert main(snapshot_file, generate=lambda: result) == 1
    assert not snapshot_file.exists()
    assert "broken" in capsys.readouterr().err


def test_main_writes_sorted_snapshot_when_every_feature_loads(tmp_path):
    snapshot_file = tmp_path / "snapshot.json"
    fragments = {
        "zeta": {"paths": {"/z": {}}, "components": {"schemas": {}}},
        "alpha": {"paths": {}, "components": {"schemas": {}}},
    }

    assert main(snapshot_file, generate=lambda: SnapshotResult(fragments=fragments, skipped=())) == 0
    assert json.loads(snapshot_file.read_text()) == fragments
    assert snapshot_file.read_text() == json.dumps(fragments, indent=2, sort_keys=True) + "\n"
