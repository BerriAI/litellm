from litellm.proxy import _lazy_openapi_snapshot as snapshot_module


def test_load_snapshot_returns_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(snapshot_module, "SNAPSHOT_FILE", tmp_path / "missing.json")

    assert snapshot_module.load_snapshot() is None


def test_load_snapshot_reads_json(monkeypatch, tmp_path):
    snapshot_file = tmp_path / "snapshot.json"
    snapshot_file.write_text('{"mcp": {"paths": {}}}')
    monkeypatch.setattr(snapshot_module, "SNAPSHOT_FILE", snapshot_file)

    assert snapshot_module.load_snapshot() == {"mcp": {"paths": {}}}


def test_load_snapshot_returns_none_for_invalid_json(monkeypatch, tmp_path):
    snapshot_file = tmp_path / "snapshot.json"
    snapshot_file.write_text("{")
    monkeypatch.setattr(snapshot_module, "SNAPSHOT_FILE", snapshot_file)

    assert snapshot_module.load_snapshot() is None


def test_normalize_operation_ids_uses_each_http_method():
    paths = {
        "/proxy/{endpoint}": {
            "delete": {"operationId": "proxy_route_proxy__endpoint__put"},
            "get": {"operationId": "proxy_route_proxy__endpoint__put"},
            "post": {"operationId": "proxy_route_proxy__endpoint__put"},
            "put": {"operationId": "proxy_route_proxy__endpoint__put"},
        }
    }

    snapshot_module._normalize_operation_ids(paths)

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

    snapshot_module._normalize_operation_ids(paths)

    operations = paths["/proxy/{endpoint}"]
    assert operations["get"]["operationId"] == "custom_operation"
    assert operations["post"]["operationId"] == "custom_operation"


def test_normalize_operation_ids_skips_invalid_entries():
    paths = {
        "/not-a-dict": "skip",
        "/no-http-methods": {"parameters": []},
        "/invalid-operation": {
            "get": ["skip"],
            "post": {"operationId": 123},
            "parameters": [],
        },
    }

    snapshot_module._normalize_operation_ids(paths)

    assert paths["/not-a-dict"] == "skip"
    assert paths["/no-http-methods"] == {"parameters": []}
    assert paths["/invalid-operation"]["get"] == ["skip"]
    assert paths["/invalid-operation"]["post"] == {"operationId": 123}
