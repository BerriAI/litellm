from litellm.proxy._lazy_openapi_snapshot import _normalize_operation_ids


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
