from types import SimpleNamespace

from litellm.proxy._lazy_openapi_snapshot import _stable_generate_unique_id


def test_stable_generate_unique_id_sorts_route_methods():
    route = SimpleNamespace(
        name="langfuse_proxy_route",
        path_format="/langfuse/{endpoint}",
        methods={"POST", "GET", "DELETE", "PATCH", "PUT"},
    )

    assert (
        _stable_generate_unique_id(route)
        == "langfuse_proxy_route_langfuse__endpoint__delete"
    )
