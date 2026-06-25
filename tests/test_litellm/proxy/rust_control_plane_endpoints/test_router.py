"""Unit tests for the combined Rust control-plane router."""

from litellm.proxy.rust_control_plane_endpoints.router import rust_control_plane_router


def test_router_mounts_rust_consumed_endpoints():
    route_paths = {
        getattr(route, "path", None) for route in rust_control_plane_router.routes
    }

    assert "/v1/rust_control_plane/authentication" in route_paths
    assert "/v1/rust_control_plane/logs" in route_paths
