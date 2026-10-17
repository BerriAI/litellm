import pytest

from litellm.proxy.control_plane.proxy.allowlist import Allowed, Denied, decide


@pytest.mark.parametrize(
    "path",
    [
        "/key/generate",
        "key/list",
        "/team/new",
        "/user/info",
        "/model/info",
        "/management/v1/keys",
        "/audit",
        "/spend/logs",
    ],
)
def test_management_routes_are_allowed_with_a_normalised_leading_slash(path) -> None:
    assert decide(path) == Allowed("/" + path.lstrip("/"))


@pytest.mark.parametrize(
    ("path", "fragment"),
    [
        ("/v1/chat/completions", "inference"),
        ("/chat/completions", "inference"),
        ("/v1/embeddings", "inference"),
        ("/v1/messages", "inference"),
        ("/mcp/tools/call", "inference"),
        ("/login", "auth bootstrap"),
        ("/sso/key/generate", "auth bootstrap"),
        ("/key/generate/service_account", "auth bootstrap"),
        ("/config/yaml", "auth bootstrap"),
        ("/key/../config/yaml", "traversal"),
        ("/key//info", "traversal"),
        ("/health/drain", "lifecycle"),
        ("/health/drain/", "lifecycle"),
    ],
)
def test_inference_auth_and_traversal_are_denied(path, fragment) -> None:
    decision = decide(path)
    assert isinstance(decision, Denied), f"{path} was allowed"
    assert fragment in decision.reason


@pytest.mark.parametrize(
    "path",
    [
        "/key%2F..%2Flogin",
        "/key/%2e%2e/login",
        "/key/%252e%252e/login",
        "/key/./info",
        "/key//info",
        "/key/list/.",
        "/key/list/..",
        "/key/\\../login",
        "/credentialsx",
        "/auditor/logs",
        "/healthz",
        "/settings_export",
    ],
)
def test_encoded_dotted_or_boundary_breaking_paths_are_denied(path: str) -> None:
    assert isinstance(decide(path), Denied), path


@pytest.mark.parametrize(
    "path", ["/credentials", "/credentials/by_name/x", "/audit", "/audit/1", "/health", "/settings"]
)
def test_bare_management_routes_match_only_on_a_route_boundary(path: str) -> None:
    assert decide(path) == Allowed(path)


def test_decided_path_is_the_decoded_path_the_dataplane_will_receive() -> None:
    assert decide("key/info%3Fx") == Denied("/key/info%3Fx", "encoded or reserved path characters are not allowed")
    assert decide("/team/my%20team/info") == Allowed("/team/my team/info")
