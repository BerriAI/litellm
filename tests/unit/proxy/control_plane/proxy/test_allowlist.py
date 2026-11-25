from typing import Final

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
        "/v1/model/info",
        "/model_group/info",
        "/management/v1/keys",
        "/audit",
        "/spend/logs",
        "/router/settings",
    ],
)
def test_management_routes_are_allowed_with_a_normalised_leading_slash(path: str) -> None:
    assert decide(path) == Allowed("/" + path.lstrip("/"))


@pytest.mark.parametrize(
    ("path", "fragment"),
    [
        ("/v1/chat/completions", "inference"),
        ("/chat/completions", "inference"),
        ("/v1/embeddings", "inference"),
        ("/v1/messages", "inference"),
        ("/v1/mcp/server/oauth/client/register", "OAuth"),
        ("/v1/mcp/server/oauth/client/callback", "OAuth"),
        ("/guardrails/apply_guardrail", "guardrail apply"),
        ("/guardrails/apply_guardrail/123", "guardrail apply"),
        ("/mcp/tools/call", "inference"),
        ("/login", "auth bootstrap"),
        ("/sso/key/generate", "auth bootstrap"),
        ("/key/service-account/generate", "auth bootstrap"),
        ("/key%2Fservice-account%2Fgenerate", "auth bootstrap"),
        ("/key/service-account/generate/", "auth bootstrap"),
        ("/key/service-account/generate%2F", "auth bootstrap"),
        ("/config/yaml", "auth bootstrap"),
        ("/key/../config/yaml", "traversal"),
        ("/key//info", "traversal"),
        ("/health/drain", "lifecycle"),
        ("/health/drain/", "lifecycle"),
    ],
)
def test_inference_auth_and_traversal_are_denied(path: str, fragment: str) -> None:
    decision: Final = decide(path)
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
        "/router_settings",
        "/healthz",
        "/settings_export",
    ],
)
def test_encoded_dotted_or_boundary_breaking_paths_are_denied(path: str) -> None:
    assert isinstance(decide(path), Denied), path


@pytest.mark.parametrize("path", ["/credentials", "/credentials/by_name/x", "/audit", "/audit/1", "/settings"])
def test_bare_management_routes_match_only_on_a_route_boundary(path: str) -> None:
    assert decide(path) == Allowed(path)


@pytest.mark.parametrize("path", ["/health/readiness", "/health/liveliness", "/health/liveness", "/health/latest"])
def test_safe_health_routes_are_allowed(path: str) -> None:
    assert decide(path) == Allowed(path)


@pytest.mark.parametrize("version", ["", "/v1", "/v2", "/v12"])
@pytest.mark.parametrize("route", ["/key/service-account/generate", "/guardrails/apply_guardrail"])
@pytest.mark.parametrize("suffix", ["", "/", "/child"])
def test_versioned_bootstrap_and_guardrail_routes_are_always_denied(version: str, route: str, suffix: str) -> None:
    decision: Final = decide(version + route + suffix)
    assert isinstance(decision, Denied)
    assert "auth bootstrap" in decision.reason


@pytest.mark.parametrize(
    "route",
    [
        "oauth/session",
        "oauth/server-id/authorize",
        "oauth/server-id/token",
        "oauth/server-id/register",
        "server-id/oauth-user-credential",
        "server-id/oauth-user-credential/status",
        "oauth2/client/register",
        "server-id/OAUTH-user-credential",
        "OaUtH2/client/token",
    ],
)
def test_mcp_oauth_prefixed_segments_cannot_forward_credentials(route: str) -> None:
    decision: Final = decide("/v1/mcp/server/" + route)
    assert isinstance(decision, Denied)
    assert "OAuth" in decision.reason


@pytest.mark.parametrize("route", ["list", "new", "server-id", "server-id/health"])
def test_mcp_management_without_oauth_segments_remains_allowed(route: str) -> None:
    path: Final = "/v1/mcp/server/" + route
    assert decide(path) == Allowed(path)


@pytest.mark.parametrize(
    "path",
    [
        "/health",
        "/health/services",
        "/health/test_connection",
        "/health/readiness/details",
        "/health/liveness/details",
    ],
)
def test_health_routes_that_probe_or_expose_details_are_denied(path: str) -> None:
    assert isinstance(decide(path), Denied)


def test_decided_path_is_the_decoded_path_the_dataplane_will_receive() -> None:
    assert decide("key/info%3Fx") == Denied("/key/info%3Fx", "encoded or reserved path characters are not allowed")
    assert decide("/team/my%20team/info") == Allowed("/team/my team/info")


def test_single_trailing_slash_is_removed_without_allowing_empty_segments() -> None:
    root: Final = decide("/")
    assert root == Denied("/", "only management routes are proxied, inference goes straight to the dataplane")
    assert decide("key/list/") == Allowed("/key/list")
    assert isinstance(decide("//x"), Denied)
    assert isinstance(decide("/a//b"), Denied)
    assert isinstance(decide("/key/list//"), Denied)
