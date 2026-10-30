from typing import Final

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.native_mcp_policy import (
    NativeMCPAccess,
    NativeMCPDenial,
    evaluate_native_mcp_policy,
    resolve_native_mcp_policy,
)

URL: Final = "https://mcp.example.com/github"


def policy(tools: object = ("search_code",), url: str = URL) -> dict[str, object]:
    return {"mcp_servers": [{"name": "github", "url": url, "tools": tools}]}


def identity(tools: object = ("search_code",)) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(metadata={"native_mcp_policy": policy(tools)})


@pytest.mark.parametrize(
    "metadata,allowed",
    [({}, True), ({"native_mcp_policy": {}}, False), ({"native_mcp_policy": policy()}, True)],
)
def test_evaluator_distinguishes_absent_empty_and_granted_policy(metadata: dict[str, object], allowed: bool) -> None:
    resolved: Final = resolve_native_mcp_policy(UserAPIKeyAuth(metadata=metadata))
    decision: Final = evaluate_native_mcp_policy(resolved, (NativeMCPAccess("github", URL, ("search_code",)),))
    assert (decision is None) == allowed


@pytest.mark.parametrize(
    "access",
    [
        NativeMCPAccess("github", URL, ("delete_repo",)),
        NativeMCPAccess("github", "https://other.example.com/mcp", ("search_code",)),
        NativeMCPAccess("other", URL, ("search_code",)),
        NativeMCPAccess("github", URL, "*"),
    ],
)
def test_evaluator_requires_matching_server_url_and_tool_grants(access: NativeMCPAccess) -> None:
    decision: Final = evaluate_native_mcp_policy(resolve_native_mcp_policy(identity()), (access,))
    assert isinstance(decision, NativeMCPDenial)


def test_evaluator_unions_team_and_key_grants_without_combining_different_urls() -> None:
    same_url: Final = UserAPIKeyAuth(
        team_metadata={"native_mcp_policy": policy(["search_code"])},
        metadata={"native_mcp_policy": policy(["get_file"])},
    )
    accesses: Final = (NativeMCPAccess("github", URL, ("search_code", "get_file")),)
    assert evaluate_native_mcp_policy(resolve_native_mcp_policy(same_url), accesses) is None
    other_url: Final = same_url.model_copy(
        update={"metadata": {"native_mcp_policy": policy(["get_file"], URL + "/other")}}
    )
    denied: Final = evaluate_native_mcp_policy(resolve_native_mcp_policy(other_url), accesses)
    assert isinstance(denied, NativeMCPDenial)
    assert "get_file" in denied.message


def test_evaluator_whole_server_grant_permits_unbounded_access() -> None:
    assert (
        evaluate_native_mcp_policy(resolve_native_mcp_policy(identity("*")), (NativeMCPAccess("github", URL, "*"),))
        is None
    )


def test_evaluator_malformed_stored_policy_denies_even_with_other_valid_grants() -> None:
    caller: Final = identity().model_copy(update={"team_metadata": {"native_mcp_policy": {"mcp_servers": "invalid"}}})
    assert isinstance(evaluate_native_mcp_policy(resolve_native_mcp_policy(caller), ()), NativeMCPDenial)


def test_evaluator_gateway_only_keeps_existing_permissions_but_mixed_is_rejected() -> None:
    resolved: Final = resolve_native_mcp_policy(identity())
    gateway: Final = NativeMCPAccess("local", "litellm_proxy/mcp/local", (), "gateway")
    assert evaluate_native_mcp_policy(resolved, (gateway,)) is None
    assert isinstance(
        evaluate_native_mcp_policy(resolved, (gateway, NativeMCPAccess("github", URL, ("search_code",)))),
        NativeMCPDenial,
    )
