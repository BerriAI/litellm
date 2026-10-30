from typing import Final

import pytest

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import check_tools_allowlist
from litellm.proxy.auth.native_mcp_policy import (
    NativeMCPAccess,
    NativeMCPDenial,
    check_native_mcp_request,
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


def request(tools: object = ("search_code",), url: str = URL) -> dict[str, object]:
    return {"tools": [{"type": "mcp", "server_label": "github", "server_url": url, "allowed_tools": tools}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/v1/responses", "/v1/messages", "/v1/chat/completions"])
async def test_existing_allowlist_hook_rejects_ungranted_native_tool(route: str) -> None:
    with pytest.raises(ProxyException, match="delete_repo") as error:
        await check_tools_allowlist(request(("delete_repo",)), identity(), None, route)
    assert error.value.code == "400"
    assert error.value.type == "tool_access_denied"


@pytest.mark.parametrize(
    "body",
    [
        request(("delete_repo",)),
        request(url="https://other.example.com/github"),
        request(None),
        {"mcp_servers": [{"type": "url", "name": "github", "url": URL}]},
        {"tools": [{"type": "mcp", "server_label": "github", "server_url": URL, "allowed_tools": ["search_code", 42]}]},
        {"tools": [{"type": "namespace", "tools": request(("delete_repo",))["tools"]}]},
        {"extra_body": request(("delete_repo",))},
    ],
)
def test_native_policy_rejects_ungranted_or_unverifiable_attachments(body: dict[str, object]) -> None:
    with pytest.raises(ProxyException):
        check_native_mcp_request(identity(), body)


@pytest.mark.parametrize(
    "body",
    [
        request(),
        request({"tool_names": ["search_code"]}),
        {
            "mcp_servers": [
                {"type": "url", "name": "github", "url": URL, "tool_configuration": {"allowed_tools": ["search_code"]}}
            ]
        },
        {
            "mcp_servers": [{"type": "url", "name": "github", "url": URL}],
            "tools": [
                {
                    "type": "mcp_toolset",
                    "mcp_server_name": "github",
                    "default_config": {"enabled": False},
                    "configs": {"search_code": {"enabled": True}, "delete_repo": {"enabled": False}},
                }
            ],
        },
    ],
)
def test_native_policy_accepts_bounded_requests_without_rewriting(body: dict[str, object]) -> None:
    import copy

    before: Final = copy.deepcopy(body)
    assert check_native_mcp_request(identity(), body) is None
    assert body == before


def test_missing_policy_preserves_behavior_but_empty_policy_denies_native_mcp() -> None:
    assert check_native_mcp_request(UserAPIKeyAuth(), request()) is None
    with pytest.raises(ProxyException):
        check_native_mcp_request(UserAPIKeyAuth(metadata={"native_mcp_policy": {}}), request())


def test_team_and_key_grants_union_only_for_the_same_url() -> None:
    caller: Final = UserAPIKeyAuth(
        team_metadata={"native_mcp_policy": policy(["search_code"])},
        metadata={"native_mcp_policy": policy(["get_file"])},
    )
    assert check_native_mcp_request(caller, request(["search_code", "get_file"])) is None
    wrong_url: Final = caller.model_copy(
        update={"metadata": {"native_mcp_policy": policy(["get_file"], "https://other.example.com/mcp")}}
    )
    with pytest.raises(ProxyException, match="get_file"):
        check_native_mcp_request(wrong_url, request(["search_code", "get_file"]))


def test_whole_server_grant_permits_unbounded_attachment() -> None:
    assert check_native_mcp_request(identity("*"), request(None)) is None


def test_unrelated_data_field_cannot_hide_native_mcp_attachment() -> None:
    with pytest.raises(ProxyException, match="delete_repo"):
        check_native_mcp_request(identity(), {**request(["delete_repo"]), "data": {"unrelated": True}})


@pytest.mark.parametrize("wrapped", [False, True])
def test_native_tool_choice_cannot_select_an_ungranted_tool(wrapped: bool) -> None:
    selection: Final = {"type": "mcp", "server_label": "github", "name": "delete_repo"}
    choice: Final = {"type": "allowed_tools", "tools": [selection], "mode": "required"} if wrapped else selection
    with pytest.raises(ProxyException, match="delete_repo"):
        check_native_mcp_request(identity(), {**request(), "tool_choice": choice})


def test_direct_provider_passthrough_is_outside_v1_scope() -> None:
    caller: Final = identity().model_copy(update={"request_route": "/bedrock/model/test/invoke"})
    assert check_native_mcp_request(caller, request(["delete_repo"])) is None


def test_empty_native_policy_preserves_gateway_controls() -> None:
    caller: Final = UserAPIKeyAuth(metadata={"native_mcp_policy": {}})
    gateway: Final = request(None, "litellm_proxy/mcp/github")
    assert check_native_mcp_request(caller, gateway) is None


def test_mixed_gateway_and_native_attachments_are_rejected_only_with_active_policy() -> None:
    mixed: Final = {
        "tools": [
            {"type": "mcp", "server_url": "litellm_proxy/mcp/github"},
            {"type": "mcp", "server_label": "github", "server_url": URL, "allowed_tools": ["search_code"]},
        ]
    }
    assert check_native_mcp_request(UserAPIKeyAuth(), mixed) is None
    with pytest.raises(ProxyException, match="Mixing provider-native MCP"):
        check_native_mcp_request(identity(), mixed)
