from typing import Final, Literal

import pytest
from mcp import MCPError
from mcp.types import CallToolRequest, CallToolRequestParams, InputRequiredResult

from litellm.proxy._experimental.mcp_server.contracts import OperationContext
from litellm.proxy._experimental.mcp_server.interactions import bind_target, open_continuation, seal_continuation
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def test_continuation_is_repeatable_and_bound_to_caller_operation_and_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "local-continuation-test")
    context: Final = OperationContext(
        _caller=UserAPIKeyAuth(user_id="alice", team_id="team"), mcp_servers=("upstream",)
    )
    operation: Final = CallToolRequest(params=CallToolRequestParams(name="confirm", arguments={"amount": 3}))
    server: Final = MCPServer(server_id="upstream", name="upstream", url="https://example.com/mcp", transport="http")
    sealed: Final = seal_continuation(
        bind_target(InputRequiredResult(request_state="opaque"), server), operation, context, now=100
    )
    retry: Final = operation.model_copy(
        update={"params": operation.params.model_copy(update={"request_state": sealed.request_state})}
    )
    state: Final = open_continuation(retry, context, now=101)
    assert state is not None
    assert state.upstream_state == "opaque"
    assert open_continuation(retry, context, now=102) == state
    with pytest.raises(MCPError, match="Invalid or expired"):
        open_continuation(retry, OperationContext(_caller=UserAPIKeyAuth(user_id="bob", team_id="team")), now=101)
    with pytest.raises(MCPError, match="Invalid or expired"):
        open_continuation(
            retry.model_copy(update={"params": retry.params.model_copy(update={"arguments": {"amount": 4}})}),
            context,
            now=101,
        )
    with pytest.raises(MCPError, match="Invalid or expired"):
        open_continuation(retry, context, now=700)
    monkeypatch.setenv("LITELLM_SALT_KEY", "rotated-test-salt")
    with pytest.raises(MCPError, match="Invalid or expired"):
        open_continuation(retry, context, now=101)


def test_continuation_missing_salt_does_not_reject_initial_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_SALT_KEY", raising=False)
    context: Final = OperationContext(_caller=UserAPIKeyAuth(user_id="alice"))
    operation: Final = CallToolRequest(params=CallToolRequestParams(name="confirm", arguments={}))
    assert open_continuation(operation, context, now=100) is None
    server: Final = MCPServer(server_id="upstream", name="upstream", url="https://example.com/mcp", transport="http")
    with pytest.raises(MCPError, match="LITELLM_SALT_KEY"):
        seal_continuation(bind_target(InputRequiredResult(request_state="opaque"), server), operation, context, now=100)


@pytest.mark.parametrize("identity", [None, UserAPIKeyAuth(), UserAPIKeyAuth(team_id="team")])
def test_continuation_requires_stable_authenticated_principal(
    identity: UserAPIKeyAuth | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt")
    operation: Final = CallToolRequest(params=CallToolRequestParams(name="confirm", arguments={}))
    server: Final = MCPServer(server_id="server", name="server", url="https://example.com/mcp", transport="http")
    with pytest.raises(MCPError, match="authenticated caller identity"):
        seal_continuation(
            bind_target(InputRequiredResult(request_state="opaque"), server),
            operation,
            OperationContext(_caller=identity),
            now=100,
        )


def test_continuation_keeps_original_expiry_and_survives_credential_rotation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt")
    before: Final = OperationContext(_caller=UserAPIKeyAuth(user_id="alice", api_key="old-test-key"))
    after: Final = OperationContext(_caller=UserAPIKeyAuth(user_id="alice", api_key="new-test-key"))
    operation: Final = CallToolRequest(params=CallToolRequestParams(name="confirm", arguments={"a": 1, "b": 2}))
    server: Final = MCPServer(server_id="server", name="server", url="https://example.com/mcp", transport="http")
    bound: Final = bind_target(InputRequiredResult(request_state="opaque"), server)
    first: Final = seal_continuation(bound, operation, before, now=100)
    retry: Final = operation.model_copy(
        update={
            "params": operation.params.model_copy(
                update={"request_state": first.request_state, "arguments": {"b": 2, "a": 1}}
            )
        }
    )
    state: Final = open_continuation(retry, after, now=200)
    assert state is not None
    second: Final = seal_continuation(bound, retry, after, now=699, previous=state)
    final: Final = retry.model_copy(
        update={"params": retry.params.model_copy(update={"request_state": second.request_state})}
    )
    assert open_continuation(final, after, now=699) == state
    with pytest.raises(MCPError, match="Invalid or expired"):
        open_continuation(final, after, now=700)


@pytest.mark.parametrize("change", ["tamper", "team", "org", "principal", "method"])
def test_altered_continuations_are_rejected(change: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from mcp.types import GetPromptRequest, GetPromptRequestParams

    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt")
    context: Final = OperationContext(_caller=UserAPIKeyAuth(user_id="alice", team_id="one", org_id="org"))
    operation: Final = CallToolRequest(params=CallToolRequestParams(name="confirm", arguments={}))
    server: Final = MCPServer(server_id="server", name="server", url="https://example.com/mcp", transport="http")
    first: Final = seal_continuation(
        bind_target(InputRequiredResult(request_state="opaque"), server), operation, context, now=100
    )
    token: Final = first.request_state
    assert token is not None
    retry: Final = (
        GetPromptRequest(params=GetPromptRequestParams(name="confirm", arguments={}, request_state=token))
        if change == "method"
        else operation.model_copy(
            update={
                "params": operation.params.model_copy(
                    update={"request_state": token + "a" if change == "tamper" else token}
                )
            }
        )
    )
    caller: Final = UserAPIKeyAuth(
        user_id="bob" if change == "principal" else "alice",
        team_id="two" if change == "team" else "one",
        org_id="other" if change == "org" else "org",
    )
    with pytest.raises(MCPError) as error:
        open_continuation(retry, OperationContext(_caller=caller), now=101)
    assert error.value.error.code == -32602


def test_input_responses_without_state_and_unbound_results_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from mcp.types import ElicitResult
    from litellm.proxy._experimental.mcp_server.interactions import BoundInputRequiredResult

    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt")
    context: Final = OperationContext(_caller=UserAPIKeyAuth(user_id="alice"))
    operation: Final = CallToolRequest(
        params=CallToolRequestParams(name="confirm", input_responses={"consent": ElicitResult(action="accept")})
    )
    with pytest.raises(MCPError, match="require a gateway continuation"):
        open_continuation(operation, context, now=100)
    with pytest.raises(MCPError, match="target is unavailable"):
        seal_continuation(BoundInputRequiredResult(request_state="opaque"), operation, context, now=100)


def test_authenticated_but_invalid_state_payload_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy._experimental.mcp_server.state_tokens import seal_state
    from litellm.proxy._experimental.mcp_server.outbound_credentials.result import Ok

    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt")
    sealed: Final = seal_state(
        {"unsupported": "state-schema"}, purpose="mcp:interaction:repeatable:v1", expires_at=200, now=100
    )
    assert isinstance(sealed, Ok)
    operation: Final = CallToolRequest(params=CallToolRequestParams(name="confirm", request_state=sealed.ok))
    with pytest.raises(MCPError, match="Invalid or expired"):
        open_continuation(operation, OperationContext(_caller=UserAPIKeyAuth(user_id="alice")), now=101)


@pytest.mark.asyncio
async def test_modern_sampling_errors_abort_without_relaying_form_input() -> None:
    import anyio
    from mcp import ClientSession
    from mcp.shared.message import SessionMessage
    from mcp.types import (
        CreateMessageRequest,
        CreateMessageRequestParams,
        ElicitRequest,
        ElicitRequestFormParams,
        ErrorData,
    )
    from litellm.proxy._experimental.mcp_server.interactions import ModernClientInteraction

    async def sampling(context: object, params: CreateMessageRequestParams) -> ErrorData:
        return ErrorData(code=-32603, message="Sampling refused by gateway policy")

    send, receive = anyio.create_memory_object_stream[SessionMessage](1)
    try:
        interaction: Final = ModernClientInteraction(ClientSession(receive, send, sampling_callback=sampling), allow_elicitation=True)
        form: Final = ElicitRequest(
            params=ElicitRequestFormParams(message="Confirm", requested_schema={"type": "object", "properties": {}})
        )
        refused: Final = await interaction.request("consent", form)
        assert isinstance(refused, ErrorData)
        assert refused.code == -32602
        with pytest.raises(MCPError, match="Sampling refused by gateway policy"):
            await interaction.prepare(
                InputRequiredResult(
                    request_state="opaque",
                    input_requests={
                        "consent": form,
                        "sample": CreateMessageRequest(params=CreateMessageRequestParams(messages=[], max_tokens=1)),
                    },
                )
            )
    finally:
        await send.aclose()
        await receive.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_response", ["legacy_state", "gateway_override", "unbound_result"])
async def test_gateway_rejects_invalid_interaction_boundaries(
    invalid_response: Literal["legacy_state", "gateway_override", "unbound_result"], monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import AsyncMock, patch
    from mcp.types import CreateMessageResult, ElicitResult, TextContent
    from litellm.proxy._experimental.mcp_server import operations
    from litellm.proxy._experimental.mcp_server.contracts import WireCompat
    from litellm.proxy._experimental.mcp_server.interactions import BoundInputRequiredResult

    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt")
    context: Final = OperationContext(
        _caller=UserAPIKeyAuth(user_id="alice"),
        wire_compat=WireCompat.LEGACY if invalid_response == "legacy_state" else WireCompat.MODERN,
    )
    operation: Final = CallToolRequest(params=CallToolRequestParams(name="confirm", arguments={}))
    server: Final = MCPServer(server_id="server", name="server", url="https://example.com/mcp", transport="http")
    if invalid_response == "legacy_state":
        operation.params.request_state = "untrusted-state"
    elif invalid_response == "gateway_override":
        bound: Final = bind_target(
            BoundInputRequiredResult(
                request_state="opaque",
                gateway_responses={
                    "sample": CreateMessageResult(
                        role="assistant", content=TextContent(type="text", text="gateway-owned"), model="test-model"
                    )
                },
            ),
            server,
        )
        sealed: Final = seal_continuation(bound, operation, context, now=100)
        operation.params.request_state = sealed.request_state
        operation.params.input_responses = {"sample": ElicitResult(action="accept")}
    message: Final = {
        "legacy_state": "Continuations require the modern MCP protocol",
        "gateway_override": "Cannot replace gateway input responses",
        "unbound_result": "MCP continuation target is unavailable",
    }[invalid_response]
    dispatch: Final = AsyncMock(return_value=InputRequiredResult(request_state="unbound-upstream-state"))
    with (
        patch.object(operations, "_execute_mcp_server_tool_call", dispatch),
        patch.object(operations.global_mcp_server_manager, "get_mcp_server_by_id", return_value=server),
        patch.object(operations.time, "time", return_value=101),
    ):
        with pytest.raises(MCPError, match=message) as rejected:
            await operations.GatewayOperations().execute(operation, context)
    assert rejected.value.error.code == -32602
    if invalid_response == "unbound_result":
        dispatch.assert_awaited_once()
    else:
        dispatch.assert_not_awaited()
