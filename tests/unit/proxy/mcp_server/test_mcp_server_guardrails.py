"""Per-MCP-server ``guardrails``: a server's listed guardrails run for its tool calls only."""

from typing import Final, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import CallToolResult, TextContent

from litellm.integrations.custom_guardrail import MCP_SERVER_GUARDRAILS_KEY, CustomGuardrail
from litellm.proxy.mcp_server import operations
from litellm.proxy.mcp_server.mcp_server_manager import MCPServerManager
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer

SCOPED: Final = "scoped-guardrail"


def _guardrail(mode: GuardrailEventHooks, default_on: bool = False) -> CustomGuardrail:
    return CustomGuardrail(guardrail_name=SCOPED, event_hook=mode, default_on=default_on)


def _server(guardrails: list[str] | None) -> MCPServer:
    return MCPServer(
        server_id="tickets-id",
        name="tickets",
        server_name="tickets",
        url="https://example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        guardrails=guardrails,
    )


@pytest.mark.parametrize("hook", [GuardrailEventHooks.pre_mcp_call, GuardrailEventHooks.post_mcp_call])
def test_listed_guardrail_runs_for_its_server_without_default_on(hook: GuardrailEventHooks) -> None:
    assert _guardrail(hook).should_run_guardrail({MCP_SERVER_GUARDRAILS_KEY: [SCOPED]}, hook) is True


@pytest.mark.parametrize("server_guardrails", [[], ["other"]])
def test_unlisted_guardrail_does_not_run_on_another_server(server_guardrails: list[str]) -> None:
    guardrail: Final = _guardrail(GuardrailEventHooks.pre_mcp_call)
    data: Final = {MCP_SERVER_GUARDRAILS_KEY: server_guardrails}
    assert guardrail.should_run_guardrail(data, GuardrailEventHooks.pre_mcp_call) is False


def test_listed_guardrail_still_honours_its_mode() -> None:
    guardrail: Final = _guardrail(GuardrailEventHooks.post_mcp_call)
    data: Final = {MCP_SERVER_GUARDRAILS_KEY: [SCOPED]}
    assert guardrail.should_run_guardrail(data, GuardrailEventHooks.pre_mcp_call) is False


def test_server_list_is_ignored_outside_mcp_events() -> None:
    guardrail: Final = CustomGuardrail(guardrail_name=SCOPED, event_hook=GuardrailEventHooks.pre_call)
    data: Final = {MCP_SERVER_GUARDRAILS_KEY: [SCOPED]}
    assert guardrail.should_run_guardrail(data, GuardrailEventHooks.pre_call) is False


def test_server_listing_overrides_key_opt_out() -> None:
    guardrail: Final = _guardrail(GuardrailEventHooks.pre_mcp_call, default_on=True)
    data: Final = {MCP_SERVER_GUARDRAILS_KEY: [SCOPED], "metadata": {"disable_global_guardrail": True}}
    assert guardrail.should_run_guardrail(data, GuardrailEventHooks.pre_mcp_call) is True


@pytest.mark.asyncio
async def test_config_guardrails_are_loaded_onto_the_server(
    config_only_mcp_manager_factory: type[MCPServerManager],
) -> None:
    manager: Final = config_only_mcp_manager_factory()
    await manager.load_servers_from_config(
        {"tickets": {"url": "https://example.com/mcp", "transport": "http", "guardrails": [SCOPED]}}
    )
    servers: Final = [s for s in manager.config_mcp_servers.values() if s.name == "tickets"]
    assert [s.guardrails for s in servers] == [[SCOPED]]


@pytest.mark.asyncio
@pytest.mark.parametrize(("listed", "expected"), [([SCOPED], [SCOPED]), (None, [])])
async def test_pre_call_tool_check_passes_server_guardrails(listed: list[str] | None, expected: list[str]) -> None:
    manager: Final = MCPServerManager()
    proxy_logging: Final = MagicMock(spec=ProxyLogging)
    proxy_logging.create_mcp_request_object_from_kwargs = MagicMock(return_value=MagicMock())
    proxy_logging.convert_mcp_to_llm_format = MagicMock(return_value={"model": "fake"})
    seen_data: Final[list[dict[str, object]]] = []
    seen_server_guardrails: Final[list[object]] = []

    def has_pre_call_guardrails(*_args: object, **kwargs: object) -> bool:
        seen_server_guardrails.append(kwargs["server_guardrails"])
        return False

    async def pre_call_hook(**kwargs: object) -> None:
        seen_data.append(cast(dict[str, object], kwargs["data"]))

    proxy_logging.has_pre_call_guardrails = has_pre_call_guardrails
    proxy_logging.pre_call_hook = pre_call_hook
    proxy_logging.enforce_mcp_server_rate_limits = AsyncMock()

    with (
        patch.object(manager, "check_allowed_or_banned_tools", return_value=True),
        patch.object(manager, "check_tool_permission_for_key_team", new_callable=AsyncMock),
        patch.object(manager, "validate_allowed_params"),
    ):
        await manager.pre_call_tool_check(
            name="get_ticket",
            arguments={},
            server_name="tickets",
            user_api_key_auth=None,
            proxy_logging_obj=proxy_logging,
            server=_server(listed),
        )

    assert [d[MCP_SERVER_GUARDRAILS_KEY] for d in seen_data] == [expected]
    assert seen_server_guardrails == [listed or ()]


@pytest.mark.asyncio
async def test_post_call_guardrails_receive_server_guardrails() -> None:
    result: Final = CallToolResult(content=[TextContent(type="text", text="ok")])
    proxy_logging: Final = MagicMock(spec=ProxyLogging)
    seen_request_data: Final[list[dict[str, object]]] = []

    async def post_mcp_call_hook(**kwargs: object) -> CallToolResult:
        seen_request_data.append(cast(dict[str, object], kwargs["request_data"]))
        return result

    proxy_logging.post_mcp_call_hook = post_mcp_call_hook

    with patch("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging):
        await operations._run_post_mcp_call_guardrails(  # pyright: ignore[reportPrivateUsage]
            result=result,
            litellm_logging_obj=None,
            original_auth=None,
            guardrail_auth=None,
            request_data={MCP_SERVER_GUARDRAILS_KEY: ["forged"]},
            server_guardrails=(SCOPED,),
        )

    assert [d[MCP_SERVER_GUARDRAILS_KEY] for d in seen_request_data] == [[SCOPED]]
