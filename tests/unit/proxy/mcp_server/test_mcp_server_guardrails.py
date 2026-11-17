"""Per-MCP-server ``guardrails``: a server's listed guardrails run for its tool calls only."""

import logging
from typing import Final, Literal, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import CallToolResult, ListToolsResult, TextContent
from mcp.types import Tool as MCPTool

import litellm
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import MCP_SERVER_GUARDRAILS_KEY, CustomGuardrail
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server import operations
from litellm.proxy.mcp_server.mcp_server_manager import MCPServerManager
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.types.utils import GenericGuardrailAPIInputs

SCOPED: Final = "scoped-guardrail"


def _guardrail(mode: GuardrailEventHooks, default_on: bool = False) -> CustomGuardrail:
    return CustomGuardrail(guardrail_name=SCOPED, event_hook=mode, default_on=default_on)


def _server(guardrails: tuple[str, ...] | None) -> MCPServer:
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
    assert [s.guardrails for s in servers] == [(SCOPED,)]


@pytest.mark.asyncio
@pytest.mark.parametrize(("loaded", "expect_warning"), [((), True), ((SCOPED,), False)])
async def test_config_server_listing_an_unloaded_guardrail_warns(
    config_only_mcp_manager_factory: type[MCPServerManager],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    loaded: tuple[str, ...],
    expect_warning: bool,
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [_guardrail(GuardrailEventHooks.pre_mcp_call) for _ in loaded])
    manager: Final = config_only_mcp_manager_factory()

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        await manager.load_servers_from_config(
            {"tickets": {"url": "https://example.com/mcp", "transport": "http", "guardrails": [SCOPED]}}
        )

    warnings: Final = tuple(message for message in caplog.messages if "guardrails that are not loaded" in message)
    assert len(warnings) == int(expect_warning)
    assert all(SCOPED in message and "tickets" in message for message in warnings)


@pytest.mark.asyncio
@pytest.mark.parametrize(("listed", "expected"), [((SCOPED,), [SCOPED]), (None, [])])
async def test_pre_call_tool_check_passes_server_guardrails(
    listed: tuple[str, ...] | None, expected: list[str]
) -> None:
    manager: Final = MCPServerManager()
    proxy_logging: Final = MagicMock(spec=ProxyLogging)
    proxy_logging.create_mcp_request_object_from_kwargs = MagicMock(return_value=MagicMock())
    proxy_logging.convert_mcp_to_llm_format = MagicMock(return_value={"model": "fake"})
    seen_data: Final[list[dict[str, object]]] = []

    async def pre_call_hook(**kwargs: object) -> None:
        seen_data.append(cast(dict[str, object], kwargs["data"]))

    proxy_logging.has_pre_call_guardrails = MagicMock(return_value=False)
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


class _CallRecordingGuardrail(CustomGuardrail):
    def __init__(self, *, default_on: bool) -> None:
        super().__init__(guardrail_name=SCOPED, event_hook=GuardrailEventHooks.pre_mcp_call, default_on=default_on)
        self.scanned_arguments: tuple[object, ...] = ()

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: Logging | None = None,
    ) -> GenericGuardrailAPIInputs:
        self.scanned_arguments += (request_data.get("mcp_arguments"),)
        return inputs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("listed", "default_on", "upstream_listings"),
    [((SCOPED,), False, 0), (None, True, 1)],
    ids=["server-listed", "default-on"],
)
async def test_server_listed_guardrail_runs_on_the_call_without_an_extra_tool_listing(
    monkeypatch: pytest.MonkeyPatch,
    listed: tuple[str, ...] | None,
    default_on: bool,
    upstream_listings: int,
) -> None:
    server: Final = MCPServer(
        server_id="tickets-id",
        name="tickets",
        server_name="tickets",
        url="http://upstream",
        transport=MCPTransport.http,
        guardrails=listed,
    )
    manager: Final = MCPServerManager()
    manager.registry = {server.server_id: server}
    manager.tool_name_to_mcp_server_name_mapping = {"tickets-get_ticket": "tickets"}
    client: Final = AsyncMock()
    client.discovery_auth_fingerprint = AsyncMock(return_value="test-credential-hash")
    client.call_tool.return_value = CallToolResult(content=[TextContent(type="text", text="ok")])
    manager.create_mcp_client = AsyncMock(return_value=client)
    manager._fetch_tools_with_timeout = AsyncMock(
        return_value=ListToolsResult(tools=[MCPTool(name="get_ticket", description="Read a ticket", inputSchema={})])
    )
    guardrail: Final = _CallRecordingGuardrail(default_on=default_on)
    monkeypatch.setattr(litellm, "callbacks", [guardrail])
    ProxyLogging._callback_capabilities_cache.clear()

    try:
        await manager.call_tool(
            server_name="tickets",
            name="get_ticket",
            arguments={"ticket": "T-1"},
            user_api_key_auth=UserAPIKeyAuth(api_key="sk-test", user_id="user-1"),
            proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
            listed_tool=None,
        )
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert manager._fetch_tools_with_timeout.await_count == upstream_listings
    assert guardrail.scanned_arguments.count({"ticket": "T-1"}) == 1


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
