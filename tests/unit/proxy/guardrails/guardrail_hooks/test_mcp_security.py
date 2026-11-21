"""
Tests for MCP Security Guardrail.

Validates that the guardrail blocks requests referencing unregistered MCP servers
and allows requests with only registered servers. Covers both /chat/completions
and /responses API paths (same pre_call_hook logic, different call_type).
"""

import io
import json
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.mcp_security import initialize_guardrail
from litellm.proxy.guardrails.guardrail_hooks.mcp_security.mcp_security_guardrail import (
    MCPSecurityGuardrail,
)
from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
from litellm.proxy.openai_files_endpoints.batch_guardrails import (
    BatchScanResult,
    RecordDropped,
    scan_batch_input_file,
)
from litellm.types.guardrails import Guardrail, GuardrailEventHooks, LitellmParams
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.types.mcp_server.mcp_toolset import MCPToolset


@pytest.fixture
def guardrail():
    return MCPSecurityGuardrail(
        guardrail_name="test-mcp-security",
        event_hook=GuardrailEventHooks.pre_call,
        default_on=True,
        on_violation="block",
    )


class TestExtractMCPServerNames:
    def test_extracts_litellm_proxy_mcp_servers(self):
        tools = [
            {"type": "mcp", "server_url": "litellm_proxy/mcp/zapier"},
            {"type": "mcp", "server_url": "litellm_proxy/mcp/github"},
            {"type": "function", "function": {"name": "get_weather"}},
        ]
        names = MCPSecurityGuardrail._extract_mcp_server_names_from_tools(tools)
        assert names == {"zapier", "github"}

    def test_ignores_non_mcp_tools(self):
        tools = [
            {"type": "function", "function": {"name": "get_weather"}},
        ]
        names = MCPSecurityGuardrail._extract_mcp_server_names_from_tools(tools)
        assert names == set()

    def test_ignores_external_mcp_servers(self):
        tools = [
            {"type": "mcp", "server_url": "https://external-server.com/mcp"},
        ]
        names = MCPSecurityGuardrail._extract_mcp_server_names_from_tools(tools)
        assert names == set()

    def test_empty_tools(self):
        names = MCPSecurityGuardrail._extract_mcp_server_names_from_tools([])
        assert names == set()


def _manager_with_saved_server() -> MCPServerManager:
    manager: Final = MCPServerManager()
    server: Final = MCPServer(
        server_id="3f6b2c1e-saved-zapier",
        name="zapier",
        server_name="zapier",
        alias="zap",
        transport=MCPTransport.http,
        url="https://upstream.example/mcp",
    )
    manager.registry[server.server_id] = server
    return manager


class TestMCPSecurityGuardrailPreCall:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("call_type", ["acompletion", "aresponses"])
    async def test_blocks_unregistered_server(self, guardrail, call_type):
        data = {
            "tools": [
                {"type": "mcp", "server_url": "litellm_proxy/mcp/zapier"},
                {"type": "mcp", "server_url": "litellm_proxy/mcp/evil_server"},
            ],
            "model": "gpt-4",
            "guardrails": ["test-mcp-security"],
        }

        with (
            patch("litellm.proxy.proxy_server.prisma_client", None),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager", _manager_with_saved_server()),
            pytest.raises(HTTPException) as exc_info,
        ):
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=MagicMock(),
                data=data,
                call_type=call_type,
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["unregistered_servers"] == ["evil_server"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("call_type", ["acompletion", "aresponses"])
    @pytest.mark.parametrize("reference", ["zapier", "zap", "3f6b2c1e-saved-zapier"])
    async def test_allows_saved_server_by_any_name_the_gateway_routes(self, guardrail, call_type, reference):
        data = {
            "tools": [{"type": "mcp", "server_url": f"litellm_proxy/mcp/{reference}"}],
            "model": "gpt-4",
            "guardrails": ["test-mcp-security"],
        }

        with (
            patch("litellm.proxy.proxy_server.prisma_client", None),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager", _manager_with_saved_server()),
        ):
            result = await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=MagicMock(),
                data=data,
                call_type=call_type,
            )
        assert result == data

    @pytest.mark.asyncio
    async def test_passthrough_no_mcp_tools(self, guardrail):
        """Request with no MCP tools passes through without checking registry."""
        data = {
            "tools": [
                {"type": "function", "function": {"name": "get_weather"}},
            ],
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "guardrails": ["test-mcp-security"],
        }

        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(),
            data=data,
            call_type="acompletion",
        )
        assert result == data

    @pytest.mark.asyncio
    async def test_passthrough_no_tools(self, guardrail):
        """Request with no tools at all passes through."""
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "guardrails": ["test-mcp-security"],
        }

        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(),
            data=data,
            call_type="acompletion",
        )
        assert result == data


class TestInitializeGuardrail:
    @pytest.mark.parametrize(
        "configured,expected",
        [("block", "block"), ("alert", "alert"), (None, "alert"), ("warn", "alert"), ("end_session", "alert")],
    )
    def test_on_violation_from_litellm_params(
        self,
        configured: Literal["block", "alert", "warn", "end_session"] | None,
        expected: Literal["block", "alert"],
    ):
        litellm_params = LitellmParams(guardrail="mcp_security", mode="pre_call", on_violation=configured)
        guardrail = Guardrail(guardrail_name="mcp-security-block", litellm_params=litellm_params)

        result = initialize_guardrail(litellm_params=litellm_params, guardrail=guardrail)

        assert isinstance(result, MCPSecurityGuardrail)
        assert result.on_violation == expected
        assert result in litellm.callbacks


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ["acompletion", "aresponses"])
async def test_guardrail_observes_saved_server_creation_and_deletion_on_another_worker(guardrail, call_type):
    from unittest.mock import AsyncMock

    from litellm.proxy._types import LiteLLM_MCPServerTable
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager

    manager = MCPServerManager()
    row = LiteLLM_MCPServerTable(server_id="peer-server", alias="peer_server", transport="http",
        url="https://upstream.example/mcp")
    prisma = MagicMock()
    prisma.writer_db = prisma.db
    prisma.db.litellm_mcpservertable.find_many = AsyncMock(side_effect=([row], []))
    prisma.db.litellm_config.find_unique = AsyncMock(return_value=None)
    prisma.db.litellm_mcptoolsettable.find_first = AsyncMock(return_value=None)
    data = {"tools": [{"type": "mcp", "server_url": "litellm_proxy/mcp/peer-server"}],
        "guardrails": ["test-mcp-security"]}
    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.general_settings", {}),
        patch("litellm.proxy.proxy_server.user_api_key_cache", DualCache()),
        patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager", manager),
    ):
        result = await guardrail.async_pre_call_hook(UserAPIKeyAuth(), MagicMock(), data, call_type)
        assert result == data
        with pytest.raises(HTTPException) as exc:
            await guardrail.async_pre_call_hook(UserAPIKeyAuth(), MagicMock(), data, call_type)
    assert exc.value.status_code == 400
    assert exc.value.detail["unregistered_servers"] == ["peer-server"]


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ["acompletion", "aresponses"])
async def test_allows_saved_toolset_by_name_and_blocks_unknown_name(guardrail, call_type):
    saved_toolset: Final = MCPToolset(toolset_id="7c1d-research", toolset_name="research")

    async def find_toolset(*, where: dict[str, str]) -> MCPToolset | None:
        return saved_toolset if where["toolset_name"] == saved_toolset.toolset_name else None

    prisma: Final = MagicMock()
    prisma.writer_db = prisma.db
    prisma.db.litellm_mcpservertable.find_many = AsyncMock(return_value=[])
    prisma.db.litellm_config.find_unique = AsyncMock(return_value=None)
    prisma.db.litellm_mcptoolsettable.find_first = AsyncMock(side_effect=find_toolset)
    data: Final = {
        "tools": [
            {"type": "mcp", "server_url": "litellm_proxy/mcp/research"},
            {"type": "mcp", "server_url": "litellm_proxy/mcp/evil_server"},
        ],
        "guardrails": ["test-mcp-security"],
    }
    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.general_settings", {}),
        patch("litellm.proxy.proxy_server.user_api_key_cache", DualCache()),
        patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager", MCPServerManager()),
        pytest.raises(HTTPException) as exc,
    ):
        await guardrail.async_pre_call_hook(UserAPIKeyAuth(), MagicMock(), data, call_type)
    assert exc.value.detail["unregistered_servers"] == ["evil_server"]


def _prisma_whose_toolset_read_fails() -> MagicMock:
    prisma: Final = MagicMock()
    prisma.writer_db = prisma.db
    prisma.db.litellm_mcpservertable.find_many = AsyncMock(return_value=[])
    prisma.db.litellm_config.find_unique = AsyncMock(return_value=None)
    prisma.db.litellm_mcptoolsettable.find_first = AsyncMock(side_effect=RuntimeError("toolset table unavailable"))
    return prisma


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ["acompletion", "aresponses"])
async def test_blocks_the_name_with_the_policy_result_when_the_toolset_read_fails(guardrail, call_type):
    data: Final = {
        "tools": [{"type": "mcp", "server_url": "litellm_proxy/mcp/research"}],
        "guardrails": ["test-mcp-security"],
    }
    with (
        patch("litellm.proxy.proxy_server.prisma_client", _prisma_whose_toolset_read_fails()),
        patch("litellm.proxy.proxy_server.general_settings", {}),
        patch("litellm.proxy.proxy_server.user_api_key_cache", DualCache()),
        patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager", MCPServerManager()),
        pytest.raises(HTTPException) as exc,
    ):
        await guardrail.async_pre_call_hook(UserAPIKeyAuth(), MagicMock(), data, call_type)
    assert exc.value.status_code == 400
    assert exc.value.detail["unregistered_servers"] == ["research"]


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ["acompletion", "aresponses"])
async def test_alert_mode_lets_the_request_through_when_the_toolset_read_fails(call_type, caplog):
    alerting: Final = MCPSecurityGuardrail(
        guardrail_name="test-mcp-security",
        event_hook=GuardrailEventHooks.pre_call,
        default_on=True,
        on_violation="alert",
    )
    data: Final = {
        "tools": [{"type": "mcp", "server_url": "litellm_proxy/mcp/research"}],
        "guardrails": ["test-mcp-security"],
    }
    with (
        patch("litellm.proxy.proxy_server.prisma_client", _prisma_whose_toolset_read_fails()),
        patch("litellm.proxy.proxy_server.general_settings", {}),
        patch("litellm.proxy.proxy_server.user_api_key_cache", DualCache()),
        patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager", MCPServerManager()),
        caplog.at_level("WARNING"),
    ):
        result: Final = await alerting.async_pre_call_hook(UserAPIKeyAuth(), MagicMock(), data, call_type)
    assert result == data
    assert any("unregistered MCP server(s): research" in record.getMessage() for record in caplog.records)


def _batch_record_naming(server: str) -> io.BytesIO:
    record: Final = {
        "custom_id": "row-mcp",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "us.amazon.nova-2-lite-v1:0",
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "tools": [{"type": "mcp", "server_url": f"litellm_proxy/mcp/{server}"}],
        },
    }
    return io.BytesIO(json.dumps(record).encode())


def _proxy_logging_running(guardrail: MCPSecurityGuardrail) -> MagicMock:
    async def pre_call_hook(
        user_api_key_dict: UserAPIKeyAuth, data: dict, call_type: str, guardrails_only: bool = False
    ) -> Exception | str | dict | None:
        return await guardrail.async_pre_call_hook(user_api_key_dict, MagicMock(), data, call_type)

    proxy_logging: Final = MagicMock()
    proxy_logging.pre_call_hook = pre_call_hook
    return proxy_logging


def _prisma_without_toolsets() -> MagicMock:
    prisma: Final = MagicMock()
    prisma.writer_db = prisma.db
    prisma.db.litellm_mcpservertable.find_many = AsyncMock(return_value=[])
    prisma.db.litellm_config.find_unique = AsyncMock(return_value=None)
    prisma.db.litellm_mcptoolsettable.find_first = AsyncMock(return_value=None)
    return prisma


@pytest.mark.asyncio
async def test_a_batch_record_naming_an_unknown_server_is_dropped_while_the_toolset_read_works(guardrail):
    with (
        patch("litellm.proxy.proxy_server.prisma_client", _prisma_without_toolsets()),
        patch("litellm.proxy.proxy_server.general_settings", {}),
        patch("litellm.proxy.proxy_server.user_api_key_cache", DualCache()),
        patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager", MCPServerManager()),
    ):
        result: Final = await scan_batch_input_file(
            file_source=_batch_record_naming("research"),
            request_metadata={},
            user_api_key_dict=UserAPIKeyAuth(),
            proxy_logging_obj=_proxy_logging_running(guardrail),
        )
    assert isinstance(result, BatchScanResult)
    assert result.changes == (RecordDropped(line_number=1, custom_id="row-mcp", guardrail=None),)


@pytest.mark.asyncio
async def test_a_batch_upload_is_refused_instead_of_dropping_a_record_when_the_toolset_read_fails(guardrail):
    with (
        patch("litellm.proxy.proxy_server.prisma_client", _prisma_whose_toolset_read_fails()),
        patch("litellm.proxy.proxy_server.general_settings", {}),
        patch("litellm.proxy.proxy_server.user_api_key_cache", DualCache()),
        patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager", MCPServerManager()),
        pytest.raises(HTTPException) as exc,
    ):
        await scan_batch_input_file(
            file_source=_batch_record_naming("research"),
            request_metadata={},
            user_api_key_dict=UserAPIKeyAuth(),
            proxy_logging_obj=_proxy_logging_running(guardrail),
        )
    assert exc.value.status_code == 400
    assert exc.value.detail["unregistered_servers"] == ["research"]
