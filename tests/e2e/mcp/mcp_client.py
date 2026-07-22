"""Client for the MCP e2e suite: admin server registration plus the api_key tool
surface.

An admin registers an upstream MCP server through the management API
(`/v1/mcp/server`, persisted in the DB) and grants a virtual key access to it via
`object_permission.mcp_servers`. Keys then reach the server through the REST bridge
the proxy exposes for api_key auth (`/mcp-rest/tools/list`, `/mcp-rest/tools/call`),
which `user_api_key_auth` gates the same way the JSON-RPC `/mcp` surface does. The
request/response bodies are co-located here because only this suite speaks MCP.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, RootModel

from e2e_http import Headers, NoBody, Result, unwrap
from models import KeyGenerateBody, ObjectPermission
from proxy_client import ProxyClient

McpToolArg = str | int | float | bool | list[str] | dict[str, str]
McpToolArguments = Mapping[str, McpToolArg]


class ApiKeyHeaders(Headers):
    x_litellm_api_key: str = Field(serialization_alias="x-litellm-api-key")


class McpServerNewBody(BaseModel):
    server_name: str
    alias: str
    url: str
    transport: str = "http"
    auth_type: str | None = None
    static_headers: dict[str, str] | None = None
    allowed_tools: list[str] | None = None


class McpServerNewResponse(BaseModel):
    server_id: str


class McpServerRow(BaseModel):
    server_id: str
    alias: str | None = None
    url: str | None = None


class McpServersListResponse(RootModel[list[McpServerRow]]):
    pass


class McpToolMcpInfo(BaseModel):
    server_id: str | None = None
    alias: str | None = None


class McpToolEntry(BaseModel):
    name: str
    description: str | None = None
    mcp_info: McpToolMcpInfo | None = None


class McpToolsListResponse(BaseModel):
    tools: list[McpToolEntry] = []
    error: str | None = None
    message: str | None = None

    def tool_names_for_server(self, server_id: str) -> frozenset[str]:
        return frozenset(
            tool.name
            for tool in self.tools
            if tool.mcp_info is not None and tool.mcp_info.server_id == server_id
        )

    def tool_name_containing(self, server_id: str, needle: str) -> str | None:
        needle_l = needle.lower()
        for tool in self.tools:
            if tool.mcp_info is None or tool.mcp_info.server_id != server_id:
                continue
            if needle_l in tool.name.lower() or tool.name.lower().endswith(needle_l):
                return tool.name
        return None


class BlockedWordSpec(BaseModel):
    keyword: str
    action: str = "BLOCK"


class ContentFilterMcpParams(BaseModel):
    """litellm_content_filter params scoped to the MCP tool-call hook. mode is
    pre_mcp_call because a pre_call config silently no-ops on the tools/call path
    (the event type is rewritten to pre_mcp_call for call_mcp_tool), and default_on
    is required there because per-key/request guardrail selection is dropped from
    the synthetic MCP request the hook sees."""

    guardrail: str = "litellm_content_filter"
    mode: str = "pre_mcp_call"
    default_on: bool = True
    blocked_words: list[BlockedWordSpec]


class GuardrailSpecBody(BaseModel):
    guardrail_name: str
    litellm_params: ContentFilterMcpParams


class GuardrailCreateBody(BaseModel):
    guardrail: GuardrailSpecBody


class GuardrailCreateResponse(BaseModel):
    guardrail_id: str


class McpCallToolBody(BaseModel):
    name: str
    arguments: dict[str, McpToolArg]
    server_id: str


class McpCallContent(BaseModel):
    type: str | None = None
    text: str | None = None


class McpCallToolResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    content: list[McpCallContent] = []
    is_error: bool | None = Field(default=None, alias="isError")

    @property
    def first_text(self) -> str | None:
        return self.content[0].text if self.content else None

    @property
    def all_text(self) -> str:
        return "\n".join(part.text for part in self.content if part.text)


@dataclass(frozen=True, slots=True)
class McpClient:
    proxy: ProxyClient

    def register_server(
        self,
        *,
        server_name: str,
        alias: str,
        url: str,
        transport: str = "http",
        auth_type: str | None = None,
        static_headers: dict[str, str] | None = None,
        allowed_tools: list[str] | None = None,
    ) -> str:
        return unwrap(
            self.proxy.transport.post(
                "/v1/mcp/server",
                headers=self.proxy.transport.master,
                json=McpServerNewBody(
                    server_name=server_name,
                    alias=alias,
                    url=url,
                    transport=transport,
                    auth_type=auth_type,
                    static_headers=static_headers,
                    allowed_tools=allowed_tools,
                ),
                response_type=McpServerNewResponse,
            )
        ).server_id

    def delete_server(self, server_id: str) -> None:
        _ = self.proxy.transport.delete(
            f"/v1/mcp/server/{server_id}",
            headers=self.proxy.transport.master,
            json=NoBody(),
            response_type=NoBody,
        )

    def registered_servers(self) -> list[McpServerRow]:
        return unwrap(
            self.proxy.transport.get(
                "/v1/mcp/server",
                headers=self.proxy.transport.master,
                params=NoBody(),
                response_type=McpServersListResponse,
            )
        ).root

    def generate_key(
        self,
        *,
        user_id: str,
        mcp_servers: list[str] | None,
        models: list[str] | None = None,
    ) -> str:
        object_permission = (
            ObjectPermission(mcp_servers=mcp_servers) if mcp_servers is not None else None
        )
        return self.proxy.generate_key(
            KeyGenerateBody(
                models=models if models is not None else [],
                user_id=user_id,
                object_permission=object_permission,
            )
        )

    def list_tools(self, key: str) -> Result[McpToolsListResponse]:
        return self.proxy.transport.get(
            "/mcp-rest/tools/list",
            headers=ApiKeyHeaders(x_litellm_api_key=key),
            params=NoBody(),
            response_type=McpToolsListResponse,
        )

    def register_mcp_content_filter(self, *, name: str, blocked_keyword: str) -> str:
        """Register a default-on content-filter guardrail that runs on the MCP
        tool-call hook (pre_mcp_call) and blocks a single keyword. The keyword is
        unique per test, so default_on only ever intercepts this test's own
        banned tool call on the shared proxy."""
        return unwrap(
            self.proxy.transport.post(
                "/guardrails",
                headers=self.proxy.transport.master,
                json=GuardrailCreateBody(
                    guardrail=GuardrailSpecBody(
                        guardrail_name=name,
                        litellm_params=ContentFilterMcpParams(
                            blocked_words=[BlockedWordSpec(keyword=blocked_keyword)],
                        ),
                    )
                ),
                response_type=GuardrailCreateResponse,
            )
        ).guardrail_id

    def delete_guardrail(self, guardrail_id: str) -> None:
        _ = self.proxy.transport.delete(
            f"/guardrails/{guardrail_id}",
            headers=self.proxy.transport.master,
            json=NoBody(),
            response_type=NoBody,
        )

    def call_tool(
        self,
        key: str,
        *,
        server_id: str,
        name: str,
        arguments: McpToolArguments,
    ) -> Result[McpCallToolResponse]:
        return self.proxy.transport.post(
            "/mcp-rest/tools/call",
            headers=ApiKeyHeaders(x_litellm_api_key=key),
            json=McpCallToolBody(
                name=name, arguments=dict(arguments), server_id=server_id
            ),
            response_type=McpCallToolResponse,
        )


def build_client(proxy: ProxyClient) -> McpClient:
    return McpClient(proxy=proxy)
