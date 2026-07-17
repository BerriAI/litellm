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

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, RootModel

from e2e_gateway import Gateway, build_gateway
from e2e_http import Headers, NoBody, Result, unwrap
from models import KeyGenerateBody, ObjectPermission


class ApiKeyHeaders(Headers):
    x_litellm_api_key: str = Field(serialization_alias="x-litellm-api-key")


class McpServerNewBody(BaseModel):
    server_name: str
    alias: str
    url: str
    transport: str = "http"


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


class McpCallToolBody(BaseModel):
    name: str
    arguments: dict[str, int]
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


@dataclass(frozen=True, slots=True)
class McpClient:
    gateway: Gateway

    def register_server(self, *, server_name: str, alias: str, url: str) -> str:
        return unwrap(
            self.gateway.transport.post(
                "/v1/mcp/server",
                headers=self.gateway.transport.master,
                json=McpServerNewBody(server_name=server_name, alias=alias, url=url),
                response_type=McpServerNewResponse,
            )
        ).server_id

    def delete_server(self, server_id: str) -> None:
        _ = self.gateway.transport.delete(
            f"/v1/mcp/server/{server_id}",
            headers=self.gateway.transport.master,
            json=NoBody(),
            response_type=NoBody,
        )

    def registered_servers(self) -> list[McpServerRow]:
        return unwrap(
            self.gateway.transport.get(
                "/v1/mcp/server",
                headers=self.gateway.transport.master,
                params=NoBody(),
                response_type=McpServersListResponse,
            )
        ).root

    def generate_key(self, *, user_id: str, mcp_servers: list[str] | None) -> str:
        object_permission = (
            ObjectPermission(mcp_servers=mcp_servers) if mcp_servers is not None else None
        )
        return self.gateway.generate_key(
            KeyGenerateBody(models=[], user_id=user_id, object_permission=object_permission)
        )

    def list_tools(self, key: str) -> Result[McpToolsListResponse]:
        return self.gateway.transport.get(
            "/mcp-rest/tools/list",
            headers=ApiKeyHeaders(x_litellm_api_key=key),
            params=NoBody(),
            response_type=McpToolsListResponse,
        )

    def call_tool(
        self, key: str, *, server_id: str, name: str, arguments: dict[str, int]
    ) -> Result[McpCallToolResponse]:
        return self.gateway.transport.post(
            "/mcp-rest/tools/call",
            headers=ApiKeyHeaders(x_litellm_api_key=key),
            json=McpCallToolBody(name=name, arguments=arguments, server_id=server_id),
            response_type=McpCallToolResponse,
        )


def build_client() -> McpClient:
    return McpClient(gateway=build_gateway())
