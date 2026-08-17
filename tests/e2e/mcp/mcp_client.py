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

import re
import time
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, RootModel

from e2e_config import settle_propagation
from e2e_http import Headers, NoBody, Result, Success, UnknownApiError, unwrap
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
    mcp_access_groups: list[str] | None = None


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
        mcp_access_groups: list[str] | None = None,
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
                    mcp_access_groups=mcp_access_groups,
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

    def await_registered(self, server_id: str) -> None:
        """Poll /v1/mcp/server until `server_id` is listed. Fails at poll_timeout.

        The DB row exists the moment registration returns, but a data-plane pod
        answers the listing from a registry it refreshes on a periodic DB sync, so a
        pod that joined the load balancer after the write reports the server as
        absent until its first sync.
        """
        deadline = time.monotonic() + self.proxy.poll_timeout
        while True:
            registered = frozenset(row.server_id for row in self.registered_servers())
            if server_id in registered:
                return
            if time.monotonic() >= deadline:
                raise AssertionError(
                    f"registered server {server_id} still absent from /v1/mcp/server "
                    f"{self.proxy.poll_timeout}s after registration (the data plane never synced "
                    f"the row): {registered}"
                )
            time.sleep(self.proxy.poll_interval)

    def generate_key(
        self,
        *,
        user_id: str,
        mcp_servers: list[str] | None,
        mcp_access_groups: list[str] | None = None,
        models: list[str] | None = None,
    ) -> str:
        object_permission = (
            ObjectPermission(mcp_servers=mcp_servers, mcp_access_groups=mcp_access_groups)
            if mcp_servers is not None or mcp_access_groups is not None
            else None
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

    def await_tool(self, key: str, server_id: str, needle: str) -> str:
        """Poll tools/list until `server_id` serves a tool matching `needle`, and
        return its fully-qualified name. Fails at poll_timeout.

        /v1/mcp/server returns as soon as the DB row is written, but the gateway
        runs the initialize + tools/list handshake against the upstream lazily on
        the first request that needs it, and reports a server it has not
        discovered yet exactly like a dead one: an empty tool list. Waiting is
        what separates the two.
        """
        deadline = time.monotonic() + self.proxy.poll_timeout
        while True:
            result = self.list_tools(key)
            if isinstance(result, Success):
                tool_name = result.data.tool_name_containing(server_id, needle)
                if tool_name is not None:
                    return tool_name
            if time.monotonic() >= deadline:
                raise AssertionError(
                    f"server {server_id} never served a tool matching {needle!r} within "
                    f"{self.proxy.poll_timeout}s of registration (upstream unreachable, or "
                    f"the key's grant was not applied); last tools/list: {result}"
                )
            time.sleep(self.proxy.poll_interval)

    def await_call_tool(
        self,
        key: str,
        *,
        server_id: str,
        name: str,
        arguments: McpToolArguments,
    ) -> McpCallToolResponse:
        """Poll tools/call until the result is not a multi-worker registry miss.

        Retries only on the gateway's own cold-worker 500 shapes (Tool <name>
        not found / server_not_found). Upstream tool errors and other 500s fail
        immediately so non-idempotent calls are not repeated.
        """
        deadline = time.monotonic() + self.proxy.poll_timeout
        last: Result[McpCallToolResponse] | None = None
        while True:
            last = self.call_tool(key, server_id=server_id, name=name, arguments=arguments)
            if not _is_mcp_not_synced(last, tool_name=name):
                return unwrap(last)
            if time.monotonic() >= deadline:
                raise AssertionError(
                    f"tools/call for {name!r} on server {server_id} still missing on the "
                    f"data plane after {self.proxy.poll_timeout}s (multi-worker registry lag); "
                    f"last result: {last}"
                )
            time.sleep(self.proxy.poll_interval)

    def await_call_tool_denied(
        self,
        key: str,
        *,
        server_id: str,
        name: str,
        arguments: McpToolArguments,
    ) -> UnknownApiError:
        """Poll tools/call until a cold-worker miss clears and the call is 403 access_denied."""
        deadline = time.monotonic() + self.proxy.poll_timeout
        last: Result[McpCallToolResponse] | None = None
        while True:
            last = self.call_tool(key, server_id=server_id, name=name, arguments=arguments)
            if isinstance(last, UnknownApiError) and last.status_code == 403:
                return last
            if not _is_mcp_not_synced(last, tool_name=name):
                raise AssertionError(
                    f"ungranted key's tools/call was not 403 access_denied: {last}"
                )
            if time.monotonic() >= deadline:
                raise AssertionError(
                    f"ungranted key never got 403 for {name!r} within {self.proxy.poll_timeout}s; "
                    f"last result: {last}"
                )
            time.sleep(self.proxy.poll_interval)

    def register_mcp_content_filter(self, *, name: str, blocked_keyword: str) -> str:
        """Register a default-on content-filter guardrail that runs on the MCP
        tool-call hook (pre_mcp_call) and blocks a single keyword. The keyword is
        unique per test, so default_on only ever intercepts this test's own
        banned tool call on the shared proxy."""
        guardrail_id = unwrap(
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
        settle_propagation(time.monotonic())
        return guardrail_id

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


def _is_mcp_not_synced(
    result: Result[McpCallToolResponse],
    *,
    tool_name: str | None = None,
) -> bool:
    """True only for gateway multi-worker registry misses, not upstream errors.

    Matches the proxy's own shapes:
    - ValueError ``Tool <name> not found`` wrapped as HTTP 500 (cold tool map /
      unresolved server on this process)
    - REST ``server_not_found`` when this worker has not loaded the MCP server row

    Does not treat arbitrary 500 bodies that merely mention "tool" and "not found"
    (e.g. upstream MCP payload text) as lag, so await_call_tool does not retry
    real failures or non-idempotent calls.
    """
    if not isinstance(result, UnknownApiError) or result.status_code != 500:
        return False
    body = result.body
    body_l = body.lower()

    if "server_not_found" in body_l:
        return True
    if re.search(r"mcp server ['\"][^'\"]+['\"] was not found", body_l):
        return True

    # Gateway: "Tool search_datadog_logs not found" (optionally inside a longer message)
    if tool_name is not None:
        return (
            re.search(rf"\btool\s+{re.escape(tool_name)}\s+not found\b", body_l) is not None
        )
    return re.search(r"\btool\s+\S+\s+not found\b", body_l) is not None


def build_client(proxy: ProxyClient) -> McpClient:
    return McpClient(proxy=proxy)
