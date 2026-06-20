"""v2-owned MCP egress manager (the cutover seam).

``MCPServerManagerV2`` subclasses v1's ``MCPServerManager`` and overrides the per-server egress
methods to route through the v2 ``UpstreamConnection`` + ``resolve()`` instead of
``_create_mcp_client``. Registry, RBAC, cross-server aggregation, namespacing
(``_create_prefixed_*``), and static-header resolution are inherited from v1 unchanged. It is the
egress manager, constructed at the composition root (see
``mcp_server_manager._make_global_mcp_server_manager``); there is no opt-in flag (v2 is the egress
implementation).

Every per-server op method shares two seams: ``_should_defer`` (the v1-vs-v2 gate, temporary
strangler scaffolding that shrinks to zero as modes migrate, then is deleted with
``_create_mcp_client``) and ``_v2_connection`` (the permanent resolve()+build seam that returns a
configured ``UpstreamConnection``). ``super()`` is reserved for what v2 does not own yet:
unmapped/misconfigured modes, OpenAPI tools (registry, S1.8), the per-request ``mcp_auth_header``
override/inbound-token path, and the JWT-signer guardrail.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, NoReturn, Optional, Union

from typing_extensions import assert_never

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
from litellm.proxy._types import MCPTransport

if TYPE_CHECKING:
    from mcp.shared.session import ProgressFnT
    from mcp.types import (
        CallToolResult,
        GetPromptResult,
        Prompt,
        ReadResourceResult,
        Resource,
    )
    from mcp.types import Tool as MCPTool
    from pydantic import AnyUrl

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.gateway.mcp.outbound_credentials.types import CredError
    from litellm.proxy.gateway.mcp.result import Result
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    from .v2_egress import ConnError, UpstreamConnection


class MCPServerManagerV2(MCPServerManager):
    """v2 egress manager; see the module docstring."""

    @staticmethod
    def _jwt_signer_configured() -> bool:
        from litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer import (
            get_mcp_jwt_signer,
        )

        return get_mcp_jwt_signer() is not None

    def _should_defer(
        self,
        server: MCPServer,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]],
    ) -> bool:
        """Whether this request falls back to v1 (super()) instead of the v2 egress path.

        True for what v2 does not own yet: unmapped/misconfigured modes (to_server_spec is None,
        e.g. bearer_token), OpenAPI servers (registry, not a connection), the per-request
        mcp_auth_header override/inbound-token path, and the JWT-signer guardrail. Temporary
        strangler scaffolding: returns True for fewer cases as modes migrate, then is deleted
        (with _create_mcp_client) once nothing is left on v1.
        """
        from litellm.proxy._experimental.mcp_server.v2_resolver_bridge import (
            to_server_spec,
        )

        return (
            to_server_spec(server) is None
            or bool(server.spec_path)
            or bool(mcp_auth_header)
            or self._jwt_signer_configured()
        )

    async def _v2_connection(
        self,
        server: MCPServer,
        user_api_key_auth: Optional[UserAPIKeyAuth],
        *,
        raw_headers: Optional[Dict[str, str]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        subject_token: Optional[str] = None,
        forward_caller_headers: bool = False,
        raise_on_missing_env: bool = False,
    ) -> Result[UpstreamConnection, CredError]:
        """The shared egress seam: resolve auth via resolve() and build the UpstreamConnection.

        Maps the v1 MCPServer to a v2 ServerSpec, resolves the credential for the Subject, merges
        static/env-var headers, and constructs the (not-yet-opened) connection. Returns the
        connection or the CredError from resolve(). Every per-server op method goes through here, so
        a mode migration or header change is a one-place change that lights up all ops. Callers must
        gate with _should_defer first (so to_server_spec is not None here).
        """
        from litellm.proxy._experimental.mcp_server.v2_egress import UpstreamConnection
        from litellm.proxy._experimental.mcp_server.v2_resolver_bridge import (
            provider,
            to_server_spec,
            to_subject,
        )
        from litellm.proxy.gateway.mcp.result import Error, Ok

        spec = to_server_spec(server)
        assert spec is not None  # guaranteed by _should_defer (spec is None -> v1)
        auth = await provider().resolve(
            to_subject(user_api_key_auth, subject_token), spec
        )
        if isinstance(auth, Error):
            return Error(auth.error)
        egress = await self._build_egress_headers(
            server,
            user_api_key_auth,
            raw_headers,
            forward_caller_headers=forward_caller_headers,
            raise_on_missing_env=raise_on_missing_env,
        )
        headers = {**(extra_headers or {}), **egress} or None
        is_stdio = server.transport == MCPTransport.stdio
        return Ok(
            UpstreamConnection(
                server.url,
                transport=server.transport,
                auth=auth.ok,
                extra_headers=headers,
                command=server.command,
                args=server.args,
                env=self._build_stdio_env(server, raw_headers) if is_stdio else None,
            )
        )

    async def _build_egress_headers(
        self,
        server: MCPServer,
        user_api_key_auth: Optional[UserAPIKeyAuth],
        raw_headers: Optional[Dict[str, str]],
        *,
        forward_caller_headers: bool,
        raise_on_missing_env: bool,
    ) -> Dict[str, str]:
        """The non-credential egress headers: configured static headers (with per-user env-var
        interpolation) plus, on the call path, the caller headers the server forwards upstream.

        The credential (Authorization) is resolve()'s, never this dict. raise_on_missing_env
        propagates MCPMissingUserEnvVarsError (412 + setup URL) on the call path; list ops degrade.
        forward_caller_headers gates caller-header forwarding (call only; list/prompts/resources do
        not forward, matching v1).
        """
        static = await self._resolve_static_headers_with_env_vars(
            server, user_api_key_auth, raise_on_missing=raise_on_missing_env
        )
        forwarded = (
            self._forwarded_request_headers(server, raw_headers)
            if forward_caller_headers
            else {}
        )
        return {**forwarded, **(static or {})}

    @staticmethod
    def _forwarded_request_headers(
        server: MCPServer,
        raw_headers: Optional[Dict[str, str]],
    ) -> Dict[str, str]:
        """Caller request headers the server is configured to forward (server.extra_headers pulled
        from raw_headers). Authorization is always stripped: the upstream credential is resolve()'s,
        so the forwarder never ships inbound auth upstream (the credential-isolation invariant; the
        passthrough/override paths that would forward an inbound token defer to v1).
        """
        if not server.extra_headers or not raw_headers:
            return {}
        normalized = {k.lower(): v for k, v in raw_headers.items()}
        return {
            header: normalized[header.lower()]
            for header in server.extra_headers
            if header.lower() != "authorization"
            and normalized.get(header.lower()) is not None
        }

    async def _get_tools_from_server(
        self,
        server: MCPServer,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        add_prefix: bool = True,
        raw_headers: Optional[Dict[str, str]] = None,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[MCPTool]:
        from litellm.proxy.gateway.mcp.result import Error

        if self._should_defer(server, mcp_auth_header):
            return await super()._get_tools_from_server(
                server,
                mcp_auth_header,
                extra_headers,
                add_prefix,
                raw_headers,
                user_api_key_auth,
            )
        conn = await self._v2_connection(
            server,
            user_api_key_auth,
            raw_headers=raw_headers,
            extra_headers=extra_headers,
        )
        if isinstance(conn, Error):
            self._egress_list_failure(server, conn.error)
            return []
        result = await conn.ok.list_tools()
        if isinstance(result, Error):
            self._egress_list_failure(server, result.error)
            return []
        return self._create_prefixed_tools(result.ok, server, add_prefix=add_prefix)

    async def get_prompts_from_server(
        self,
        server: MCPServer,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        add_prefix: bool = True,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> List[Prompt]:
        from litellm.proxy.gateway.mcp.result import Error

        if self._should_defer(server, mcp_auth_header):
            return await super().get_prompts_from_server(
                server, mcp_auth_header, extra_headers, add_prefix, raw_headers
            )
        conn = await self._v2_connection(
            server, None, raw_headers=raw_headers, extra_headers=extra_headers
        )
        if isinstance(conn, Error):
            self._egress_list_failure(server, conn.error)
            return []
        result = await conn.ok.list_prompts()
        if isinstance(result, Error):
            self._egress_list_failure(server, result.error)
            return []
        return self._create_prefixed_prompts(result.ok, server, add_prefix=add_prefix)

    async def get_resources_from_server(
        self,
        server: MCPServer,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        add_prefix: bool = True,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> List[Resource]:
        from litellm.proxy.gateway.mcp.result import Error

        if self._should_defer(server, mcp_auth_header):
            return await super().get_resources_from_server(
                server, mcp_auth_header, extra_headers, add_prefix, raw_headers
            )
        conn = await self._v2_connection(
            server, None, raw_headers=raw_headers, extra_headers=extra_headers
        )
        if isinstance(conn, Error):
            self._egress_list_failure(server, conn.error)
            return []
        result = await conn.ok.list_resources()
        if isinstance(result, Error):
            self._egress_list_failure(server, result.error)
            return []
        return self._create_prefixed_resources(result.ok, server, add_prefix=add_prefix)

    async def read_resource_from_server(
        self,
        server: MCPServer,
        url: AnyUrl,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> ReadResourceResult:
        from litellm.proxy.gateway.mcp.result import Error

        if self._should_defer(server, mcp_auth_header):
            return await super().read_resource_from_server(
                server, url, mcp_auth_header, extra_headers, raw_headers
            )
        conn = await self._v2_connection(
            server,
            None,
            raw_headers=raw_headers,
            extra_headers=extra_headers,
            raise_on_missing_env=True,
        )
        if isinstance(conn, Error):
            self._egress_item_failure(server, conn.error)
        result = await conn.ok.read_resource(url)
        if isinstance(result, Error):
            self._egress_item_failure(server, result.error)
        return result.ok

    async def get_prompt_from_server(
        self,
        server: MCPServer,
        prompt_name: str,
        arguments: Optional[Dict[str, object]] = None,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> GetPromptResult:
        from litellm.proxy.gateway.mcp.result import Error

        if self._should_defer(server, mcp_auth_header):
            return await super().get_prompt_from_server(
                server,
                prompt_name,
                arguments,
                mcp_auth_header,
                extra_headers,
                raw_headers,
            )
        conn = await self._v2_connection(
            server,
            None,
            raw_headers=raw_headers,
            extra_headers=extra_headers,
            raise_on_missing_env=True,
        )
        if isinstance(conn, Error):
            self._egress_item_failure(server, conn.error)
        # MCP prompt arguments are strings on the wire; coerce the loose dict to match the op's type.
        str_args = {k: str(v) for k, v in arguments.items()} if arguments else None
        result = await conn.ok.get_prompt(prompt_name, str_args)
        if isinstance(result, Error):
            self._egress_item_failure(server, result.error)
        return result.ok

    async def _open_and_call_tool(
        self,
        mcp_server: MCPServer,
        original_tool_name: str,
        arguments: Dict[str, object],
        *,
        mcp_auth_header: Optional[str],
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]],
        oauth2_headers: Optional[Dict[str, str]],
        raw_headers: Optional[Dict[str, str]],
        hook_extra_headers: Optional[Dict[str, str]],
        host_progress_callback: Optional[ProgressFnT],
        user_api_key_auth: Optional[UserAPIKeyAuth],
    ) -> CallToolResult:
        from litellm.proxy.gateway.mcp.result import Error

        # The effective inbound credential is the per-server header if present, else the deprecated
        # mcp_auth_header (matches v1); an override here means the request defers to v1.
        server_auth_header: Optional[Union[str, Dict[str, str]]] = mcp_auth_header
        if mcp_server_auth_headers:
            from litellm.proxy._experimental.mcp_server.utils import (
                lookup_mcp_server_auth_in_headers,
            )

            found = lookup_mcp_server_auth_in_headers(
                mcp_server_auth_headers,
                alias=mcp_server.alias,
                server_name=mcp_server.server_name,
            )
            if found is not None:
                server_auth_header = found

        # Defer to v1 for guardrail-injected headers (JWT signer etc.), which the v2 path does not
        # apply, in addition to the usual _should_defer cases (unmapped mode, OpenAPI, override).
        if hook_extra_headers or self._should_defer(mcp_server, server_auth_header):
            return await super()._open_and_call_tool(
                mcp_server,
                original_tool_name,
                arguments,
                mcp_auth_header=mcp_auth_header,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
                hook_extra_headers=hook_extra_headers,
                host_progress_callback=host_progress_callback,
                user_api_key_auth=user_api_key_auth,
            )
        conn = await self._v2_connection(
            mcp_server,
            user_api_key_auth,
            raw_headers=raw_headers,
            forward_caller_headers=True,
            raise_on_missing_env=True,
        )
        if isinstance(conn, Error):
            self._egress_item_failure(mcp_server, conn.error)
        result = await conn.ok.call_tool(
            original_tool_name, arguments, host_progress_callback
        )
        if isinstance(result, Error):
            self._egress_item_failure(mcp_server, result.error)
        return result.ok

    def _egress_list_failure(
        self, server: MCPServer, error: "CredError | ConnError"
    ) -> None:
        # List path (tools/prompts/resources): an upstream 401/403 (or a per-user mode with no usable
        # credential) surfaces as MCPUpstreamAuthError so the client gets a 401 + WWW-Authenticate and
        # starts the OAuth flow (the LIT-3795 behavior for non-delegated interactive oauth2). Any
        # other failure is logged and the caller degrades to an empty list, so one bad server never
        # collapses the federated catalog. The typed partial-failure marker is a later surface.
        if error.tag == "unauthorized":
            from litellm.proxy._experimental.mcp_server.exceptions import (
                MCPUpstreamAuthError,
            )

            raise MCPUpstreamAuthError(
                status_code=401, www_authenticate=None, server_name=server.name
            )
        verbose_logger.warning(
            "v2 egress: items unavailable for %s (%s): %s",
            server.name,
            server.server_id,
            error.summary,
        )

    def _egress_item_failure(
        self, server: MCPServer, error: "CredError | ConnError"
    ) -> NoReturn:
        # Single-result path (call_tool / read_resource / get_prompt): no list to degrade to, so
        # every failure raises. unauthorized -> MCPUpstreamAuthError (401 + re-auth); the rest map to
        # a semantically correct MCPUpstreamError. assert_never keeps the map total -- a new error
        # variant fails the build here until it is given a mapping.
        from litellm.proxy._experimental.mcp_server.exceptions import (
            MCPUpstreamAuthError,
            MCPUpstreamError,
        )

        match error.tag:
            case "unauthorized":
                raise MCPUpstreamAuthError(
                    status_code=401, www_authenticate=None, server_name=server.name
                )
            case "upstream_unavailable":
                raise MCPUpstreamError(502, server.name, error.summary)
            case "precondition_required":
                raise MCPUpstreamError(428, server.name, error.summary)
            case "misconfigured" | "unsupported_mode" | "not_implemented":
                raise MCPUpstreamError(500, server.name, error.summary)
        assert_never(error.tag)
