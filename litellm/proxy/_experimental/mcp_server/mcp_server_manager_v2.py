"""v2-owned MCP egress manager (the cutover seam).

``MCPServerManagerV2`` subclasses v1's ``MCPServerManager`` and overrides the per-server egress
methods to route through the v2 ``UpstreamConnection`` + ``resolve()`` instead of
``_create_mcp_client``. Registry, RBAC, cross-server aggregation, namespacing
(``_create_prefixed_tools``), and static-header resolution are inherited from v1 unchanged. It is
the egress manager, constructed at the composition root (see
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

from typing import TYPE_CHECKING, Dict, List, Optional, Union

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
from litellm.proxy._types import MCPTransport

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool

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
        resolved_static = await self._resolve_static_headers_with_env_vars(
            server, user_api_key_auth, raise_on_missing=False
        )
        headers = {**(extra_headers or {}), **(resolved_static or {})} or None
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
            return self._egress_list_failure(server, conn.error)
        result = await conn.ok.list_tools()
        if isinstance(result, Error):
            return self._egress_list_failure(server, result.error)
        return self._create_prefixed_tools(result.ok, server, add_prefix=add_prefix)

    def _egress_list_failure(
        self, server: MCPServer, error: "CredError | ConnError"
    ) -> List[MCPTool]:
        # List path: an upstream 401/403 (or a per-user mode with no usable credential) surfaces as
        # MCPUpstreamAuthError so the client gets a 401 + WWW-Authenticate and starts the OAuth flow
        # (this is the LIT-3795 behavior for non-delegated interactive oauth2). Any other failure
        # degrades to an empty tool list (logged), so one bad server never collapses the federated
        # catalog. The typed partial-failure marker is a later, separate surface.
        if error.tag == "unauthorized":
            from litellm.proxy._experimental.mcp_server.exceptions import (
                MCPUpstreamAuthError,
            )

            raise MCPUpstreamAuthError(
                status_code=401, www_authenticate=None, server_name=server.name
            )
        verbose_logger.warning(
            "v2 egress: tools unavailable for %s (%s): %s",
            server.name,
            server.server_id,
            error.summary,
        )
        return []
