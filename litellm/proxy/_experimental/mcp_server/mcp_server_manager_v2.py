"""v2-owned MCP egress manager (the cutover seam).

``MCPServerManagerV2`` subclasses v1's ``MCPServerManager`` and overrides the per-server egress
methods to route through the v2 ``UpstreamConnection`` + ``resolve()`` instead of
``_create_mcp_client``. Registry, RBAC, cross-server aggregation, namespacing
(``_create_prefixed_tools``), and static-header resolution are inherited from v1 unchanged. It is
the egress manager, constructed at the composition root (see
``mcp_server_manager._make_global_mcp_server_manager``); there is no opt-in flag (v2 is the egress
implementation).

Migration is the override progression: each egress mode is wired through ``resolve()`` +
``UpstreamConnection``, live-validated, and committed one at a time. Modes that are not yet wired
(passthrough / token_exchange, which need the caller's inbound token) simply fail closed until their
commit. ``super()`` is reserved for non-egress concerns that migrate as their own subsystems:
OpenAPI tools (registry, S1.8), the per-request ``mcp_auth_header`` override/inbound-token path, and
the JWT-signer guardrail.
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
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    from .v2_egress import ConnError


class MCPServerManagerV2(MCPServerManager):
    """v2 egress manager; see the module docstring."""

    @staticmethod
    def _jwt_signer_configured() -> bool:
        from litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer import (
            get_mcp_jwt_signer,
        )

        return get_mcp_jwt_signer() is not None

    async def _get_tools_from_server(
        self,
        server: MCPServer,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        add_prefix: bool = True,
        raw_headers: Optional[Dict[str, str]] = None,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[MCPTool]:
        from litellm.proxy._experimental.mcp_server.v2_egress import UpstreamConnection
        from litellm.proxy._experimental.mcp_server.v2_resolver_bridge import (
            provider,
            to_server_spec,
            to_subject,
        )
        from litellm.proxy.gateway.mcp.result import Error

        spec = to_server_spec(server)
        # Stay on v1 for what hasn't migrated to the egress transport: unmapped/not-yet-wired modes
        # (spec is None, e.g. aws_sigv4), OpenAPI tools (registry, S1.8), the per-request
        # mcp_auth_header override/inbound-token path (migrated with passthrough/token_exchange), and
        # the JWT-signer guardrail.
        if (
            spec is None
            or server.spec_path
            or mcp_auth_header
            or self._jwt_signer_configured()
        ):
            return await super()._get_tools_from_server(
                server,
                mcp_auth_header,
                extra_headers,
                add_prefix,
                raw_headers,
                user_api_key_auth,
            )

        auth = await provider().resolve(to_subject(user_api_key_auth, None), spec)
        if isinstance(auth, Error):
            return self._egress_list_failure(server, auth.error)

        resolved_static = await self._resolve_static_headers_with_env_vars(
            server, user_api_key_auth, raise_on_missing=False
        )
        headers = {**(extra_headers or {}), **(resolved_static or {})} or None
        is_stdio = server.transport == MCPTransport.stdio

        result = await UpstreamConnection(
            server.url,
            transport=server.transport,
            auth=auth.ok,
            extra_headers=headers,
            command=server.command,
            args=server.args,
            env=self._build_stdio_env(server, raw_headers) if is_stdio else None,
        ).list_tools()
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
