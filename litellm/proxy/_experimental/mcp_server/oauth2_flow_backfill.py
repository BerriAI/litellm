"""Startup backfill for oauth2 MCP server rows persisted before oauth2_flow was written.

Rows created before the write-side stamps (DCR persist, UI create, REST create) carry a
null ``oauth2_flow`` and rely on read-time field-shape inference, which cannot tell a
DCR-registered interactive server from an M2M server unless endpoint discovery succeeds
first. This backfill classifies each null row once, at rest, using signals inference
never had, and persists the result so the read path never has to infer again.

Signal order, strongest first:

1. Per-user OAuth token rows exist for the server: only the interactive flow mints
   per-user tokens, so this is definitive and immune to the discovery trap. BYOK API
   keys share the same table (``LiteLLM_MCPUserCredentials``), so only rows whose
   payload decodes as a ``type: oauth2`` token count as proof; bare keys and
   undecodable rows prove nothing about the flow.
2. ``authorization_url`` persisted: interactive needs a user-facing authorization
   endpoint; M2M (RFC 6749 section 4.4) never has one.
3. ``registration_url`` persisted: dynamic client registration (RFC 7591) exists to mint
   clients for the interactive flow; M2M servers are configured with static credentials.
4. ``token_url`` plus decryptable ``client_id`` and ``client_secret``: ambiguous, left
   unstamped. The shape is shared by M2M servers and DCR-registered interactive servers
   whose authorization endpoint lives only in discovery (registered but never signed
   in), so stamping client_credentials here could permanently route per-user traffic
   through the proxy's stored client credential. The row keeps working through the
   request-time backstop and a warning names it with the one-line fix (set oauth2_flow
   via the dashboard or ``PUT /v1/mcp/server``); a completed interactive sign-in also
   heals it via rule 1 at the next boot.
5. Anything else is interactive: matching how ``needs_user_oauth_token`` treats a null
   flow, so the stamp never changes runtime routing for rows no rule recognizes.

The backfill never stamps client_credentials: M2M is asserted by a human (config
requires it, the API accepts it, the dashboard sets it), mirroring the config-level
validation error. Runs before the first registry load on every boot and is idempotent:
a healed fleet has no null rows and the backfill exits after one query.
"""

import json
from collections import Counter
from typing import Any, Literal, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy._experimental.mcp_server.db import _decode_oauth_payload, decrypt_credentials
from litellm.proxy.utils import PrismaClient
from litellm.types.mcp import MCPCredentials

OAuth2Flow = Literal["client_credentials", "authorization_code"]
BackfillRule = Literal[
    "per_user_tokens",
    "authorization_url",
    "registration_url",
    "ambiguous_m2m_shape",
    "interactive_default",
]

_BACKFILL_AUDIT_ACTOR = "oauth2_flow_backfill"


def _decrypted_credentials(raw_credentials: Any) -> Optional[MCPCredentials]:
    if raw_credentials is None:
        return None
    if isinstance(raw_credentials, str):
        try:
            parsed = json.loads(raw_credentials)
        except (ValueError, TypeError):
            return None
    else:
        parsed = raw_credentials
    if not isinstance(parsed, dict):
        return None
    return decrypt_credentials(credentials=dict(parsed))


def classify_null_flow_row(
    *,
    has_per_user_tokens: bool,
    authorization_url: Optional[str],
    registration_url: Optional[str],
    token_url: Optional[str],
    credentials: Optional[MCPCredentials],
) -> tuple[Optional[OAuth2Flow], BackfillRule]:
    if has_per_user_tokens:
        return "authorization_code", "per_user_tokens"
    if authorization_url:
        return "authorization_code", "authorization_url"
    if registration_url:
        return "authorization_code", "registration_url"
    if token_url and credentials and credentials.get("client_id") and credentials.get("client_secret"):
        return None, "ambiguous_m2m_shape"
    return "authorization_code", "interactive_default"


async def backfill_null_oauth2_flows(prisma_client: PrismaClient) -> dict[BackfillRule, int]:
    """Classify every ``auth_type=oauth2`` row whose ``oauth2_flow`` is null; stamp the provable
    ones, warn on the ambiguous ones, and return counts per rule."""
    null_rows: list[Any] = await prisma_client.db.litellm_mcpservertable.find_many(
        where={"auth_type": "oauth2", "oauth2_flow": None},
    )
    if not null_rows:
        return {}

    server_ids = [row.server_id for row in null_rows]
    token_rows: list[Any] = await prisma_client.db.litellm_mcpusercredentials.find_many(
        where={"server_id": {"in": server_ids}},
    )
    server_ids_with_oauth_tokens: set[str] = {
        token_row.server_id for token_row in token_rows if _decode_oauth_payload(token_row.credential_b64) is not None
    }

    classified = tuple(
        (
            row,
            classify_null_flow_row(
                has_per_user_tokens=row.server_id in server_ids_with_oauth_tokens,
                authorization_url=row.authorization_url,
                registration_url=row.registration_url,
                token_url=row.token_url,
                credentials=_decrypted_credentials(row.credentials),
            ),
        )
        for row in null_rows
    )

    for row, (flow, rule) in classified:
        if flow is None:
            verbose_proxy_logger.warning(
                "oauth2_flow backfill: server_id=%s is ambiguous (client credentials + token_url, "
                "no interactive signal); left unstamped. Set oauth2_flow explicitly via the "
                "dashboard or PUT /v1/mcp/server: client_credentials if this server is M2M, or "
                "complete an interactive sign-in and it will be stamped authorization_code at the "
                "next boot.",
                row.server_id,
            )
        else:
            verbose_proxy_logger.info(
                "oauth2_flow backfill: server_id=%s stamped %s (rule=%s)",
                row.server_id,
                flow,
                rule,
            )

    stamped_flows = {flow for _, (flow, _) in classified if flow is not None}
    for stamped_flow in stamped_flows:
        server_ids_for_flow = [row.server_id for row, (row_flow, _) in classified if row_flow == stamped_flow]
        await prisma_client.db.litellm_mcpservertable.update_many(
            where={"server_id": {"in": server_ids_for_flow}, "oauth2_flow": None},
            data={"oauth2_flow": stamped_flow, "updated_by": _BACKFILL_AUDIT_ACTOR},
        )

    counts: dict[BackfillRule, int] = dict(Counter(rule for _, (_, rule) in classified))
    verbose_proxy_logger.info(
        "oauth2_flow backfill: processed %d oauth2 server row(s): %s",
        len(null_rows),
        counts,
    )
    return counts
