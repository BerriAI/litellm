"""One-time heal for MCP server rows whose ``issuer`` a released version wrote by itself.

Until the write was removed, OAuth discovery stamped the issuer it discovered onto the ``issuer``
column trust-on-first-use. That column means "the admin pinned this trust anchor", so the next
registry build read the gateway's own output back as admin intent: the server turned issuer-anchored
(RFC 8414 section 3.3), its stored authorization/token/registration URLs stopped applying, and a
failed issuer-document fetch left it with no authorize endpoint (GH #34985).

Deleting the write fixes every row created afterwards but cannot fix a row already stamped, which
still reads as pinned. This heals those rows by clearing the stamp so their configured endpoints
apply again.

The signal is a heuristic, and deliberately a narrow one. ``updated_by`` records only the most recent
writer, and no audit trail says which field that writer touched, so "discovery wrote this issuer" is
not directly knowable. Two independent clauses bound it, and each rules out a different way of
destroying a pin an admin meant.

Configured endpoints must be present. A deliberately pinned row very often has none, both because the
Issuer field is documented as overriding them and because ``update_mcp_server`` clears them when an
issuer changes, so "issuer set, endpoints empty" is the canonical shape of a real pin and must never
be cleared on this evidence. Skipping those rows costs little: with nothing configured to restore, the
anchored and resource-rooted paths resolve from the same upstream document, and the row still gets the
unresolved-endpoint retry and the anchored-discard warning.

The configured endpoints must also share the issuer's origin. A stamped issuer is by construction the
one self-attested by the authorization-server document discovery reached from this very server, so
endpoints typed alongside it address that same authority. An admin who pinned an issuer and typed
endpoints for a different authority is expressing an intent that clearing the issuer would discard, so
that row is warned about and never healed.

What survives both clauses is a row whose configured endpoints and stamped issuer share an origin,
which is exactly the GH #34985 shape. An admin who pinned that same origin by hand lands here too, and
for them the clear is close to a no-op: their typed endpoints keep serving and still anchor the
RFC 9700 corroboration gate, with only the stricter section 3.3 anchoring lost. Every heal logs the
cleared value so it can be restored, and the clear is recorded under this module's actor so the heal
runs at most once per row.
"""

from typing import Final, Protocol
from urllib.parse import urlparse

from litellm._logging import verbose_proxy_logger
from litellm.proxy._experimental.mcp_server.oauth_utils import canonicalize_url_identity
from litellm.proxy.utils import PrismaClient

# The actor the removed discovery write-back stamped rows with.
_DISCOVERY_ACTOR: Final = "mcp_oauth_discovery"

# The actor recorded on a healed row, which also makes the heal idempotent: once a row is cleared it
# no longer matches ``updated_by == _DISCOVERY_ACTOR`` and is never reconsidered.
_BACKFILL_ACTOR: Final = "mcp_oauth_issuer_stamp_backfill"

_AUTH_TYPES_WITH_ISSUER_ANCHORING: Final = ("oauth2", "true_passthrough", "oauth_delegate")


def _origin(url: str) -> str | None:
    """The scheme-and-authority identity of ``url``, or ``None`` when it has none.

    Built on the shared URL canonicalizer so the lowercase-host and default-port rules match the
    RFC 8414 issuer comparison the resolution path uses, instead of being re-derived here.
    """
    parsed: Final = urlparse(canonicalize_url_identity(url))
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


class _MCPServerRow(Protocol):
    """The MCP server row fields this heal reads, so the untyped DB record is narrowed once here."""

    server_id: str
    alias: str | None
    server_name: str | None
    auth_type: str | None
    issuer: str | None
    authorization_url: str | None
    token_url: str | None
    registration_url: str | None
    updated_by: str | None


def _is_stamped_issuer_row(row: _MCPServerRow) -> bool:
    """Whether this row carries the full signature of a gateway-written issuer stamp.

    The whole rule lives here, including the writer check the query also filters on, so the decision
    to clear an admin-visible field is auditable in one place rather than split between a predicate
    and a query.
    """
    if getattr(row, "updated_by", None) != _DISCOVERY_ACTOR:
        return False
    if not (getattr(row, "issuer", None) or "").strip():
        return False
    if getattr(row, "auth_type", None) not in _AUTH_TYPES_WITH_ISSUER_ANCHORING:
        return False
    configured: Final = tuple(
        value.strip()
        for value in (row.authorization_url, row.token_url, row.registration_url)
        if value and value.strip()
    )
    if not configured:
        return False
    issuer_origin: Final = _origin(row.issuer or "")
    return issuer_origin is not None and all(_origin(endpoint) == issuer_origin for endpoint in configured)


async def backfill_discovery_stamped_issuers(prisma_client: PrismaClient) -> int:
    """Clear gateway-written issuer stamps, returning the number of rows healed."""
    candidate_rows: Final[list[_MCPServerRow]] = await prisma_client.db.litellm_mcpservertable.find_many(
        where={
            "updated_by": _DISCOVERY_ACTOR,
            "auth_type": {"in": list(_AUTH_TYPES_WITH_ISSUER_ANCHORING)},
        },
    )
    stamped: Final = tuple(row for row in candidate_rows if _is_stamped_issuer_row(row))
    if not stamped:
        return 0

    healed = 0
    for row in stamped:
        try:
            await prisma_client.db.litellm_mcpservertable.update(
                where={"server_id": row.server_id},
                data={"issuer": None, "updated_by": _BACKFILL_ACTOR},
            )
        except Exception as exc:  # noqa: BLE001 - per-row best effort; the next boot retries
            verbose_proxy_logger.warning(
                "MCP issuer stamp backfill: could not heal server_id=%s: %s", row.server_id, exc
            )
            continue
        healed += 1
        verbose_proxy_logger.warning(
            "MCP issuer stamp backfill: cleared issuer %r on server_id=%s (alias=%s). OAuth discovery "
            "had written that value onto the Issuer column, which made the server issuer-anchored and "
            "fail-closed, and its configured Authorization/Token/Registration URLs were being ignored "
            "as a result; those now apply again. If you pinned this issuer deliberately, set it again "
            "via the dashboard or PUT /v1/mcp/server to restore RFC 8414 section 3.3 anchoring.",
            row.issuer,
            row.server_id,
            row.alias or row.server_name,
        )

    if healed:
        verbose_proxy_logger.warning(
            "MCP issuer stamp backfill: healed %d server(s) whose Issuer had been written by OAuth "
            "discovery rather than by an admin",
            healed,
        )
    return healed
