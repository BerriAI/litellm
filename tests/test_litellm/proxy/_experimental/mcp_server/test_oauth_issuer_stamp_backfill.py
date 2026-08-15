"""Tests for the one-time heal of issuer values a released version's discovery write-back stamped."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy._experimental.mcp_server.oauth_issuer_stamp_backfill import (
    backfill_discovery_stamped_issuers,
)


def _row(**overrides):
    fields = {
        "server_id": "srv-1",
        "alias": "srv_one",
        "server_name": "srv_one",
        "auth_type": "oauth2",
        "issuer": "https://idp.example.com",
        "authorization_url": "https://idp.example.com/authorize",
        "token_url": "https://idp.example.com/token",
        "registration_url": None,
        "updated_by": "mcp_oauth_discovery",
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _prisma(rows):
    prisma_client = MagicMock()
    prisma_client.db.litellm_mcpservertable.find_many = AsyncMock(return_value=rows)
    prisma_client.db.litellm_mcpservertable.update = AsyncMock()
    return prisma_client


@pytest.mark.asyncio
async def test_clears_the_stamp_and_records_its_own_actor():
    """The GH #34985 row: discovery wrote the issuer, so the server reads as issuer-anchored and its
    configured endpoints are ignored. Clearing the stamp makes them apply again. The heal records its
    own actor, which is also what makes it idempotent: the row no longer matches the discovery-actor
    filter, so it is never reconsidered on a later boot."""
    prisma_client = _prisma([_row()])

    assert await backfill_discovery_stamped_issuers(prisma_client) == 1

    call = prisma_client.db.litellm_mcpservertable.update.call_args
    assert call.kwargs["where"] == {"server_id": "srv-1"}
    assert call.kwargs["data"]["issuer"] is None
    assert call.kwargs["data"]["updated_by"] == "mcp_oauth_issuer_stamp_backfill"

    where = prisma_client.db.litellm_mcpservertable.find_many.call_args.kwargs["where"]
    assert where["updated_by"] == "mcp_oauth_discovery"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"updated_by": "some-admin@example.com"}, "an admin was the last writer, so the pin is theirs"),
        ({"issuer": None}, "nothing to heal"),
        ({"issuer": "   "}, "blank issuer is not a pin"),
        (
            {"authorization_url": None, "token_url": None, "registration_url": None},
            "issuer set with no configured endpoints is the canonical shape of a deliberate pin, and "
            "there is nothing configured for anchoring to discard anyway",
        ),
        (
            {"authorization_url": "https://other-idp.example.com/authorize", "token_url": None},
            "endpoints addressing a different authority than the issuer are an intent a clear would "
            "discard, so the row is warned about rather than healed",
        ),
        (
            {"issuer": "https://pinned.example.com"},
            "same shape from the other side: a pinned issuer whose origin differs from the configured "
            "endpoints cannot have been derived from them by discovery",
        ),
    ],
)
async def test_leaves_rows_alone_that_do_not_carry_the_defect_signature(overrides, reason):
    """updated_by records only the most recent writer and no audit trail says which field it touched,
    so the heal is deliberately narrow: it fires only on the full signature of the defect. Every
    exclusion here protects a row whose issuer may be a deliberate admin pin."""
    prisma_client = _prisma([_row(**overrides)])

    assert await backfill_discovery_stamped_issuers(prisma_client) == 0, reason
    prisma_client.db.litellm_mcpservertable.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_heals_across_url_forms_that_denote_the_same_origin():
    """Origin comparison runs through the shared canonicalizer, so a default port or host casing
    difference between the stamped issuer and the endpoints an admin typed does not make a #34985 row
    look like a deliberate pin at a different authority."""
    prisma_client = _prisma(
        [
            _row(
                issuer="https://IDP.example.com:443",
                authorization_url="https://idp.example.com/authorize",
                token_url="https://idp.example.com/token",
            )
        ]
    )

    assert await backfill_discovery_stamped_issuers(prisma_client) == 1


@pytest.mark.asyncio
async def test_query_is_scoped_to_auth_types_where_an_issuer_anchors():
    """Only the discovery auth types read an issuer as a trust anchor; clearing it elsewhere would be
    an unrelated mutation."""
    prisma_client = _prisma([])

    await backfill_discovery_stamped_issuers(prisma_client)

    where = prisma_client.db.litellm_mcpservertable.find_many.call_args.kwargs["where"]
    assert set(where["auth_type"]["in"]) == {"oauth2", "true_passthrough", "oauth_delegate"}


@pytest.mark.asyncio
async def test_a_failed_row_does_not_abort_the_rest():
    """Per-row best effort: one write failure must not leave later rows unhealed, and the next boot
    retries the failed one since its updated_by is unchanged."""
    prisma_client = _prisma([_row(server_id="bad"), _row(server_id="good")])
    prisma_client.db.litellm_mcpservertable.update = AsyncMock(
        side_effect=[Exception("write failed"), MagicMock()]
    )

    assert await backfill_discovery_stamped_issuers(prisma_client) == 1
    assert prisma_client.db.litellm_mcpservertable.update.await_count == 2
