"""Mints a virtual key for a JWT claim, or reuses the one gateway-auth-core picked, and inserts the claim's
mapping row. gateway-auth-core has already decided whether to register and for whom, so this module only writes.
When two requests register one claim at once, the loser deletes the key it minted and returns the winner's mapping.

TODO: moves to Rust in one piece, together with key generation.

Blocked on:
- `generate_key_helper_fn`, shared with `/key/generate`, which Rust has no port of
  - validates router settings and model budgets, computes budget reset times
  - upserts the user row, checks the key's project and team
  - inserts the key and syncs its access groups

After the move:
- the mapping insert and orphan delete move with the mint, so the race cleanup never spans the boundary
- this module, `PythonAuthHost.create_key_mapping`, its `RustAuthStep` and its `answer.py` case are deleted"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.auth_checks import get_jwt_key_mapping_object
from litellm.proxy.db.db_span import db_span
from litellm.proxy.utils import hash_token
from litellm.repositories.table_repositories import JWTKeyMappingRepository
from litellm.repositories.verification_token_repository import VerificationTokenRepository

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


async def _mint_key(
    claim_field: str,
    claim_value: str,
    team_id: str | None,
    user_id: str | None,
    org_id: str | None,
    agent_id: str | None,
) -> str:
    from litellm.proxy.management_endpoints.key_management_endpoints import (  # noqa: PLC0415  # imports auth
        generate_key_helper_fn,
    )

    minted: Final = await generate_key_helper_fn(
        llm_router=None,
        request_type="key",
        table_name="key",
        team_id=team_id,
        user_id=user_id,
        organization_id=org_id,
        agent_id=agent_id,
        metadata={"auto_registered": True, "jwt_claim_field": claim_field, "jwt_claim_value": claim_value},
    )
    return hash_token(minted["token"])


async def create_key_mapping(
    prisma_client: PrismaClient,
    claim_field: str,
    claim_value: str,
    issuer: str | None,
    behavior: str,
    existing_hash: str | None,
    team_id: str | None,
    user_id: str | None,
    org_id: str | None,
    agent_id: str | None,
) -> tuple[str | None, bool]:
    """The mapped key's hash, and whether another request created the mapping first."""
    token_hash: Final = (
        existing_hash
        if existing_hash is not None
        else await _mint_key(claim_field, claim_value, team_id, user_id, org_id, agent_id)
    )
    try:
        async with db_span("auto_register_jwt_mapping", "LiteLLM_JWTKeyMapping"):
            await JWTKeyMappingRepository(prisma_client).table.create(
                data={
                    "jwt_issuer": issuer or "",
                    "jwt_claim_name": claim_field,
                    "jwt_claim_value": claim_value,
                    "token": token_hash,
                    "created_by": behavior,
                    "updated_by": behavior,
                }
            )
    except Exception as e:
        error: Final = str(e).lower()
        if "unique" not in error and "p2002" not in error:
            raise
        verbose_proxy_logger.debug(
            "JWT Key Mapping (%s): unique conflict on create, fetching winner's mapping for %s='%s'.",
            behavior,
            claim_field,
            claim_value,
        )
        if existing_hash is None:
            await _delete_orphan(prisma_client, token_hash)
        winner: Final = await get_jwt_key_mapping_object(
            jwt_claim_name=claim_field, jwt_claim_value=claim_value, prisma_client=prisma_client, jwt_issuer=issuer
        )
        return winner, True
    verbose_proxy_logger.info(
        "JWT Key Mapping (%s): %s virtual key for %s='%s'.",
        behavior,
        "created new" if existing_hash is None else "mapped existing",
        claim_field,
        claim_value,
    )
    return token_hash, False


async def _delete_orphan(prisma_client: PrismaClient, token_hash: str) -> None:
    try:
        async with db_span("delete_orphaned_jwt_key", "LiteLLM_VerificationToken"):
            await VerificationTokenRepository(prisma_client).table.delete(where={"token": token_hash})
    except Exception as delete_err:  # noqa: BLE001  # an unmapped orphan is inert, so a failed cleanup only logs
        verbose_proxy_logger.warning(
            "JWT Key Mapping (auto_register): failed to delete orphaned key after race: %s", delete_err
        )
