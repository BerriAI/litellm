from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, NamedTuple

from litellm.repositories import rust_database
from litellm.repositories.table_repositories import PrismaTableRepository
from litellm.rust_bridge._native import RustJwtKeyMappingRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models  # noqa: F401  # used by quoted base-class subscripts


class JWTClaimScope(NamedTuple):
    """The issuer (empty for the global scope), claim name and claim value a mapping row is stored under."""

    jwt_issuer: str
    jwt_claim_name: str
    jwt_claim_value: str


class InsertedMapping(NamedTuple):
    """The hashed key now mapped to the claim, None when the mapping holding it is inactive or gone, and whether
    this insert wrote the mapping."""

    token: str | None
    created: bool


class JWTKeyMappingRepository(PrismaTableRepository["prisma_models.LiteLLM_JWTKeyMapping"]):
    table_name = "litellm_jwtkeymapping"

    async def mapped_token(self, jwt_claim_name: str, jwt_claim_value: str, jwt_issuer: str | None) -> str | None:
        """The hashed key an active mapping gives this claim in exactly ``jwt_issuer``'s scope (global when falsy)."""
        return await rust_database.required_repository(RustJwtKeyMappingRepository).mapped_token(
            jwt_claim_name, jwt_claim_value, jwt_issuer or None
        )

    async def mapped_claims(self, hashed_tokens: Sequence[str]) -> tuple[JWTClaimScope, ...]:
        """The claim scopes mapped to any of ``hashed_tokens``."""
        claims: Final = await rust_database.required_repository(RustJwtKeyMappingRepository).mapped_claims(
            list(hashed_tokens)
        )
        return tuple(JWTClaimScope(*claim) for claim in claims)

    async def insert(
        self, jwt_claim_name: str, jwt_claim_value: str, token: str, *, jwt_issuer: str | None, written_by: str
    ) -> InsertedMapping:
        """Maps the claim to ``token`` in exactly ``jwt_issuer``'s scope (global when falsy), leaving a mapping
        that already holds the claim as it is."""
        created, mapped = await rust_database.required_repository(RustJwtKeyMappingRepository).insert(
            jwt_claim_name, jwt_claim_value, token, jwt_issuer=jwt_issuer or None, written_by=written_by
        )
        return InsertedMapping(mapped, created)
