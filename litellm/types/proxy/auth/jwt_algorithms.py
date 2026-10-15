from typing import Final, Literal, TypeAlias

ApprovedJwtAlgorithm: TypeAlias = Literal[
    "RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512"
]

APPROVED_JWT_ALGORITHMS: Final[tuple[ApprovedJwtAlgorithm, ...]] = (
    "RS256",
    "RS384",
    "RS512",
    "PS256",
    "PS384",
    "PS512",
    "ES256",
    "ES384",
    "ES512",
)

LegacyJwtAlgorithm: TypeAlias = Literal["EdDSA"]

LEGACY_JWT_ALGORITHMS: Final[tuple[LegacyJwtAlgorithm, ...]] = ("EdDSA",)

StoredJwtAlgorithm: TypeAlias = ApprovedJwtAlgorithm | LegacyJwtAlgorithm
"""A value read back from stored/declared config: approved, or legacy EdDSA carried under
read-side grace outside FIPS mode (write-side validation stays approved-only)."""
