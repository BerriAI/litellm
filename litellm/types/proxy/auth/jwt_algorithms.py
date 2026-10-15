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

SymmetricJwtAlgorithm: TypeAlias = Literal["HS256", "HS384", "HS512"]

SYMMETRIC_JWT_ALGORITHMS: Final[tuple[SymmetricJwtAlgorithm, ...]] = ("HS256", "HS384", "HS512")

StoredJwtAlgorithm: TypeAlias = ApprovedJwtAlgorithm | LegacyJwtAlgorithm | SymmetricJwtAlgorithm
"""A value read back from stored/declared config: approved, or a not-approved value (legacy
EdDSA, symmetric HS*) carried under FIPS-gated read-side grace so a pre-allowlist row keeps
signing outside FIPS mode. Write-side validation and the verification-side allowlists stay
approved-only (plus EdDSA outside FIPS); nothing here widens them."""
