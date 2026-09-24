from typing import Final, Literal

ApprovedJwtAlgorithm = Literal["RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512"]

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

LEGACY_JWT_ALGORITHMS: Final = ("EdDSA",)
