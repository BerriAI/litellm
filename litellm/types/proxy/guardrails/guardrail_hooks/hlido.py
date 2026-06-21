from typing import List, Optional

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class HlidoGuardrailConfigModelOptionalParams(BaseModel):
    min_score: Optional[int] = Field(
        default=None,
        description=(
            "Minimum Hlido trust score (0-100) an agent must have. "
            "Defaults to 60 when neither min_score nor allowed_tiers is set."
        ),
    )
    allowed_tiers: Optional[List[str]] = Field(
        default=None,
        description=(
            "Hlido tiers the agent must be in, e.g. ['VITAL', 'STRONG']. "
            "When set, agents outside these tiers are blocked."
        ),
    )
    slugs: Optional[List[str]] = Field(
        default=None,
        description=(
            "Hlido agent slugs to verify on every request. Merged with the "
            "per-request 'hlido_slugs' list in request metadata."
        ),
    )
    on_unverified: Optional[str] = Field(
        default=None,
        description=(
            "Action when a slug has no Hlido review: 'allow' (default) or 'block'."
        ),
    )
    on_error: Optional[str] = Field(
        default=None,
        description=(
            "Action when the Hlido API is unreachable: 'allow' (default) or 'block'."
        ),
    )
    cache_ttl: Optional[float] = Field(
        default=None,
        description="Seconds to cache trust lookups per slug. Defaults to 300.",
    )
    trust_request_slugs: Optional[bool] = Field(
        default=None,
        description=(
            "When true, the per-request 'hlido_slugs' list in request metadata "
            "is also checked. Defaults to false so callers cannot choose their "
            "own trust subject; only server-configured 'slugs' are verified."
        ),
    )
    max_request_slugs: Optional[int] = Field(
        default=None,
        description=(
            "Maximum number of caller-supplied slugs to verify per request "
            "(only when trust_request_slugs is true). Defaults to 20."
        ),
    )
    request_timeout: Optional[float] = Field(
        default=None,
        description=(
            "Total timeout in seconds for each Hlido API call. Defaults to 10."
        ),
    )
    max_cache_entries: Optional[int] = Field(
        default=None,
        description=(
            "Maximum number of slugs held in the in-memory trust cache before "
            "eviction. Defaults to 1024."
        ),
    )


class HlidoGuardrailConfigModel(
    GuardrailConfigModel[HlidoGuardrailConfigModelOptionalParams]
):
    api_base: Optional[str] = Field(
        default=None,
        description=(
            "Hlido API base URL. Falls back to the HLIDO_API_BASE environment "
            "variable, defaults to https://hlido.eu"
        ),
    )
    api_key: Optional[str] = Field(
        default=None,
        description=(
            "Optional Hlido API key (hlk_live_*) for higher rate limits. The "
            "free tier needs no key. Falls back to the HLIDO_API_KEY "
            "environment variable."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Hlido Agent Trust"
