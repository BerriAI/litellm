"""Typed configuration for governor v2.

Free of any redis/prisma import so proxy core can read the feature flag before
heavyweight settings load. ``is_governor_v2_enabled`` is the single gate the
rest of LiteLLM consults; everything under this package is inert until it
returns True.
"""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

GOVERNOR_V2_ENV = "LITELLM_GOVERNOR_V2"


class _GovernorV2Flag(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = Field(default=False, validation_alias=AliasChoices(GOVERNOR_V2_ENV))


def is_governor_v2_enabled() -> bool:
    return _GovernorV2Flag().enabled


class GovernorV2Config(BaseSettings):
    model_config = SettingsConfigDict(
        populate_by_name=True, extra="ignore", env_prefix="LITELLM_GOVERNOR_V2_"
    )

    enabled: bool = Field(default=False, validation_alias=AliasChoices(GOVERNOR_V2_ENV))

    l1_max_entries: int = Field(default=16_384)
    l1_staleness_ms_rates: int = Field(default=250)
    l1_staleness_ms_budgets: int = Field(default=1_000)

    rate_counter_ttl_seconds: int = Field(default=120)
    budget_reset_grace_seconds: int = Field(default=3_600)
    reconcile_dedup_ttl_seconds: int = Field(default=3_600)

    flush_interval_seconds: int = Field(default=5)
    pending_flush_max_entries: int = Field(default=50_000)

    audit_sink: str = Field(default="verbose_logger")
    audit_retention_days: int = Field(default=30)
    audit_queue_max_entries: int = Field(default=50_000)

    tag_verdict_cap: int = Field(default=10)

    require_noeviction_for_fail_closed: bool = Field(default=True)

    @classmethod
    def from_env(cls) -> "GovernorV2Config":
        return cls()
