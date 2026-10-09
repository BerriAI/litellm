"""Environment policy for telemetry: the veto, the pinned groups, and which variables an admin should be told about"""

from dataclasses import dataclass
from typing import Final

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm.telemetry.consent import OFF, ConsentError, TelemetryConsent, parse_consent

ENV_PREFIX: Final = "LITELLM_TELEMETRY_"


class TelemetrySettings(BaseSettings):
    """``LITELLM_TELEMETRY_*`` env vars"""

    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, case_sensitive=False, extra="ignore", frozen=True)

    disabled: bool = False
    groups: str | None = None
    endpoint: str | None = None
    flush_interval_seconds: float = Field(default=300.0, gt=0)
    settle_timeout_seconds: float = Field(default=2.0, ge=0)


def load_settings() -> TelemetrySettings | ValidationError:
    try:
        return TelemetrySettings()
    except ValidationError as e:
        return e


@dataclass(frozen=True, slots=True)
class EnvPolicy:
    vetoed: bool
    pinned: TelemetryConsent | None
    set_variables: tuple[str, ...]

    def effective(self, stored: TelemetryConsent | None) -> TelemetryConsent:
        if self.vetoed:
            return OFF
        if self.pinned is not None:
            return self.pinned
        return stored if stored is not None else OFF

    @property
    def locked_off(self) -> bool:
        return self.vetoed or self.pinned == OFF


def env_policy(settings: TelemetrySettings) -> EnvPolicy | ConsentError:
    pinned: Final = parse_consent(settings.groups.split(",")) if settings.groups is not None else None
    if pinned is not None and not isinstance(pinned, TelemetryConsent):
        return pinned
    return EnvPolicy(
        vetoed=settings.disabled,
        pinned=pinned,
        set_variables=tuple(sorted(f"{ENV_PREFIX}{name.upper()}" for name in settings.model_fields_set)),
    )
