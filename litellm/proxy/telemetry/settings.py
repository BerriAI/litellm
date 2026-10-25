"""Environment policy for telemetry: the veto, the pinned groups, and which variables an admin should be told about"""

from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Final

from pydantic import AfterValidator, Field, HttpUrl, ValidationError, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm.telemetry.consent import OFF, TelemetryConsent, parse_consent

ENV_PREFIX: Final = "LITELLM_TELEMETRY_"
FLUSH_INTERVAL_SECONDS: Final = 300.0


class TelemetryDestination(str, Enum):
    LOCAL_TABLE = "local_table"
    HTTP = "http"


def _pinned_consent(groups: str) -> TelemetryConsent:
    consent: Final = parse_consent(groups.split(","))
    if not isinstance(consent, TelemetryConsent):
        raise ValueError(consent.message())  # noqa: TRY004  # pydantic only turns ValueError into a ValidationError
    return consent


def _valid_groups(groups: str | None) -> str | None:
    if groups is not None:
        _ = _pinned_consent(groups)
    return groups


class TelemetrySettings(BaseSettings):
    """``LITELLM_TELEMETRY_*`` env vars"""

    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, case_sensitive=False, extra="ignore", frozen=True)

    disabled: bool = False
    groups: Annotated[str | None, AfterValidator(_valid_groups)] = None
    endpoint: HttpUrl | None = None
    destination: TelemetryDestination | None = None
    settle_timeout_seconds: float = Field(default=2.0, ge=0)
    retention_days: int = Field(default=30, gt=0)

    @field_validator("destination")
    @classmethod
    def _http_needs_an_endpoint(
        cls, destination: TelemetryDestination | None, info: ValidationInfo
    ) -> TelemetryDestination | None:
        if destination is TelemetryDestination.HTTP and info.data.get("endpoint") is None:
            raise ValueError(f"http needs {ENV_PREFIX}ENDPOINT")
        return destination


def load_settings() -> TelemetrySettings | ValidationError:
    try:
        return TelemetrySettings()
    except ValidationError as e:
        return e


def describe_errors(error: ValidationError) -> str:
    """Field names and messages only: the rejected value of ``endpoint`` can carry a token"""
    return "; ".join(
        f"{ENV_PREFIX}{'.'.join(map(str, detail['loc'])).upper()}: {detail['msg']}"
        for detail in error.errors(include_input=False, include_url=False)
    )


@dataclass(frozen=True, slots=True)
class EnvPolicy:
    vetoed: bool
    pinned: TelemetryConsent | None
    destination: TelemetryDestination | None
    set_variables: tuple[str, ...]

    def effective(self, stored: TelemetryConsent | None) -> TelemetryConsent:
        if self.vetoed:
            return OFF
        if self.pinned is not None:
            return self.pinned
        return stored if stored is not None else OFF

    def effective_destination(self, stored: TelemetryDestination | None) -> TelemetryDestination | None:
        return self.destination if self.destination is not None else stored

    @property
    def locked_off(self) -> bool:
        return self.vetoed or self.pinned == OFF


def env_policy(settings: TelemetrySettings) -> EnvPolicy:
    return EnvPolicy(
        vetoed=settings.disabled,
        pinned=_pinned_consent(settings.groups) if settings.groups is not None else None,
        destination=settings.destination,
        set_variables=tuple(sorted(f"{ENV_PREFIX}{name.upper()}" for name in settings.model_fields_set)),
    )
