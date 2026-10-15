"""Ternary FOCUS export admin endpoint types for LiteLLM Proxy"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Final, Literal, TypeAlias

from pydantic import ConfigDict, Field, JsonValue, field_validator, model_validator
from typing_extensions import Self

from litellm.types.llms.base import LiteLLMBaseModel

TERNARY_DEFAULT_BASE_URL: Final = "https://api.ternary.app"

TernaryExportFrequency: TypeAlias = Literal["daily", "interval"]


def _reject_blank(value: str | None) -> str | None:
    if value is not None and not value.strip():
        raise ValueError("must be a non-empty string")
    return value


class _Frozen(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)


class TernarySettings(_Frozen):
    """Ternary export settings. The init request body, and the stored row once the API key is decrypted"""

    api_key: str = Field(..., description="Ternary AI Gateway connection API key")
    connection_id: str = Field(..., description="Ternary AI Gateway connection ID")
    ternary_base_url: str = Field(default=TERNARY_DEFAULT_BASE_URL, description="Ternary API base URL")
    frequency: TernaryExportFrequency = Field(
        default="daily", description="Export cadence. 'interval' is meant for short setup-validation loops"
    )
    interval_seconds: int | None = Field(
        default=None, description="Seconds between exports when frequency is 'interval'"
    )

    _reject_blank_fields = field_validator("api_key", "connection_id", "ternary_base_url")(_reject_blank)

    @model_validator(mode="after")
    def interval_needs_seconds(self) -> Self:
        if self.frequency == "interval" and (self.interval_seconds is None or self.interval_seconds <= 0):
            raise ValueError("interval_seconds must be a positive integer when frequency is 'interval'")
        return self


class TernarySettingsUpdate(_Frozen):
    """Request model for updating Ternary export settings. Omitted fields keep their stored value"""

    api_key: str | None = Field(default=None, description="New Ternary connection API key")
    connection_id: str | None = Field(default=None, description="New Ternary connection ID")
    ternary_base_url: str | None = Field(default=None, description="New Ternary API base URL")
    frequency: TernaryExportFrequency | None = Field(default=None, description="New export cadence")
    interval_seconds: int | None = Field(default=None, description="New seconds between exports for 'interval'")

    _reject_blank_fields = field_validator("api_key", "connection_id", "ternary_base_url")(_reject_blank)


class TernarySettingsView(_Frozen):
    """Response model for viewing Ternary export settings with the API key masked"""

    api_key_masked: str | None = Field(default=None, description="Masked API key")
    connection_id: str | None = Field(default=None, description="Ternary connection ID")
    ternary_base_url: str | None = Field(default=None, description="Ternary API base URL")
    frequency: TernaryExportFrequency | None = Field(default=None, description="Export cadence")
    interval_seconds: int | None = Field(default=None, description="Seconds between exports for 'interval'")
    status: str | None = Field(default=None, description="Configuration status")


class TernaryMessageResponse(_Frozen):
    """Response model for Ternary settings writes"""

    message: str
    status: str


class TernaryDryRunRequest(_Frozen):
    """Request model for a Ternary dry run, which previews the export locally without contacting Ternary"""

    limit: int = Field(default=500, gt=0, description="Maximum number of usage rows to preview")


class TernaryDryRunSummary(_Frozen):
    total_records: int
    estimated_spend: float = Field(description="LiteLLM's estimated spend for the previewed rows, not billed cost")
    total_tokens: int
    unique_teams: int
    unique_models: int
    days: Sequence[str] = Field(description="UTC days the previewed rows fall on, one upload each")


class TernaryDryRunResponse(_Frozen):
    message: str
    status: str
    usage_data: Sequence[Mapping[str, JsonValue]] = Field(description="Sample of LiteLLM daily spend rows")
    normalized_data: Sequence[Mapping[str, JsonValue]] = Field(
        description="The same rows as FOCUS records, including token counts in Tags, as Ternary would receive them"
    )
    summary: TernaryDryRunSummary


class TernaryExportRequest(_Frozen):
    """Request model for a manual Ternary export. With no bounds, exports the scheduled window"""

    limit: int | None = Field(default=None, gt=0, description="Optional cap on the number of usage rows to export")
    start_time_utc: datetime | None = Field(default=None, description="Start of the export window in UTC")
    end_time_utc: datetime | None = Field(default=None, description="End of the export window in UTC")

    @field_validator("start_time_utc", "end_time_utc")
    @classmethod
    def assume_utc(cls, value: datetime | None) -> datetime | None:
        if value is None or value.tzinfo is not None:
            return value
        return value.replace(tzinfo=timezone.utc)

    @model_validator(mode="after")
    def bounds_come_together(self) -> Self:
        if (self.start_time_utc is None) != (self.end_time_utc is None):
            raise ValueError("start_time_utc and end_time_utc must be provided together")
        if (
            self.start_time_utc is not None
            and self.end_time_utc is not None
            and self.start_time_utc >= self.end_time_utc
        ):
            raise ValueError("start_time_utc must be before end_time_utc")
        return self


class TernaryExportResponse(_Frozen):
    message: str
    status: str
    start_time_utc: datetime
    end_time_utc: datetime
