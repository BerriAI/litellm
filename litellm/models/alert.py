from datetime import date as Date
from datetime import datetime, timezone
from typing import Annotated, Final, Literal, Protocol, TypeAlias

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

AlertType: TypeAlias = Literal["spend", "traffic", "latency", "reliability", "model_deprecation"]
AlertFilter: TypeAlias = AlertType | Literal["performance"]
AlertStatus: TypeAlias = Literal["open", "resolved", "dismissed"]
AlertResourceType: TypeAlias = Literal["gateway", "key", "team", "user", "model"]
MAX_ALERT_ID: Final = 2**63 - 1


def _validate_alert_id(value: str) -> str:
    if int(value) > MAX_ALERT_ID:
        raise ValueError("Alert ID exceeds the supported range")
    return value


AlertID: TypeAlias = Annotated[
    str, Field(pattern=r"^[1-9][0-9]{0,18}$", max_length=19), AfterValidator(_validate_alert_id)
]


class MetricEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    value: float = Field(ge=0, allow_inf_nan=False, description="Observed metric value in the alert window.")
    baseline: float = Field(ge=0, allow_inf_nan=False, description="Comparison value in the same unit.")
    unit: Literal["usd", "requests", "milliseconds", "ratio"] = Field(description="Unit for both values.")
    sample_size: int = Field(ge=0, le=MAX_ALERT_ID, description="Number of requests in the alert window.")


class DeprecationEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    deprecation_date: Date = Field(description="Date the affected model retires.")
    replacement_model: str | None = Field(
        default=None, min_length=1, max_length=1024, description="Suggested replacement."
    )


class AlertDetection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: AlertType = Field(description="Built-in alert category.")
    resource_type: AlertResourceType = Field(description="Kind of affected resource.")
    resource_id: str = Field(
        min_length=1,
        max_length=1024,
        description="Resource ID. Whole-gateway alerts use 'gateway'. Keys use their SHA-256 hash, never a secret.",
    )
    window_start: AwareDatetime = Field(
        description="Inclusive start of the observed window, normalized to UTC milliseconds."
    )
    window_end: AwareDatetime = Field(
        description="Exclusive end of the observed window, normalized to UTC milliseconds."
    )
    evidence: MetricEvidence | DeprecationEvidence = Field(description="Small saved snapshot explaining the alert.")

    @field_validator("window_start", "window_end")
    @classmethod
    def normalize_window(cls, value: datetime) -> datetime:
        return value.astimezone(timezone.utc).replace(microsecond=value.microsecond // 1000 * 1000)

    @model_validator(mode="after")
    def validate_detection(self) -> Self:
        if self.window_end <= self.window_start:
            raise ValueError("window_end must be after window_start")
        if self.resource_type == "gateway" and self.resource_id != "gateway":
            raise ValueError("Whole-gateway alerts must use resource_id 'gateway'")
        if self.resource_type == "key" and (
            len(self.resource_id) != 64 or any(c not in "0123456789abcdef" for c in self.resource_id)
        ):
            raise ValueError("Key resource IDs must be SHA-256 hashes")
        if self.type == "model_deprecation":
            if not isinstance(self.evidence, DeprecationEvidence) or self.resource_type != "model":
                raise ValueError("Model deprecations require a model resource and deprecation evidence")
            return self
        expected_unit: Final = {
            "spend": "usd",
            "traffic": "requests",
            "latency": "milliseconds",
            "reliability": "ratio",
        }[self.type]
        if not isinstance(self.evidence, MetricEvidence) or self.evidence.unit != expected_unit:
            raise ValueError("Evidence unit must match the alert type")
        if self.type == "reliability" and (self.evidence.value > 1 or self.evidence.baseline > 1):
            raise ValueError("Error ratios must be between zero and one")
        return self

    def identity(self) -> tuple[str, ...]:
        return (
            self.type,
            self.resource_type,
            self.resource_id,
            self.window_start.astimezone(timezone.utc).isoformat(),
            self.window_end.astimezone(timezone.utc).isoformat(),
        )


class Alert(AlertDetection):
    id: AlertID = Field(description="Opaque alert ID, also used as a pagination cursor.")
    status: AlertStatus = Field(description="Current review status.")
    created_at: AwareDatetime = Field(description="When the alert was first saved.")
    updated_at: AwareDatetime = Field(description="When its status last changed.")
    updated_by: str | None = Field(default=None, max_length=1024, description="User who last changed the status.")


class AlertHistoryPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    date: Date = Field(description="UTC day represented by this value.")
    value: float = Field(allow_inf_nan=False, description="Recorded daily value in the alert metric's unit.")


class AlertView(Alert):
    resource_name: str | None = Field(default=None, max_length=1024, description="Current resource display name.")
    history: tuple[AlertHistoryPoint, ...] | None = Field(
        default=None,
        max_length=8,
        description="Current Usage values for the seven baseline days and observed day. Null if unavailable.",
    )


class AlertUpdate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: AlertStatus = Field(description="Mark resolved, dismiss, or reopen the alert.")


class AlertPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: tuple[Alert, ...] = Field(description="Alerts ordered newest first.")
    next_cursor: AlertID | None = Field(
        default=None, description="Pass as cursor to fetch the next page, or null at the end."
    )


class AlertViewPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: tuple[AlertView, ...] = Field(description="Alerts ordered newest first, with current Usage history.")
    next_cursor: AlertID | None = Field(default=None, description="Cursor for the next page, or null at the end.")


class AlertStore(Protocol):
    async def record(self, detection: AlertDetection) -> Alert: ...

    async def list(
        self, status: AlertStatus, limit: int, cursor: int | None, alert_type: AlertFilter | None = None
    ) -> AlertPage: ...

    async def enrich(self, alerts: tuple[Alert, ...]) -> tuple[AlertView, ...]: ...

    async def get(self, alert_id: int) -> Alert | None: ...

    async def set_status(self, alert_id: int, status: AlertStatus, actor: str) -> Alert | None: ...
