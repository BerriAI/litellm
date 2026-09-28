from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_POOL_NAME: Final = "default"
RESERVED_WORKLOAD_CLASS_NAMES: Final = frozenset({DEFAULT_POOL_NAME, "default_pool"})
FAIRNESS_SETTINGS_KEY: Final = "fairness_settings"
MAX_QUEUE_WAIT_HEADER: Final = "x-litellm-max-queue-wait"
MAX_QUEUE_WAIT_SECONDS: Final = 600.0
SHARE_ROUNDING_TOLERANCE: Final = 1e-9

FairnessRejectReason = Literal["capacity_exhausted", "queue_full", "queue_deadline_exceeded", "client_disconnected"]


class WorkloadClass(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    reserved_share: float = Field(ge=0.0, le=1.0, description="fraction of model RPM/TPM reserved once saturated")
    max_queue_wait_seconds: float = Field(
        default=0.0,
        ge=0.0,
        le=MAX_QUEUE_WAIT_SECONDS,
        description="how long an over-limit request may wait; 0 rejects immediately",
    )
    description: str | None = Field(default=None, max_length=256)


class FairnessSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    workload_classes: tuple[WorkloadClass, ...] = ()
    default_reserved_share: float = Field(default=0.25, ge=0.0, le=1.0)
    default_max_queue_wait_seconds: float = Field(default=0.0, ge=0.0, le=MAX_QUEUE_WAIT_SECONDS)
    saturation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    saturation_check_cache_ttl: int = Field(default=60, ge=0, le=3600)
    max_queue_depth_per_class: int = Field(default=100, ge=1, le=100_000)
    queue_poll_interval_seconds: float = Field(default=0.1, gt=0.0, le=5.0)

    @model_validator(mode="after")
    def _consistent_workload_classes(self) -> "FairnessSettings":
        names: Final = tuple(workload_class.name for workload_class in self.workload_classes)
        if len(names) != len(frozenset(names)):
            raise ValueError("workload class names must be unique")
        if RESERVED_WORKLOAD_CLASS_NAMES & frozenset(names):
            raise ValueError(
                f"workload class names {sorted(RESERVED_WORKLOAD_CLASS_NAMES)!r} are reserved for keys and teams "
                "without a workload class"
            )
        total_share: Final = (
            sum(workload_class.reserved_share for workload_class in self.workload_classes) + self.default_reserved_share
        )
        if total_share > 1.0 + SHARE_ROUNDING_TOLERANCE:
            raise ValueError(
                f"reserved shares add up to {total_share:.0%} including the {self.default_reserved_share:.0%} default "
                "pool; keep the total at or below 100%"
            )
        return self

    def max_queue_wait_for(self, class_name: str | None) -> float:
        for workload_class in self.workload_classes:
            if workload_class.name == class_name:
                return workload_class.max_queue_wait_seconds
        return self.default_max_queue_wait_seconds

    def reserved_shares(self) -> Mapping[str, float]:
        return MappingProxyType(
            {workload_class.name: workload_class.reserved_share for workload_class in self.workload_classes}
        )


class FairnessSettingsResponse(BaseModel):
    settings: FairnessSettings
    persisted: bool


class WorkloadClassStatus(BaseModel):
    name: str
    reserved_share: float
    reserved_rpm: int | None
    reserved_tpm: int | None
    current_requests: int
    current_tokens: int
    queue_depth: int
    max_queue_wait_seconds: float
    queued_total: int
    admitted_after_wait_total: int
    rejected_capacity_total: int
    rejected_queue_full_total: int
    rejected_deadline_total: int
    disconnected_total: int
    avg_queue_wait_seconds: float


class ModelFairnessStatus(BaseModel):
    model_group: str
    rpm: int | None
    tpm: int | None
    saturation: float
    enforcing_reservations: bool
    current_requests: int
    current_tokens: int
    classes: tuple[WorkloadClassStatus, ...]


class FairnessStatusResponse(BaseModel):
    enabled: bool
    limiter_active: bool
    saturation_threshold: float
    window_size_seconds: int
    stats_window_seconds: int
    models: tuple[ModelFairnessStatus, ...]
