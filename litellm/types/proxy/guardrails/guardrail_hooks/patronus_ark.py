from collections.abc import Mapping
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .base import GuardrailConfigModel

Source: TypeAlias = Literal["user_input", "tool_call", "tool_result"]
Action: TypeAlias = Literal["off", "block", "redact"]


class ArkWolfConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: Literal["off", "local", "api"] = "off"
    bundle_dir: str | None = None
    threshold: float = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)
    precision: Literal["auto", "fp16", "int8_int4"] = "fp16"
    window_tokens: int = Field(default=256, ge=3)
    overlap_tokens: int = Field(default=0, ge=0)
    batch_size: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_local(self) -> "ArkWolfConfig":
        if self.backend == "local" and not self.bundle_dir:
            raise ValueError("Local Wolf requires bundle_dir")
        if self.overlap_tokens >= self.window_tokens - 2:
            raise ValueError("Wolf overlap must be smaller than the content window")
        return self


class ArkConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pii: Action = "redact"
    dlp: Action = "redact"
    native_pi: bool = True
    wolf: ArkWolfConfig = Field(default_factory=ArkWolfConfig)
    pii_sources: tuple[Source, ...] = ("user_input", "tool_call", "tool_result")
    dlp_sources: tuple[Source, ...] = ("user_input", "tool_call", "tool_result")
    pi_sources: tuple[Source, ...] = ("user_input", "tool_result")
    pii_mode: Literal["strict", "balanced", "recall"] = "balanced"
    dlp_mode: Literal["strict", "balanced", "recall"] = "balanced"
    rules: Mapping[str, bool] = Field(default_factory=dict)
    max_input_bytes: int = Field(default=1_048_576, gt=0)
    max_findings: int = Field(default=10_000, gt=0)
    max_queued: int = Field(default=8, ge=0)
    queue_timeout: float = Field(default=0.5, gt=0, allow_inf_nan=False)


class PatronusArkConfigModel(GuardrailConfigModel[BaseModel]):
    model_config = ConfigDict(frozen=False)
    ark: ArkConfig | None = Field(default=None, description="Patronus Ark local checks and Wolf Defender settings")

    @staticmethod
    def ui_friendly_name() -> str:
        return "Patronus Ark"


class WolfFinalResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    class_name: Literal["benign", "attack"]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


class WolfInjectionResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    level: Literal["L3"]
    final_result: WolfFinalResult


class WolfCompletion(BaseModel):
    model_config = ConfigDict(frozen=True)
    state: Literal["complete", "degraded", "failed"]
    failures: tuple[object, ...] = ()


class WolfJob(BaseModel):
    model_config = ConfigDict(frozen=True)
    job_id: str
    status: Literal["running", "completed", "failed"]
    completion: WolfCompletion | None = None
    categories: Mapping[str, WolfInjectionResult] = Field(default_factory=dict)


class WolfSubmission(BaseModel):
    model_config = ConfigDict(frozen=True)
    jobs: tuple[WolfJob, ...] = Field(min_length=1, max_length=1)
