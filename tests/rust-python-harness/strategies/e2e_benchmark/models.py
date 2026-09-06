from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Backend = Literal["python", "rust"]
Route = Literal["ocr", "aocr"]
Phase = Literal["timing", "memory"]
Profile = Literal["small", "request_medium", "request_large", "response_medium", "response_large"]


class BenchmarkModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Options(BenchmarkModel):
    iterations: int = Field(default=100, ge=1)
    warmup: int = Field(default=10, ge=1)
    repeats: int = Field(default=4, ge=1)
    min_time: float = Field(default=1, ge=0, allow_inf_nan=False)
    profiles: tuple[Profile, ...] = ("small", "request_medium", "request_large", "response_medium", "response_large")
    routes: tuple[Route, ...] = ("ocr", "aocr")
    timeout: float = Field(default=120, gt=0, allow_inf_nan=False)
    sample_interval_ms: float = Field(default=5, ge=1, allow_inf_nan=False)
    output: str | None = None


class Invocation(BenchmarkModel):
    model: str
    document_url: str
    route: Route
    provider_url: str
    iterations: int
    warmup: int
    phase: Phase
    min_time: float = Field(default=0, ge=0, allow_inf_nan=False)


class Ready(BenchmarkModel):
    response_digest: str
    python_version: str
    native_sha256: str | None


class Timing(BenchmarkModel):
    latency_ms: tuple[float, ...]
    cpu_ms: float
    elapsed_ms: float


class Memory(BenchmarkModel):
    baseline_rss_bytes: int
    sampled_peak_rss_bytes: int
    retained_rss_bytes: int
    samples: int


class Measurement(BenchmarkModel):
    backend: Backend
    repeat: int
    profile: Profile
    route: Route
    document_bytes: int
    response_bytes: int
    response_pages: int
    fixture_sha256: str
    ready: Ready
    timing: Timing
    memory: Memory
