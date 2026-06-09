"""Verdicts and the combined admission decision."""

from dataclasses import dataclass
from typing import Literal, Mapping

FailMode = Literal["closed", "open"]
Status = Literal["admitted", "admitted_degraded", "rejected"]


@dataclass(frozen=True)
class Correlation:
    call_id: str
    trace_id: str | None = None
    span_id: str | None = None


@dataclass(frozen=True)
class RateLimitHeaders:
    limit: int | None
    remaining: int | None
    reset_seconds: int | None
    reason: str | None
    degraded: bool

    def as_dict(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.limit is not None:
            out["x-ratelimit-limit"] = str(self.limit)
        if self.remaining is not None:
            out["x-ratelimit-remaining"] = str(self.remaining)
        if self.reset_seconds is not None:
            out["x-ratelimit-reset"] = str(self.reset_seconds)
        if self.reason is not None:
            out["x-ratelimit-reason"] = self.reason
        if self.degraded:
            out["x-ratelimit-degraded"] = "true"
        return out


@dataclass(frozen=True)
class Verdict:
    policy_id: str
    status: Status
    reason: str | None
    limit: float | None
    observed: float | None
    fail_mode: FailMode
    degraded: bool

    def __post_init__(self) -> None:
        if self.degraded and self.status == "admitted":
            raise ValueError("degraded verdicts cannot be a clean admit")

    @classmethod
    def admitted(
        cls,
        policy_id: str,
        fail_mode: FailMode,
        *,
        limit: float | None = None,
        observed: float | None = None,
    ) -> "Verdict":
        return cls(
            policy_id=policy_id,
            status="admitted",
            reason=None,
            limit=limit,
            observed=observed,
            fail_mode=fail_mode,
            degraded=False,
        )

    @classmethod
    def rejected(
        cls,
        policy_id: str,
        fail_mode: FailMode,
        *,
        reason: str,
        limit: float | None = None,
        observed: float | None = None,
        degraded: bool = False,
    ) -> "Verdict":
        return cls(
            policy_id=policy_id,
            status="rejected",
            reason=reason,
            limit=limit,
            observed=observed,
            fail_mode=fail_mode,
            degraded=degraded,
        )

    @classmethod
    def admitted_degraded(
        cls, policy_id: str, fail_mode: FailMode, *, reason: str
    ) -> "Verdict":
        return cls(
            policy_id=policy_id,
            status="admitted_degraded",
            reason=reason,
            limit=None,
            observed=None,
            fail_mode=fail_mode,
            degraded=True,
        )


@dataclass(frozen=True)
class Decision:
    status: Status
    verdicts: tuple[Verdict, ...]
    request_id: str
    latency_ms: float

    @property
    def rejected_by(self) -> tuple[Verdict, ...]:
        return tuple(v for v in self.verdicts if v.status == "rejected")

    @property
    def degraded(self) -> bool:
        return any(v.degraded for v in self.verdicts)

    def _binding_verdict(self) -> Verdict | None:
        rejected = self.rejected_by
        if rejected:
            return rejected[0]
        degraded = tuple(v for v in self.verdicts if v.status == "admitted_degraded")
        if degraded:
            return degraded[0]
        return self.verdicts[0] if self.verdicts else None

    @property
    def rate_limit_headers(self) -> RateLimitHeaders:
        binding = self._binding_verdict()
        if binding is None:
            return RateLimitHeaders(None, None, None, None, self.degraded)
        remaining: int | None = None
        if binding.limit is not None and binding.observed is not None:
            remaining = max(0, int(binding.limit - binding.observed))
        limit_int = int(binding.limit) if binding.limit is not None else None
        return RateLimitHeaders(
            limit=limit_int,
            remaining=remaining,
            reset_seconds=None,
            reason=binding.reason,
            degraded=self.degraded,
        )

    @property
    def headers(self) -> Mapping[str, str]:
        return self.rate_limit_headers.as_dict()
