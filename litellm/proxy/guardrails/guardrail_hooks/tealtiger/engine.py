"""
TealEngine: deterministic, dependency-free PII / cost / tool-auth policy
evaluation. No network calls in the governance path.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from types import MappingProxyType
from typing import Final

from .patterns import default_patterns


class PolicyMode(str, Enum):
    ENFORCE = "ENFORCE"
    MONITOR = "MONITOR"


class Action(str, Enum):
    ALLOW = "ALLOW"
    REDACT = "REDACT"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class Decision:
    action: str
    reason_code: str
    correlation_id: str
    redacted_text: str | None = None
    details: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def to_receipt(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "correlation_id": self.correlation_id,
                "action": self.action,
                "reason_code": self.reason_code,
                "details": self.details,
            }
        )


class TealEngine:
    def __init__(
        self,
        policies: Sequence[Mapping[str, object]],
        mode: PolicyMode = PolicyMode.ENFORCE,
    ) -> None:
        self.policies = policies
        self.mode = mode
        self.patterns = default_patterns()
        self._lock = threading.Lock()
        self._spend_by_day: dict[str, float] = {}  # mutable-ok: live counter under lock

        self._pii_policy = next((p for p in policies if p["type"] == "pii"), None)
        self._cost_policy = next((p for p in policies if p["type"] == "cost"), None)
        self._tool_policy = next(
            (p for p in policies if p["type"] == "tool_auth"), None
        )

    def evaluate_text(self, text: str) -> Decision:
        correlation_id: Final = str(uuid.uuid4())
        if not text or not self._pii_policy:
            return Decision(Action.ALLOW.value, "NO_VIOLATIONS", correlation_id)

        findings: Final = tuple(
            name for name, pat in self.patterns.items() if pat.search(text)
        )
        if not findings:
            return Decision(Action.ALLOW.value, "NO_VIOLATIONS", correlation_id)

        pii_action: Final = self._pii_policy.get("action", "REDACT")
        if pii_action == "BLOCK" and self.mode == PolicyMode.ENFORCE:
            return Decision(
                Action.BLOCK.value,
                f"PII_DETECTED:{','.join(findings)}",
                correlation_id,
                details=MappingProxyType({"types": findings}),
            )

        if self.mode == PolicyMode.MONITOR:
            return Decision(
                Action.ALLOW.value,
                f"PII_DETECTED:{','.join(findings)}",
                correlation_id,
                details=MappingProxyType({"types": findings}),
            )

        redacted: Final = self._redact(text, findings)
        return Decision(
            Action.REDACT.value,
            f"PII_DETECTED:{','.join(findings)}",
            correlation_id,
            redacted_text=redacted,
            details=MappingProxyType({"types": findings}),
        )

    def _redact(self, text: str, findings: Sequence[str]) -> str:
        result = text  # rebind-ok: progressively transformed across findings, one sub() pass per match
        for name in findings:
            result = self.patterns[name].sub(f"[REDACTED:{name.upper()}]", result)
        return result

    def check_tool(self, tool_name: str) -> bool:
        if not self._tool_policy:
            return True
        raw_blocklist: Final = self._tool_policy.get("blocklist")
        blocklist: Final = (
            raw_blocklist if isinstance(raw_blocklist, (list, tuple)) else ()
        )
        if tool_name in blocklist:
            return False
        raw_allowlist: Final = self._tool_policy.get("allowlist")
        if raw_allowlist is not None:
            allowlist: Final = (
                raw_allowlist if isinstance(raw_allowlist, (list, tuple)) else ()
            )
            return tool_name in allowlist
        return True

    def check_budget(self, session_id: str = "default") -> tuple[bool, float, float]:
        if not self._cost_policy:
            return False, 0.0, 0.0
        raw_limit: Final = self._cost_policy.get("daily_limit_usd")
        if not isinstance(raw_limit, (int, float)):
            return False, 0.0, 0.0
        limit: Final[float] = float(raw_limit)
        today: Final = date.today().isoformat()  # noqa: DTZ011  # daily budget boundary is intentionally local-server-time, not UTC
        with self._lock:
            spent: Final = self._spend_by_day.get(today, 0.0)
        return spent >= limit, spent, limit

    def track_cost(self, tokens: int, cost_per_1k_tokens: float = 0.002) -> float:
        cost: Final = (tokens / 1000.0) * cost_per_1k_tokens
        today: Final = date.today().isoformat()  # noqa: DTZ011  # daily budget boundary is intentionally local-server-time, not UTC
        with self._lock:
            self._spend_by_day[today] = self._spend_by_day.get(today, 0.0) + cost
        return cost
