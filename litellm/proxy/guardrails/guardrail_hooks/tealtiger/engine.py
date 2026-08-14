"""
TealEngine: deterministic, dependency-free PII / cost / tool-auth policy
evaluation. No network calls in the governance path.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .patterns import default_patterns


class PolicyMode(str, Enum):
    ENFORCE = "ENFORCE"
    MONITOR = "MONITOR"


class Action(str, Enum):
    ALLOW = "ALLOW"
    REDACT = "REDACT"
    BLOCK = "BLOCK"


@dataclass
class Decision:
    action: str
    reason_code: str
    correlation_id: str
    redacted_text: str | None = None
    details: dict = field(default_factory=dict)

    def to_receipt(self) -> dict:
        return {
            "correlation_id": self.correlation_id,
            "action": self.action,
            "reason_code": self.reason_code,
            "details": self.details,
        }


class TealEngine:
    def __init__(self, policies: list[dict], mode: PolicyMode = PolicyMode.ENFORCE):
        self.policies = policies
        self.mode = mode
        self.patterns = default_patterns()
        self._lock = threading.Lock()
        self._spend_by_day: dict[str, float] = {}

        self._pii_policy = next((p for p in policies if p["type"] == "pii"), None)
        self._cost_policy = next((p for p in policies if p["type"] == "cost"), None)
        self._tool_policy = next((p for p in policies if p["type"] == "tool_auth"), None)

    def evaluate_text(self, text: str) -> Decision:
        correlation_id = str(uuid.uuid4())
        if not text or not self._pii_policy:
            return Decision(Action.ALLOW.value, "NO_VIOLATIONS", correlation_id)

        findings = [name for name, pat in self.patterns.items() if pat.search(text)]
        if not findings:
            return Decision(Action.ALLOW.value, "NO_VIOLATIONS", correlation_id)

        pii_action = self._pii_policy.get("action", "REDACT")
        if pii_action == "BLOCK" and self.mode == PolicyMode.ENFORCE:
            return Decision(
                Action.BLOCK.value,
                f"PII_DETECTED:{','.join(findings)}",
                correlation_id,
                details={"types": findings},
            )

        if self.mode == PolicyMode.MONITOR:
            return Decision(
                Action.ALLOW.value,
                f"PII_DETECTED:{','.join(findings)}",
                correlation_id,
                details={"types": findings},
            )

        redacted = text
        for name in findings:
            redacted = self.patterns[name].sub(f"[REDACTED:{name.upper()}]", redacted)
        return Decision(
            Action.REDACT.value,
            f"PII_DETECTED:{','.join(findings)}",
            correlation_id,
            redacted_text=redacted,
            details={"types": findings},
        )

    def check_tool(self, tool_name: str) -> bool:
        if not self._tool_policy:
            return True
        allowlist = self._tool_policy.get("allowlist")
        blocklist = self._tool_policy.get("blocklist")
        if blocklist and tool_name in blocklist:
            return False
        if allowlist is not None:
            return tool_name in allowlist
        return True

    def check_budget(self, session_id: str = "default") -> tuple[bool, float, float]:
        if not self._cost_policy:
            return False, 0.0, 0.0
        limit = self._cost_policy.get("daily_limit_usd")
        if limit is None:
            return False, 0.0, 0.0
        today = datetime.now().astimezone().date().isoformat()
        with self._lock:
            spent = self._spend_by_day.get(today, 0.0)
        return spent >= limit, spent, limit

    def track_cost(self, tokens: int, cost_per_1k_tokens: float = 0.002) -> float:
        cost = (tokens / 1000.0) * cost_per_1k_tokens
        today = datetime.now().astimezone().date().isoformat()
        with self._lock:
            self._spend_by_day[today] = self._spend_by_day.get(today, 0.0) + cost
        return cost
