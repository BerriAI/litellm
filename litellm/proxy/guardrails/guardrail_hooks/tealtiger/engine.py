"""
TealEngine: deterministic, dependency-free PII / cost / tool-auth policy evaluation.
No network calls in the governance path. Every check returns a Decision value.
"""

import math
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import reduce
from types import MappingProxyType
from typing import Final

from litellm.types.proxy.guardrails.guardrail_hooks.tealtiger import (
    CostPolicy,
    PiiPolicy,
    Policy,
    PolicyMode,
    ToolAuthPolicy,
)

from .budget import BudgetLedger
from .decision import NO_VIOLATIONS, Action, Decision, RiskLevel
from .patterns import CRITICAL_PATTERNS, resolve_scope
from .request import Identity

_CHARS_PER_TOKEN: Final = 4
_NO_DETAILS: Final[Mapping[str, object]] = MappingProxyType({})
_NO_PATTERNS: Final[Mapping[str, re.Pattern[str]]] = MappingProxyType({})


def estimate_tokens(texts: Sequence[str]) -> int:
    """Rough token estimate (about 4 characters per token). apply_guardrail is not handed usage data."""
    return math.ceil(sum(len(text) for text in texts) / _CHARS_PER_TOKEN)


def _estimate_cost_usd(policy: CostPolicy, texts: Sequence[str]) -> float:
    return estimate_tokens(texts) / 1000.0 * policy.cost_per_1k_tokens


class TealEngine:
    def __init__(
        self,
        policies: Sequence[Policy],
        mode: PolicyMode = PolicyMode.ENFORCE,
        ledger: BudgetLedger | None = None,
    ) -> None:
        self.mode: Final = mode
        self._ledger: Final = ledger or BudgetLedger()
        self._pii: Final = next((p for p in policies if isinstance(p, PiiPolicy)), None)
        self._cost: Final = next((p for p in policies if isinstance(p, CostPolicy)), None)
        self._tool: Final = next((p for p in policies if isinstance(p, ToolAuthPolicy)), None)
        self._patterns: Final[Mapping[str, re.Pattern[str]]] = (
            resolve_scope(self._pii.patterns) if self._pii else _NO_PATTERNS
        )

    def check_text(self, text: str) -> Decision:
        if not text or self._pii is None:
            return self._decide(Action.ALLOW, NO_VIOLATIONS, RiskLevel.NONE)
        findings: Final = tuple(name for name, pattern in self._patterns.items() if pattern.search(text))
        if not findings:
            return self._decide(Action.ALLOW, NO_VIOLATIONS, RiskLevel.NONE)
        risk: Final = RiskLevel.CRITICAL if any(f in CRITICAL_PATTERNS for f in findings) else RiskLevel.MEDIUM
        reason: Final = f"PII_DETECTED:{','.join(findings)}"
        details: Final[Mapping[str, object]] = {"types": findings}
        if self._pii.action == "BLOCK":
            return self._decide(Action.BLOCK, reason, risk, details=details)
        return self._decide(Action.REDACT, reason, risk, text=self._redact(text, findings), details=details)

    def check_tool(self, tool_name: str) -> Decision:
        policy: Final = self._tool
        if policy is None:
            return self._decide(Action.ALLOW, NO_VIOLATIONS, RiskLevel.NONE)
        allowed: Final = tool_name not in policy.blocklist and (
            policy.allowlist is None or tool_name in policy.allowlist
        )
        if allowed:
            return self._decide(Action.ALLOW, NO_VIOLATIONS, RiskLevel.NONE)
        return self._decide(Action.BLOCK, "TOOL_NOT_ALLOWLISTED", RiskLevel.HIGH, details={"tool_name": tool_name})

    def check_budget(self, identity: Identity, texts: Sequence[str]) -> Decision:
        policy: Final = self._cost
        if policy is None:
            return self._decide(Action.ALLOW, NO_VIOLATIONS, RiskLevel.NONE)
        daily, session = self._ledger.spent(identity.user, identity.session)
        candidates: Final = (
            ("REQUEST_BUDGET_EXCEEDED", _estimate_cost_usd(policy, texts), policy.request_limit_usd),
            ("SESSION_BUDGET_EXCEEDED", session, policy.session_limit_usd),
            ("DAILY_BUDGET_EXCEEDED", daily, policy.daily_limit_usd),
        )
        exceeded: Final = next(
            ((reason, spent, limit) for reason, spent, limit in candidates if limit is not None and spent >= limit),
            None,
        )
        if exceeded is None:
            return self._decide(Action.ALLOW, NO_VIOLATIONS, RiskLevel.NONE)
        reason, spent, limit = exceeded
        return self._decide(
            Action.BLOCK,
            reason,
            RiskLevel.HIGH,
            details={"spent_usd": spent, "limit_usd": limit, "user": identity.user},
        )

    def record_usage(self, identity: Identity, texts: Sequence[str]) -> None:
        """Accrue the estimated spend of text that passed through the guardrail."""
        if self._cost is not None:
            self._ledger.record(identity.user, identity.session, _estimate_cost_usd(self._cost, texts))

    def _redact(self, text: str, findings: Sequence[str]) -> str:
        return reduce(
            lambda redacted, name: self._patterns[name].sub(f"[REDACTED:{name.upper()}]", redacted), findings, text
        )

    def _decide(
        self,
        action: Action,
        reason_code: str,
        risk_level: RiskLevel,
        text: str | None = None,
        details: Mapping[str, object] = _NO_DETAILS,
    ) -> Decision:
        decision: Final = Decision(
            action, reason_code, risk_level, str(uuid.uuid4()), text, MappingProxyType(dict(details))
        )
        return self._apply_mode(decision)

    def _apply_mode(self, decision: Decision) -> Decision:
        """MONITOR mode never blocks or rewrites: the decision is downgraded to ALLOW and the receipt records what
        ENFORCE mode would have done."""
        if self.mode is PolicyMode.ENFORCE or decision.action is Action.ALLOW:
            return decision
        return replace(
            decision,
            action=Action.ALLOW,
            redacted_text=None,
            details=MappingProxyType({**decision.details, "monitor_would": decision.action.value}),
        )
